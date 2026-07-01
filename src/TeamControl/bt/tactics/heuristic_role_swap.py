"""Heuristic role assignment for dynamic BT roles.

The team-level helper ranks robots relative to each other, then assigns one
attacker, required defenders, and supporters. The Coordinator decides whether
to call this module; when the yaml flag is false, static roles remain untouched.
"""
from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.bt.tactics.line_of_sight import (
    LineOfSightResult,
    Point,
    angle_error,
    distance,
    evaluate_line_of_sight,
    face_angle,
    point_to_segment_dist,
)

BallHolder = Literal["yellow", "blue", "own", "opponent", "none", "unknown"]
BT_TUNING_FILENAME = "bt_tuning.yaml"
LEGACY_HEURISTIC_WEIGHT_FILENAME = "heuristic_weight.yaml"


@dataclass(frozen=True)
class RoleTargetCounts:
    """Team-level role bounds for the scorer."""

    attackers: int = 1
    min_defenders: int = 1
    max_defenders: int = 2
    min_supporters: int = 1
    # How many MARKER (man-marking) robots to assign while defending. Default 0
    # = off (behaviour unchanged). Raise it to have the heuristic dynamically
    # man-mark opponents when the opponent has the ball / it's in our half; the
    # Coordinator then matches each marker to a specific opponent. Capped so at
    # least ``min_supporters`` robots remain in support.
    markers: int = 0


@dataclass(frozen=True)
class AttackerScoreWeights:
    """Weights for the attacker score."""

    ball_close: float = 0.40
    approach_quality: float = 0.20
    angle_score: float = 0.14
    opponent_goal_close: float = 0.12
    goal_sight: float = 0.08
    pressure_escape: float = 0.04
    own_has_ball: float = 0.02
    opponent_has_ball_pressure: float = 0.12
    loose_ball_pressure: float = 0.10


@dataclass(frozen=True)
class DefenderScoreWeights:
    """Weights for the defender score. Defaults match the original scorer."""

    own_goal_close: float = 0.28
    ball_close: float = 0.22
    own_lane: float = 0.24
    ball_danger: float = 0.16
    pressure_escape: float = 0.06
    opponent_has_ball: float = 0.04
    # Bonus for being the deepest field player (the "last man" closest to our
    # own goal). Default 0.0 = off; raise it to keep a dedicated sweeper home.
    last_man: float = 0.0


@dataclass(frozen=True)
class SupporterScoreWeights:
    """Weights for the supporter score. Defaults match the original scorer."""

    spacing: float = 0.30
    opponent_goal_close: float = 0.20
    pressure_escape: float = 0.16
    goal_sight: float = 0.14
    not_crowding_ball: float = 0.10
    forward_lane: float = 0.08
    own_has_ball: float = 0.02
    # Bonus for lateral separation from the ball (stretching the play wide,
    # relative to teammates). Default 0.0 = off; raise it for wider support.
    width: float = 0.0


@dataclass(frozen=True)
class MarkerScoreWeights:
    """Weights for the marker score (man-to-man defending).

    A robot scores well as a MARKER when the team is defending (the opponent has
    the ball, or it's in our half) AND there is an opponent close enough to
    shadow. The Coordinator then matches each MARKER robot to a specific
    opponent. Only used when ``role_targets.markers`` > 0.
    """

    # Defending context — markers matter when the opponent threatens.
    opponent_has_ball: float = 0.30
    ball_in_our_half: float = 0.20
    # An opponent is close to this robot (opponent_pressure), i.e. there is
    # someone to mark right here.
    near_opponent: float = 0.30
    # Goal-side of the ball (sitting between the ball and our own goal).
    goal_side: float = 0.12
    # Not the lone ball-chaser — markers shadow outlets, they don't dive at the ball.
    not_crowding_ball: float = 0.08


@dataclass(frozen=True)
class ApproachQualityWeights:
    """Sub-weights for the attacker ``approach_quality`` feature.

    These control how the 0..1 approach-quality score blends being behind the
    ball (relative to the target goal), already facing the ball, and being
    close to it. Defaults match the original hard-coded 0.5 / 0.3 / 0.2 mix, so
    behaviour is unchanged until they are edited.
    """

    behind_ball: float = 0.5
    facing: float = 0.3
    distance: float = 0.2


@dataclass(frozen=True)
class RoleStabilityWeights:
    """Role-stability tuning."""

    current_role_bias: float = 0.08
    cooldown_bias: float = 0.16
    minimum_swap_interval: float = 1.0


@dataclass(frozen=True)
class DefenderStabilityWeights:
    """Extra hysteresis for defenders so the defensive anchor does not churn."""

    min_hold_seconds: float = 3.0
    stay_bias: float = 0.25
    cooldown_bias: float = 0.40
    release_margin: float = 0.12
    allow_attacker_release_margin: float = 0.18


@dataclass(frozen=True)
class ContextScaleWeights:
    """Field-relative context scales used before scoring."""

    goal_sight_clearance_field_scale: float = 0.02
    lane_width_field_scale: float = 0.04
    pressure_radius_field_scale: float = 0.12
    possession_radius_field_scale: float = 0.06


@dataclass(frozen=True)
class DefenderCountWeights:
    """Rules for when the team should allocate an extra defender."""

    add_second_when_opponent_has_ball: bool = True
    add_second_when_ball_in_our_half: bool = True
    minimum_candidates_after_attackers: int = 3


@dataclass(frozen=True)
class RoleHeuristicWeights:
    """All externally tunable role-swap values."""

    attacker: AttackerScoreWeights = field(default_factory=AttackerScoreWeights)
    defender: DefenderScoreWeights = field(default_factory=DefenderScoreWeights)
    supporter: SupporterScoreWeights = field(default_factory=SupporterScoreWeights)
    marker: MarkerScoreWeights = field(default_factory=MarkerScoreWeights)
    stability: RoleStabilityWeights = field(default_factory=RoleStabilityWeights)
    defender_stability: DefenderStabilityWeights = field(default_factory=DefenderStabilityWeights)
    context: ContextScaleWeights = field(default_factory=ContextScaleWeights)
    defender_count: DefenderCountWeights = field(default_factory=DefenderCountWeights)
    role_targets: RoleTargetCounts = field(default_factory=RoleTargetCounts)
    approach: ApproachQualityWeights = field(default_factory=ApproachQualityWeights)


@dataclass(frozen=True)
class RoleSwapContext:
    """Inputs a robot should consider before a role swap."""

    robot_id: int
    current_role: RoleType
    distance_to_ball: float
    confidence_to_score: float
    current_ball_holder: BallHolder
    goal_to_ball_distance: float
    time_since_last_swap: float
    same_role_count: int
    goal_sight: LineOfSightResult
    distance_to_own_goal: float = 0.0
    distance_to_opponent_goal: float = 0.0
    ball_in_our_half: bool = False
    robot_between_ball_and_own_goal: bool = False
    robot_between_ball_and_opponent_goal: bool = False
    angle_to_ball: float = 0.0
    approach_quality: float = 0.0
    lateral_offset_from_ball: float = 0.0
    teammate_spacing: float = math.inf
    opponent_pressure: float = 0.0
    attacker_count: int = 0
    defender_count: int = 0
    supporter_count: int = 0
    goalie_count: int = 0
    minimum_swap_interval: float = 1.0
    role_targets: RoleTargetCounts = field(default_factory=RoleTargetCounts)

    @property
    def has_goal_sight(self) -> bool:
        return self.goal_sight.is_clear

    @property
    def swap_cooldown_active(self) -> bool:
        return self.time_since_last_swap < self.minimum_swap_interval


@dataclass(frozen=True)
class RoleSwapDecision:
    """Future role-swap output.

    For now ``selected_role`` intentionally remains ``current_role``. This
    makes the module safe to import and test before coordinator integration.
    """

    selected_role: RoleType
    should_swap: bool
    context: RoleSwapContext
    reason: str


@dataclass(frozen=True)
class RoleScores:
    """Comparable role scores for one robot."""

    attacker: float
    defender: float
    supporter: float
    marker: float = 0.0


@dataclass(frozen=True)
class RoleAssignmentResult:
    """Team-level role assignment result."""

    roles: dict[int, RoleType]
    contexts: dict[int, RoleSwapContext]
    scores: dict[int, RoleScores]
    reasons: dict[int, str]


def load_role_heuristic_weights(
    config_filename: str | Path = BT_TUNING_FILENAME,
) -> RoleHeuristicWeights:
    """Load heuristic role weights from yaml, preserving defaults for omissions."""

    path = _resolve_utils_config_path(config_filename)

    if not path.exists():
        return RoleHeuristicWeights()

    with open(path, "r") as f:
        raw = yaml.load(f, Loader) or {}

    if not isinstance(raw, MappingABC):
        return RoleHeuristicWeights()

    role_swap = raw.get("role_swap", raw)
    if not isinstance(role_swap, MappingABC):
        return RoleHeuristicWeights()

    return RoleHeuristicWeights(
        attacker=_dataclass_from_section(
            AttackerScoreWeights,
            role_swap.get("attacker"),
        ),
        defender=_dataclass_from_section(
            DefenderScoreWeights,
            role_swap.get("defender"),
        ),
        supporter=_dataclass_from_section(
            SupporterScoreWeights,
            role_swap.get("supporter"),
        ),
        marker=_dataclass_from_section(
            MarkerScoreWeights,
            role_swap.get("marker"),
        ),
        stability=_dataclass_from_section(
            RoleStabilityWeights,
            role_swap.get("stability"),
        ),
        defender_stability=_dataclass_from_section(
            DefenderStabilityWeights,
            role_swap.get("defender_stability"),
        ),
        context=_dataclass_from_section(
            ContextScaleWeights,
            role_swap.get("context"),
        ),
        defender_count=_dataclass_from_section(
            DefenderCountWeights,
            role_swap.get("defender_count"),
        ),
        role_targets=_dataclass_from_section(
            RoleTargetCounts,
            role_swap.get("role_targets"),
        ),
        approach=_dataclass_from_section(
            ApproachQualityWeights,
            role_swap.get("approach"),
        ),
    )


def assign_roles_heuristically(
    snapshot: Snapshot,
    robot_ids: list[int],
    current_roles: Mapping[int, RoleType],
    *,
    base_roles: Mapping[int, RoleType] | None = None,
    time_since_last_swap: Mapping[int, float] | None = None,
    attack_goal: Point = (4.5, 0.0),
    own_goal: Point = (-4.5, 0.0),
    role_targets: RoleTargetCounts | None = None,
    heuristic_weights: RoleHeuristicWeights | None = None,
) -> RoleAssignmentResult:
    """Assign non-goalie roles by relative team ranking.

    The goalie is fixed from ``base_roles``/``current_roles``. Every other
    robot gets scored relative to the available robots on the same tick:
    closest-to-ball, closest-to-goal, spacing, pressure, and goal sight are
    normalized against the current team state rather than absolute thresholds.
    """

    weights = heuristic_weights if heuristic_weights is not None else RoleHeuristicWeights()
    targets = role_targets if role_targets is not None else weights.role_targets
    requested_ids = set(robot_ids)
    present_ids = {
        robot.robot_id
        for robot in snapshot.own_robots
        if robot.robot_id in requested_ids
    }
    if not present_ids:
        return RoleAssignmentResult(roles={}, contexts={}, scores={}, reasons={})

    base = base_roles if base_roles is not None else {}
    goalies = {
        rid
        for rid in present_ids
        if base.get(rid) == RoleType.GOALIE
        or current_roles.get(rid) == RoleType.GOALIE
    }
    candidates = sorted(present_ids - goalies)

    roles: dict[int, RoleType] = {rid: RoleType.GOALIE for rid in goalies}
    if not candidates:
        return RoleAssignmentResult(roles=roles, contexts={}, scores={}, reasons={})

    field_scale = _field_scale(snapshot, own_goal, attack_goal)
    role_counts = _count_roles(current_roles)
    ball_holder = _estimate_ball_holder(
        snapshot,
        possession_radius=(
            field_scale * weights.context.possession_radius_field_scale
        ),
    )
    contexts = {
        rid: build_role_swap_context(
            snapshot,
            rid,
            current_roles.get(rid, base.get(rid, RoleType.SUPPORTER)),
            current_ball_holder=ball_holder,
            time_since_last_swap=(
                time_since_last_swap.get(rid, math.inf)
                if time_since_last_swap is not None
                else math.inf
            ),
            same_role_count=role_counts[
                current_roles.get(rid, base.get(rid, RoleType.SUPPORTER))
            ],
            attack_goal=attack_goal,
            own_goal=own_goal,
            goal_sight_clearance=(
                field_scale * weights.context.goal_sight_clearance_field_scale
            ),
            role_counts=role_counts,
            role_targets=targets,
            lane_width=field_scale * weights.context.lane_width_field_scale,
            pressure_radius=field_scale * weights.context.pressure_radius_field_scale,
            minimum_swap_interval=weights.stability.minimum_swap_interval,
            approach_weights=weights.approach,
        )
        for rid in candidates
    }

    scores = _score_contexts(contexts, own_goal, attack_goal, weights)
    defender_target = _target_defender_count(
        contexts,
        len(candidates),
        targets,
        weights.defender_count,
    )
    sticky_defenders = _select_sticky_defenders(
        candidates,
        scores,
        current_roles,
        contexts,
        defender_target,
        weights.defender_stability,
    )
    for rid in sticky_defenders:
        roles[rid] = RoleType.DEFENDER

    attacker_pool = [rid for rid in candidates if rid not in roles]
    selected_attackers = _select_role_ids(
        attacker_pool,
        scores,
        current_roles,
        RoleType.ATTACKER,
        max(0, min(targets.attackers, len(attacker_pool))),
    )
    for rid in selected_attackers:
        roles[rid] = RoleType.ATTACKER

    remaining = [rid for rid in candidates if rid not in roles]
    defender_slots = max(0, defender_target - len(sticky_defenders))
    selected_defenders = _select_role_ids(
        remaining,
        scores,
        current_roles,
        RoleType.DEFENDER,
        max(0, min(defender_slots, len(remaining))),
    )
    for rid in selected_defenders:
        roles[rid] = RoleType.DEFENDER

    # Markers — man-mark opponents while defending (off unless role_targets.markers
    # > 0). The Coordinator matches each MARKER to a specific opponent afterwards.
    remaining = [rid for rid in candidates if rid not in roles]
    marker_target = _target_marker_count(contexts, targets, len(remaining))
    selected_markers = _select_role_ids(
        remaining,
        scores,
        current_roles,
        RoleType.MARKER,
        marker_target,
    )
    for rid in selected_markers:
        roles[rid] = RoleType.MARKER

    for rid in candidates:
        roles.setdefault(rid, RoleType.SUPPORTER)

    reasons = {
        rid: _role_reason(rid, roles[rid], contexts.get(rid), scores.get(rid))
        for rid in roles
    }
    return RoleAssignmentResult(
        roles=roles,
        contexts=contexts,
        scores=scores,
        reasons=reasons,
    )


def build_role_swap_context(
    snapshot: Snapshot,
    robot_id: int,
    current_role: RoleType,
    *,
    confidence_to_score: float = 0.0,
    current_ball_holder: BallHolder = "unknown",
    time_since_last_swap: float = 0.0,
    same_role_count: int = 1,
    attack_goal: Point = (4.5, 0.0),
    own_goal: Point = (-4.5, 0.0),
    goal_sight_clearance: float = 0.18,
    role_counts: Mapping[RoleType, int] | None = None,
    role_targets: RoleTargetCounts | None = None,
    lane_width: float = 0.35,
    pressure_radius: float = 1.0,
    minimum_swap_interval: float = 1.0,
    approach_weights: ApproachQualityWeights | None = None,
) -> RoleSwapContext:
    """Build reusable heuristic inputs for one robot.

    Goal sight is measured from the ball to the opponent goal and checks
    robots as blockers. That is the important gate before making a
    robot an attacker: if the shot lane is blocked, attacker confidence should
    be treated with suspicion by the future scorer.
    """

    robot = _find_robot(snapshot, robot_id)
    if robot is None:
        raise ValueError(f"Robot {robot_id} not found in snapshot")

    teammates = tuple(r for r in snapshot.own_robots if r.robot_id != robot_id)
    goal_blockers = tuple(snapshot.enemy_robots) + teammates
    goal_sight = evaluate_line_of_sight(
        snapshot.ball_position,
        attack_goal,
        goal_blockers,
        clearance=goal_sight_clearance,
    )
    counts = _normalise_role_counts(role_counts, current_role, same_role_count)

    return RoleSwapContext(
        robot_id=robot_id,
        current_role=current_role,
        distance_to_ball=distance(robot.position, snapshot.ball_position),
        confidence_to_score=float(confidence_to_score),
        current_ball_holder=current_ball_holder,
        goal_to_ball_distance=distance(attack_goal, snapshot.ball_position),
        time_since_last_swap=float(time_since_last_swap),
        same_role_count=int(same_role_count),
        goal_sight=goal_sight,
        distance_to_own_goal=distance(robot.position, own_goal),
        distance_to_opponent_goal=distance(robot.position, attack_goal),
        ball_in_our_half=_is_in_own_half(
            snapshot.ball_position,
            own_goal,
            attack_goal,
        ),
        robot_between_ball_and_own_goal=_point_between(
            robot.position,
            snapshot.ball_position,
            own_goal,
            lane_width,
        ),
        robot_between_ball_and_opponent_goal=_point_between(
            robot.position,
            snapshot.ball_position,
            attack_goal,
            lane_width,
        ),
        angle_to_ball=abs(
            angle_error(
                face_angle(robot.position, snapshot.ball_position),
                robot.orientation,
            )
        ),
        approach_quality=_approach_quality(
            robot,
            snapshot.ball_position,
            attack_goal,
            approach_weights if approach_weights is not None else ApproachQualityWeights(),
        ),
        lateral_offset_from_ball=abs(robot.position[1] - snapshot.ball_position[1]),
        teammate_spacing=_nearest_distance(robot.position, teammates),
        opponent_pressure=_opponent_pressure(
            robot.position,
            snapshot.enemy_robots,
            pressure_radius,
        ),
        attacker_count=counts[RoleType.ATTACKER],
        defender_count=counts[RoleType.DEFENDER],
        supporter_count=counts[RoleType.SUPPORTER],
        goalie_count=counts[RoleType.GOALIE],
        minimum_swap_interval=float(minimum_swap_interval),
        role_targets=role_targets if role_targets is not None else RoleTargetCounts(),
    )


def heuristic_role_swap(context: RoleSwapContext) -> RoleSwapDecision:
    """Placeholder role-swap decision.

    This is deliberately not the final algorithm. The Coordinator can call this
    later once we decide the exact scoring weights and team-level constraints.
    """

    if context.current_role == RoleType.GOALIE:
        reason = "goalie role remains fixed"
    elif not context.has_goal_sight and context.current_role == RoleType.ATTACKER:
        reason = "goal sight blocked; future scorer should lower attacker score"
    elif context.swap_cooldown_active:
        reason = "swap cooldown active; future scorer should avoid role churn"
    elif context.defender_count < context.role_targets.min_defenders:
        reason = "defender minimum missing; future scorer should protect defense"
    elif context.supporter_count < context.role_targets.min_supporters:
        reason = "supporter minimum missing; future scorer should preserve support shape"
    else:
        reason = "role swapping not implemented yet"

    return RoleSwapDecision(
        selected_role=context.current_role,
        should_swap=False,
        context=context,
        reason=reason,
    )


def _score_contexts(
    contexts: Mapping[int, RoleSwapContext],
    own_goal: Point,
    attack_goal: Point,
    weights: RoleHeuristicWeights,
) -> dict[int, RoleScores]:
    ball_close = _normalise_low_better(
        {rid: ctx.distance_to_ball for rid, ctx in contexts.items()}
    )
    own_goal_close = _normalise_low_better(
        {rid: ctx.distance_to_own_goal for rid, ctx in contexts.items()}
    )
    opponent_goal_close = _normalise_low_better(
        {rid: ctx.distance_to_opponent_goal for rid, ctx in contexts.items()}
    )
    spacing = _normalise_high_better(
        {rid: ctx.teammate_spacing for rid, ctx in contexts.items()}
    )
    ball_danger = _ball_danger_score(
        next(iter(contexts.values())).goal_sight.start,
        own_goal,
        attack_goal,
    )
    width = _normalise_high_better(
        {rid: ctx.lateral_offset_from_ball for rid, ctx in contexts.items()}
    )
    # The single deepest field player (closest to our own goal) is the "last
    # man"; used by the optional defender.last_man weight.
    deepest_rid = min(
        contexts,
        key=lambda rid: contexts[rid].distance_to_own_goal,
    )

    scores: dict[int, RoleScores] = {}
    for rid, ctx in contexts.items():
        angle_score = _clamp(1.0 - (ctx.angle_to_ball / math.pi))
        pressure_escape = _clamp(1.0 - ctx.opponent_pressure)
        goal_sight = 1.0 if ctx.has_goal_sight else 0.0
        own_lane = 1.0 if ctx.robot_between_ball_and_own_goal else 0.0
        forward_lane = 1.0 if ctx.robot_between_ball_and_opponent_goal else 0.0
        not_crowding_ball = _clamp(1.0 - ball_close[rid])
        opponent_has_ball = 1.0 if ctx.current_ball_holder == "opponent" else 0.0
        own_has_ball = 1.0 if ctx.current_ball_holder == "own" else 0.0
        loose_ball = 1.0 if ctx.current_ball_holder in ("none", "unknown") else 0.0
        attacker_weights = weights.attacker
        defender_weights = weights.defender
        supporter_weights = weights.supporter
        marker_weights = weights.marker
        ball_in_our_half_score = 1.0 if ctx.ball_in_our_half else 0.0

        attacker = (
            attacker_weights.ball_close * ball_close[rid]
            + attacker_weights.approach_quality * ctx.approach_quality
            + attacker_weights.angle_score * angle_score
            + attacker_weights.opponent_goal_close * opponent_goal_close[rid]
            + attacker_weights.goal_sight * goal_sight
            + attacker_weights.pressure_escape * pressure_escape
            + attacker_weights.own_has_ball * own_has_ball
            + attacker_weights.opponent_has_ball_pressure * ball_close[rid] * opponent_has_ball
            + attacker_weights.loose_ball_pressure * ball_close[rid] * loose_ball
        )
        last_man = 1.0 if rid == deepest_rid else 0.0
        defender = (
            defender_weights.own_goal_close * own_goal_close[rid]
            + defender_weights.ball_close * ball_close[rid]
            + defender_weights.own_lane * own_lane
            + defender_weights.ball_danger * ball_danger
            + defender_weights.pressure_escape * pressure_escape
            + defender_weights.opponent_has_ball * opponent_has_ball
            + defender_weights.last_man * last_man
        )
        supporter = (
            supporter_weights.spacing * spacing[rid]
            + supporter_weights.opponent_goal_close * opponent_goal_close[rid]
            + supporter_weights.pressure_escape * pressure_escape
            + supporter_weights.goal_sight * goal_sight
            + supporter_weights.not_crowding_ball * not_crowding_ball
            + supporter_weights.forward_lane * forward_lane
            + supporter_weights.own_has_ball * own_has_ball
            + supporter_weights.width * width[rid]
        )
        marker = (
            marker_weights.opponent_has_ball * opponent_has_ball
            + marker_weights.ball_in_our_half * ball_in_our_half_score
            + marker_weights.near_opponent * ctx.opponent_pressure
            + marker_weights.goal_side * own_lane
            + marker_weights.not_crowding_ball * not_crowding_ball
        )
        scores[rid] = _apply_role_stability(
            RoleScores(
                attacker=_clamp(attacker),
                defender=_clamp(defender),
                supporter=_clamp(supporter),
                marker=_clamp(marker),
            ),
            ctx,
            weights.stability,
            weights.defender_stability,
        )

    return scores


def _apply_role_stability(
    scores: RoleScores,
    context: RoleSwapContext,
    weights: RoleStabilityWeights,
    defender_weights: DefenderStabilityWeights,
) -> RoleScores:
    if context.current_role == RoleType.GOALIE:
        return scores

    if context.current_role == RoleType.DEFENDER:
        defender_bias = defender_weights.stay_bias
        if context.time_since_last_swap < defender_weights.min_hold_seconds:
            defender_bias = defender_weights.cooldown_bias
        return RoleScores(
            attacker=scores.attacker,
            defender=scores.defender + defender_bias,
            supporter=scores.supporter,
            marker=scores.marker,
        )

    bias = weights.current_role_bias
    if context.swap_cooldown_active:
        bias = weights.cooldown_bias

    return RoleScores(
        attacker=scores.attacker + (bias if context.current_role == RoleType.ATTACKER else 0.0),
        defender=scores.defender + (bias if context.current_role == RoleType.DEFENDER else 0.0),
        supporter=scores.supporter + (bias if context.current_role == RoleType.SUPPORTER else 0.0),
        marker=scores.marker + (bias if context.current_role == RoleType.MARKER else 0.0),
    )


def _select_sticky_defenders(
    robot_ids: list[int],
    scores: Mapping[int, RoleScores],
    current_roles: Mapping[int, RoleType],
    contexts: Mapping[int, RoleSwapContext],
    defender_target: int,
    weights: DefenderStabilityWeights,
) -> list[int]:
    if defender_target <= 0:
        return []

    current_defenders = [
        rid for rid in robot_ids
        if current_roles.get(rid) == RoleType.DEFENDER
    ]
    if not current_defenders:
        return []

    kept: list[int] = []
    for rid in sorted(
        current_defenders,
        key=lambda robot_id: scores[robot_id].defender,
        reverse=True,
    ):
        if len(kept) >= defender_target:
            break
        if _should_keep_current_defender(
            rid,
            robot_ids,
            scores,
            current_roles,
            contexts,
            weights,
        ):
            kept.append(rid)
    return kept


def _should_keep_current_defender(
    robot_id: int,
    robot_ids: list[int],
    scores: Mapping[int, RoleScores],
    current_roles: Mapping[int, RoleType],
    contexts: Mapping[int, RoleSwapContext],
    weights: DefenderStabilityWeights,
) -> bool:
    context = contexts.get(robot_id)
    if context is None:
        return True

    if _should_release_defender_to_attack(
        robot_id,
        robot_ids,
        scores,
        current_roles,
        contexts,
        weights,
    ):
        return False

    if context.time_since_last_swap < weights.min_hold_seconds:
        return True

    best_replacement = max(
        (
            scores[rid].defender
            for rid in robot_ids
            if rid != robot_id
        ),
        default=-math.inf,
    )
    return best_replacement <= scores[robot_id].defender + weights.release_margin


def _should_release_defender_to_attack(
    robot_id: int,
    robot_ids: list[int],
    scores: Mapping[int, RoleScores],
    current_roles: Mapping[int, RoleType],
    contexts: Mapping[int, RoleSwapContext],
    weights: DefenderStabilityWeights,
) -> bool:
    context = contexts.get(robot_id)
    if context is None:
        return False
    if context.current_ball_holder not in ("opponent", "none", "unknown"):
        return False

    best_non_defender_attacker = max(
        (
            scores[rid].attacker
            for rid in robot_ids
            if rid != robot_id and current_roles.get(rid) != RoleType.DEFENDER
        ),
        default=-math.inf,
    )
    if math.isinf(best_non_defender_attacker):
        return False

    return (
        scores[robot_id].attacker
        >= best_non_defender_attacker + weights.allow_attacker_release_margin
    )


def _select_role_ids(
    robot_ids: list[int],
    scores: Mapping[int, RoleScores],
    current_roles: Mapping[int, RoleType],
    role: RoleType,
    count: int,
) -> list[int]:
    if count <= 0:
        return []

    def key(robot_id: int) -> tuple[float, int, int]:
        role_score = _score_for_role(scores[robot_id], role)
        staying = 1 if current_roles.get(robot_id) == role else 0
        return (role_score, staying, -robot_id)

    return sorted(robot_ids, key=key, reverse=True)[:count]


def _score_for_role(scores: RoleScores, role: RoleType) -> float:
    if role == RoleType.ATTACKER:
        return scores.attacker
    if role == RoleType.DEFENDER:
        return scores.defender
    if role == RoleType.SUPPORTER:
        return scores.supporter
    if role == RoleType.MARKER:
        return scores.marker
    return 0.0


def _target_defender_count(
    contexts: Mapping[int, RoleSwapContext],
    candidate_count: int,
    targets: RoleTargetCounts,
    weights: DefenderCountWeights,
) -> int:
    if candidate_count <= targets.attackers:
        return 0

    any_ctx = next(iter(contexts.values()))
    desired = targets.min_defenders
    opponent_pressure_state = (
        (
            weights.add_second_when_opponent_has_ball
            and any_ctx.current_ball_holder == "opponent"
        )
        or (
            weights.add_second_when_ball_in_our_half
            and any_ctx.ball_in_our_half
        )
    )
    if (
        opponent_pressure_state
        and candidate_count - targets.attackers
        >= weights.minimum_candidates_after_attackers
    ):
        desired += 1

    return max(targets.min_defenders, min(targets.max_defenders, desired))


def _target_marker_count(
    contexts: Mapping[int, RoleSwapContext],
    targets: RoleTargetCounts,
    remaining_count: int,
) -> int:
    """How many MARKER robots to assign this tick.

    Markers only appear while DEFENDING (the opponent has the ball or it's in
    our half) — man-marking when we're on the front foot makes no sense. Capped
    so at least ``min_supporters`` robots stay in support. 0 when
    ``role_targets.markers`` is 0 (the default → feature off).
    """
    if targets.markers <= 0 or remaining_count <= 0 or not contexts:
        return 0
    any_ctx = next(iter(contexts.values()))
    defending = (
        any_ctx.current_ball_holder == "opponent" or any_ctx.ball_in_our_half
    )
    if not defending:
        return 0
    allowed = max(0, remaining_count - max(0, targets.min_supporters))
    return max(0, min(targets.markers, allowed))


def _role_reason(
    robot_id: int,
    role: RoleType,
    context: RoleSwapContext | None,
    scores: RoleScores | None,
) -> str:
    if role == RoleType.GOALIE:
        return "goalie fixed by base role"
    if context is None or scores is None:
        return "no heuristic context available"
    if role == RoleType.ATTACKER:
        return (
            f"attacker score={scores.attacker:.2f}; "
            f"distance_to_ball={context.distance_to_ball:.2f}; "
            f"goal_sight={context.has_goal_sight}"
        )
    if role == RoleType.DEFENDER:
        return (
            f"defender score={scores.defender:.2f}; "
            f"between_ball_and_goal={context.robot_between_ball_and_own_goal}; "
            f"ball_in_our_half={context.ball_in_our_half}"
        )
    if role == RoleType.MARKER:
        return (
            f"marker score={scores.marker:.2f}; "
            f"opponent_pressure={context.opponent_pressure:.2f}; "
            f"ball_holder={context.current_ball_holder}"
        )
    return (
        f"supporter score={scores.supporter:.2f}; "
        f"teammate_spacing={context.teammate_spacing:.2f}"
    )


def _dataclass_from_section(cls: type, section: object) -> Any:
    defaults = cls()
    if not isinstance(section, MappingABC):
        return defaults

    values: dict[str, object] = {}
    for item in fields(cls):
        if item.name not in section:
            continue
        default_value = getattr(defaults, item.name)
        values[item.name] = _coerce_config_value(section[item.name], default_value)
    return cls(**values)


def _resolve_utils_config_path(config_filename: str | Path) -> Path:
    path = Path(config_filename)
    if path.is_absolute():
        return path

    utils_dir = Path(__file__).resolve().parents[2] / "utils"
    path = utils_dir / path
    if path.exists() or path.name != BT_TUNING_FILENAME:
        return path

    legacy_path = utils_dir / LEGACY_HEURISTIC_WEIGHT_FILENAME
    return legacy_path if legacy_path.exists() else path


def _coerce_config_value(value: object, default_value: object) -> object:
    if isinstance(default_value, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(value)
    if isinstance(default_value, float):
        return float(value)
    return value


def _find_robot(snapshot: Snapshot, robot_id: int) -> RobotState | None:
    for robot in snapshot.own_robots:
        if robot.robot_id == robot_id:
            return robot
    return None


def _normalise_role_counts(
    role_counts: Mapping[RoleType, int] | None,
    current_role: RoleType,
    same_role_count: int,
) -> dict[RoleType, int]:
    counts = {role: 0 for role in RoleType}
    if role_counts is not None:
        for role, count in role_counts.items():
            counts[role] = max(0, int(count))

    counts[current_role] = max(counts[current_role], max(0, int(same_role_count)))
    return counts


def _is_in_own_half(point: Point, own_goal: Point, attack_goal: Point) -> bool:
    axis = _sub(attack_goal, own_goal)
    if _norm(axis) < 1e-9:
        return False

    midfield = (
        (own_goal[0] + attack_goal[0]) / 2.0,
        (own_goal[1] + attack_goal[1]) / 2.0,
    )
    return _dot(_sub(point, midfield), axis) < 0.0


def _point_between(point: Point, seg_a: Point, seg_b: Point, corridor: float) -> bool:
    ab = _sub(seg_b, seg_a)
    ab_len_sq = _dot(ab, ab)
    if ab_len_sq < 1e-12:
        return distance(point, seg_a) <= corridor

    t = _dot(_sub(point, seg_a), ab) / ab_len_sq
    return (
        0.0 <= t <= 1.0
        and point_to_segment_dist(point, seg_a, seg_b) <= corridor
    )


def _approach_quality(
    robot: RobotState,
    ball_position: Point,
    attack_goal: Point,
    weights: ApproachQualityWeights | None = None,
) -> float:
    """Score how naturally this robot can approach the ball to attack."""

    if weights is None:
        weights = ApproachQualityWeights()

    ball_to_goal = _sub(attack_goal, ball_position)
    ball_to_robot = _sub(robot.position, ball_position)
    denom = _norm(ball_to_goal) * _norm(ball_to_robot)
    if denom < 1e-9:
        behind_ball_score = 0.5
    else:
        # 1.0 means the robot is behind the ball relative to the goal.
        behind_ball_score = _clamp(
            (1.0 - (_dot(ball_to_goal, ball_to_robot) / denom)) / 2.0
        )

    turn_error = abs(angle_error(face_angle(robot.position, ball_position), robot.orientation))
    facing_score = _clamp(1.0 - (turn_error / math.pi))
    distance_score = 1.0 / (1.0 + distance(robot.position, ball_position))
    return _clamp(
        (weights.behind_ball * behind_ball_score)
        + (weights.facing * facing_score)
        + (weights.distance * distance_score)
    )


def _nearest_distance(position: Point, robots: tuple[RobotState, ...]) -> float:
    if not robots:
        return math.inf
    return min(distance(position, robot.position) for robot in robots)


def _opponent_pressure(position: Point, opponents: tuple[RobotState, ...], radius: float) -> float:
    """Return a 0..1 pressure score based on nearest opponent distance."""

    if radius <= 0.0 or not opponents:
        return 0.0

    nearest = _nearest_distance(position, opponents)
    return _clamp(1.0 - (nearest / radius))


def _field_scale(snapshot: Snapshot, own_goal: Point, attack_goal: Point) -> float:
    points = [own_goal, attack_goal, snapshot.ball_position]
    points.extend(robot.position for robot in snapshot.own_robots)
    points.extend(robot.position for robot in snapshot.enemy_robots)

    max_dist = distance(own_goal, attack_goal)
    for i, point_a in enumerate(points):
        for point_b in points[i + 1:]:
            max_dist = max(max_dist, distance(point_a, point_b))

    return max(max_dist, 1.0)


def _count_roles(roles: Mapping[int, RoleType]) -> dict[RoleType, int]:
    counts = {role: 0 for role in RoleType}
    for role in roles.values():
        counts[role] += 1
    return counts


def _estimate_ball_holder(
    snapshot: Snapshot,
    *,
    possession_radius: float = 0.5,
) -> BallHolder:
    own_distance = _nearest_distance(snapshot.ball_position, tuple(snapshot.own_robots))
    opponent_distance = _nearest_distance(
        snapshot.ball_position,
        tuple(snapshot.enemy_robots),
    )
    if math.isinf(own_distance) and math.isinf(opponent_distance):
        return "unknown"
    if min(own_distance, opponent_distance) > possession_radius:
        return "none"
    if abs(own_distance - opponent_distance) < 1e-9:
        return "unknown"
    return "own" if own_distance < opponent_distance else "opponent"


def _normalise_low_better(values: Mapping[int, float]) -> dict[int, float]:
    return _normalise(values, low_better=True)


def _normalise_high_better(values: Mapping[int, float]) -> dict[int, float]:
    return _normalise(values, low_better=False)


def _normalise(values: Mapping[int, float], *, low_better: bool) -> dict[int, float]:
    if not values:
        return {}

    finite = [value for value in values.values() if math.isfinite(value)]
    if not finite:
        return {key: 0.5 for key in values}

    fallback = max(finite) if low_better else min(finite)
    cleaned = {
        key: (value if math.isfinite(value) else fallback)
        for key, value in values.items()
    }
    lo = min(cleaned.values())
    hi = max(cleaned.values())
    if abs(hi - lo) < 1e-9:
        return {key: 0.5 for key in values}

    if low_better:
        return {key: (hi - value) / (hi - lo) for key, value in cleaned.items()}
    return {key: (value - lo) / (hi - lo) for key, value in cleaned.items()}


def _ball_danger_score(ball_position: Point, own_goal: Point, attack_goal: Point) -> float:
    axis = _sub(attack_goal, own_goal)
    axis_len_sq = _dot(axis, axis)
    if axis_len_sq < 1e-12:
        return 0.5

    progress_to_attack_goal = _dot(_sub(ball_position, own_goal), axis) / axis_len_sq
    return _clamp(1.0 - progress_to_attack_goal)


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _dot(a: Point, b: Point) -> float:
    return (a[0] * b[0]) + (a[1] * b[1])


def _norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
