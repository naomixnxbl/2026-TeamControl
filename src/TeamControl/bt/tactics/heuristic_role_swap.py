"""Heuristic role assignment for dynamic BT roles.

The team-level helper ranks robots relative to each other, then assigns one
attacker, required defenders, and supporters. The Coordinator decides whether
to call this module; when the yaml flag is false, static roles remain untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping

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


@dataclass(frozen=True)
class RoleTargetCounts:
    """Team-level role bounds for the future scorer."""

    attackers: int = 1
    min_defenders: int = 1
    max_defenders: int = 2
    min_supporters: int = 1


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


@dataclass(frozen=True)
class RoleAssignmentResult:
    """Team-level role assignment result."""

    roles: dict[int, RoleType]
    contexts: dict[int, RoleSwapContext]
    scores: dict[int, RoleScores]
    reasons: dict[int, str]


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
) -> RoleAssignmentResult:
    """Assign non-goalie roles by relative team ranking.

    The goalie is fixed from ``base_roles``/``current_roles``. Every other
    robot gets scored relative to the available robots on the same tick:
    closest-to-ball, closest-to-goal, spacing, pressure, and goal sight are
    normalized against the current team state rather than absolute thresholds.
    """

    targets = role_targets if role_targets is not None else RoleTargetCounts()
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
    contexts = {
        rid: build_role_swap_context(
            snapshot,
            rid,
            current_roles.get(rid, base.get(rid, RoleType.SUPPORTER)),
            current_ball_holder=_estimate_ball_holder(snapshot),
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
            goal_sight_clearance=field_scale * 0.02,
            role_counts=role_counts,
            role_targets=targets,
            lane_width=field_scale * 0.04,
            pressure_radius=field_scale * 0.12,
        )
        for rid in candidates
    }

    scores = _score_contexts(contexts, own_goal, attack_goal)
    selected_attackers = _select_role_ids(
        candidates,
        scores,
        current_roles,
        RoleType.ATTACKER,
        max(0, min(targets.attackers, len(candidates))),
    )
    for rid in selected_attackers:
        roles[rid] = RoleType.ATTACKER

    remaining = [rid for rid in candidates if rid not in roles]
    defender_target = _target_defender_count(
        contexts,
        len(candidates),
        targets,
    )
    selected_defenders = _select_role_ids(
        remaining,
        scores,
        current_roles,
        RoleType.DEFENDER,
        max(0, min(defender_target, len(remaining))),
    )
    for rid in selected_defenders:
        roles[rid] = RoleType.DEFENDER

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
    goal_blockers = tuple(snapshot.opponent_robots) + teammates
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
        approach_quality=_approach_quality(robot, snapshot.ball_position, attack_goal),
        teammate_spacing=_nearest_distance(robot.position, teammates),
        opponent_pressure=_opponent_pressure(
            robot.position,
            snapshot.opponent_robots,
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

        attacker = (
            0.40 * ball_close[rid]
            + 0.20 * ctx.approach_quality
            + 0.14 * angle_score
            + 0.12 * opponent_goal_close[rid]
            + 0.08 * goal_sight
            + 0.04 * pressure_escape
            + 0.02 * own_has_ball
        )
        defender = (
            0.28 * own_goal_close[rid]
            + 0.22 * ball_close[rid]
            + 0.24 * own_lane
            + 0.16 * ball_danger
            + 0.06 * pressure_escape
            + 0.04 * opponent_has_ball
        )
        supporter = (
            0.30 * spacing[rid]
            + 0.20 * opponent_goal_close[rid]
            + 0.16 * pressure_escape
            + 0.14 * goal_sight
            + 0.10 * not_crowding_ball
            + 0.08 * forward_lane
            + 0.02 * own_has_ball
        )
        scores[rid] = _apply_role_stability(
            RoleScores(
                attacker=_clamp(attacker),
                defender=_clamp(defender),
                supporter=_clamp(supporter),
            ),
            ctx,
        )

    return scores


def _apply_role_stability(scores: RoleScores, context: RoleSwapContext) -> RoleScores:
    if context.current_role == RoleType.GOALIE:
        return scores

    bias = 0.08
    if context.swap_cooldown_active:
        bias = 0.16

    return RoleScores(
        attacker=scores.attacker + (bias if context.current_role == RoleType.ATTACKER else 0.0),
        defender=scores.defender + (bias if context.current_role == RoleType.DEFENDER else 0.0),
        supporter=scores.supporter + (bias if context.current_role == RoleType.SUPPORTER else 0.0),
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
    return 0.0


def _target_defender_count(
    contexts: Mapping[int, RoleSwapContext],
    candidate_count: int,
    targets: RoleTargetCounts,
) -> int:
    if candidate_count <= targets.attackers:
        return 0

    any_ctx = next(iter(contexts.values()))
    desired = targets.min_defenders
    if any_ctx.ball_in_our_half and candidate_count - targets.attackers >= 3:
        desired += 1

    return max(targets.min_defenders, min(targets.max_defenders, desired))


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
    return (
        f"supporter score={scores.supporter:.2f}; "
        f"teammate_spacing={context.teammate_spacing:.2f}"
    )


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


def _approach_quality(robot: RobotState, ball_position: Point, attack_goal: Point) -> float:
    """Score how naturally this robot can approach the ball to attack."""

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
        (0.5 * behind_ball_score)
        + (0.3 * facing_score)
        + (0.2 * distance_score)
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
    points.extend(robot.position for robot in snapshot.opponent_robots)

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


def _estimate_ball_holder(snapshot: Snapshot) -> BallHolder:
    own_distance = _nearest_distance(snapshot.ball_position, tuple(snapshot.own_robots))
    opponent_distance = _nearest_distance(
        snapshot.ball_position,
        tuple(snapshot.opponent_robots),
    )
    if math.isinf(own_distance) and math.isinf(opponent_distance):
        return "unknown"
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
