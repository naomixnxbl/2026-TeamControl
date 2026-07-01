"""Supporter behaviour tree — v2.

Topology:

    SupporterRoot (Selector, memory=False)
    ├── PossessionSequence (Sequence)
    │   ├── InPossession       → SUCCESS if dist ≤ POSSESSION_DIST AND heading aligned
    │   └── DistributeSelector (Selector)
    │       ├── PassSequence (Sequence)
    │       │   ├── FindOpenTeammate   → picks least-marked own robot (excl. goalie, excl. self)
    │       │   ├── DribbleTowardTarget → RUNNING + IntentDribble until facing target
    │       │   └── PassToTeammate     → IntentPass(target_robot_id, target_pos)
    │       ├── ShootIfClose           → IntentKick(opp_goal) if dist ≤ SHOOT_DIST_THRESHOLD
    │       └── DribbleToGoal          → IntentDribble(opp_goal)
    ├── ReceivePassSequence (Sequence)
    │   ├── IsPassTarget       → SUCCESS if _active_pass_target == this robot's id
    │   └── HoldForPass        → IntentMove(current_pos, face_ball)
    ├── BallPossessionSequence (Sequence)
    │   ├── IsClosestToBall    → SUCCESS only if this robot is nearest to ball (excl. goalie)
    │   └── GoToBall           → IntentMove(ball_position, angle_to_ball)
    └── RepositionToSpace              → IntentMove(best open grid cell)

Design notes
------------
- Snapshot is injected via ``set_snapshot()`` before each ``tick()``.
  All condition and action nodes access world state through
  ``self._tree._snapshot`` (read-only Snapshot reference).
- Blackboard is injected via ``tick(blackboard)`` using the standard
  ``_blackboard_ref`` protocol (one-element list). Nodes write
  ``_blackboard_ref[0].current_intent`` to produce their output.
- No raw motor commands are produced anywhere in this module.

Known limitations
-----------------
- Attacker also chases ball unconditionally via ChaseBall. If a supporter
  is closer, both will chase. Coordinator-level arbitration is a separate task.
- ``InPossession`` uses the same flickering ``POSSESSION_DIST`` as the
  attacker. Hysteresis fix applies to both trees once implemented.
- ``ball_velocity`` is ``(0,0)`` — repositioning and passing don't factor
  in ball motion because velocity isn't wired yet.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from pathlib import Path
import py_trees
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from TeamControl.bt.contracts.blackboard import RobotBlackboard
from TeamControl.bt.contracts.intent import (
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentPass,
)
from TeamControl.bt.contracts.snapshot import Snapshot
from TeamControl.bt.tactics.line_of_sight import evaluate_line_of_sight

BT_TUNING_FILENAME = "bt_tuning.yaml"
LEGACY_HEURISTIC_WEIGHT_FILENAME = "heuristic_weight.yaml"

# -----------------------------------------------------------------------
# Tuneable constants
# -----------------------------------------------------------------------

GOALIE_ID: int = 0
GOAL_POSITION: tuple[float, float] = (4.5, 0.0)

POSSESSION_DIST: float = 0.11 # 0.11 is the sweet spot!!!!
POSSESSION_HEADING_TOL: float = 0.3

SHOOT_DIST_THRESHOLD: float = 2.0
MARKED_THRESHOLD: float = 0.5
ATTACKER_ID: int = 1
ATTACKER_PASS_BONUS: float = 1.5
GOAL_PROXIMITY_WEIGHT: float = 1.2
MAX_FIELD_DIST: float = 10.82

GRID_STEP: float = 0.5
REPOSITION_X_MIN: float = -1.0
REPOSITION_X_MAX: float = 4.0
REPOSITION_Y_MIN: float = -2.5
REPOSITION_Y_MAX: float = 2.5

PASS_ORIENT_TOL: float = 0.2
PASS_SIGNAL_TIMEOUT_TICKS: int = 100


@dataclass(frozen=True)
class SupporterBehaviorConfig:
    """Configurable supporter tree values."""

    possession_dist: float = POSSESSION_DIST
    possession_heading_tol: float = POSSESSION_HEADING_TOL
    shoot_dist_threshold: float = SHOOT_DIST_THRESHOLD
    marked_threshold: float = MARKED_THRESHOLD
    attacker_id: int = ATTACKER_ID
    attacker_pass_bonus: float = ATTACKER_PASS_BONUS
    goal_proximity_weight: float = GOAL_PROXIMITY_WEIGHT
    max_field_dist: float = MAX_FIELD_DIST
    grid_step: float = GRID_STEP
    reposition_x_min: float = REPOSITION_X_MIN
    reposition_x_max: float = REPOSITION_X_MAX
    reposition_y_min: float = REPOSITION_Y_MIN
    reposition_y_max: float = REPOSITION_Y_MAX
    pass_orient_tol: float = PASS_ORIENT_TOL
    pass_signal_timeout_ticks: int = PASS_SIGNAL_TIMEOUT_TICKS
    # When repositioning, reject candidate cells closer than this to the ball so
    # supporters don't crowd the ball carrier. Default 0.0 disables the check,
    # so every cell is considered exactly as before.
    reposition_min_ball_distance: float = 0.0
    # Bias repositioning toward the opponent goal. Adds
    # reposition_goal_weight * (max_field_dist - dist_to_goal) to each cell's
    # openness score. Default 0.0 keeps the original openness-only scoring with
    # a goal-distance tie-break, so behaviour is unchanged.
    reposition_goal_weight: float = 0.0
    # Bias repositioning toward the ball carrier. Adds
    # reposition_ball_proximity_weight * (1 - dist_to_ball / max_field_dist)
    # to each cell's score, peaking at the ball position and falling off
    # linearly with distance. Raise to ~1.5 to make proximity to the ball the
    # dominant factor, keeping supporters in pass-receiving range.
    reposition_ball_proximity_weight: float = 0.0
    # Hard cap on how far (m) from the ball a supporter will consider
    # repositioning. Cells beyond this are skipped entirely.
    # 0.0 disables the check. Default 3.5 keeps supporters in passing range.
    reposition_max_ball_distance: float = 0.0
    # Corridor radius (m) used when scoring line-of-sight quality from the ball
    # to each candidate cell. Closest obstacle within this distance sets the
    # los_score; obstacles outside it contribute nothing.
    reposition_los_score_radius: float = 0.30
    # Weight for the LOS quality score. 0.0 disables it (original behaviour).
    # Raise to ~1.0 to strongly prefer cells with a clear pass corridor from
    # the ball carrier.
    reposition_los_weight: float = 0.0

    # --- Pass target selection weights (FindOpenTeammate) ---
    # Minimum corridor half-width (m) for a pass lane to be considered open.
    # Lanes with any obstacle closer than this are hard-rejected.
    pass_lane_clearance: float = 0.15
    # Wider scan radius (m) used to score lane quality continuously.
    # Closest obstacle within this radius sets lane_quality in [0, 1].
    pass_lane_score_radius: float = 0.40
    # Ideal pass distance in metres. Distance score peaks here and falls off.
    pass_ideal_distance: float = 1.8
    # Half-window (m) for distance scoring. Score reaches 0 at ±this from ideal.
    pass_distance_window: float = 2.0
    # Receiver openness normalization: nearest opponent at this distance → score 1.
    pass_openness_radius: float = 1.5
    # Scoring weights — should sum to 1.0.
    pass_lane_weight: float = 0.30      # how open the pass corridor is
    pass_distance_weight: float = 0.40  # how close to ideal pass range
    pass_openness_weight: float = 0.20  # how much space the receiver has
    pass_goal_weight: float = 0.10      # how close receiver is to opponent goal


def load_supporter_behavior_config(
    config_filename: str | Path = BT_TUNING_FILENAME,
) -> SupporterBehaviorConfig:
    """Load supporter behavior config from yaml, preserving defaults."""

    path = _resolve_utils_config_path(config_filename)

    if not path.exists():
        return SupporterBehaviorConfig()

    with open(path, "r") as f:
        raw = yaml.load(f, Loader) or {}
    if not isinstance(raw, Mapping):
        return SupporterBehaviorConfig()

    section = _nested_mapping(raw, ("behavior_tree", "supporter"))
    if section is None:
        section = raw.get("supporter_behavior")
    if not isinstance(section, Mapping):
        return SupporterBehaviorConfig()

    defaults = SupporterBehaviorConfig()
    values = {}
    for item in fields(SupporterBehaviorConfig):
        default = getattr(defaults, item.name)
        if item.name in section:
            values[item.name] = _coerce_config_value(section[item.name], default)
        else:
            values[item.name] = default
    return SupporterBehaviorConfig(**values)


def _coerce_config_value(value, default):
    if isinstance(default, int):
        return int(value)
    return float(value)


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


def _nested_mapping(raw: Mapping, keys: tuple[str, ...]) -> Mapping | None:
    current: object = raw
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _find_robot(snap: Snapshot, robot_id: int):
    for r in snap.own_robots:
        if r.robot_id == robot_id:
            return r
    return None


# -----------------------------------------------------------------------
# Condition / action nodes
# -----------------------------------------------------------------------

class IsClosestToBall(py_trees.behaviour.Behaviour):
    """Succeed only if this robot is the closest own robot to the ball.

    Excludes the goalie (GOALIE_ID). Tie-break: lowest robot_id wins.
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("IsClosestToBall")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        bx, by = snap.ball_position
        my_dist = math.hypot(robot.position[0] - bx, robot.position[1] - by)

        for r in snap.own_robots:
            if r.robot_id == GOALIE_ID or r.robot_id == bb.robot_id:
                continue
            d = math.hypot(r.position[0] - bx, r.position[1] - by)
            if d < my_dist or (d == my_dist and r.robot_id < bb.robot_id):
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS


class GoToBall(py_trees.behaviour.Behaviour):
    """Write IntentMove toward ball position, facing the ball. Always SUCCESS."""

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("GoToBall")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE
        angle_to_ball = math.atan2(
            snap.ball_position[1] - robot.position[1],
            snap.ball_position[0] - robot.position[0],
        )
        bb.current_intent = IntentMove(
            target_pos=snap.ball_position,
            target_orientation=angle_to_ball,
        )
        bb.intent_source = "GoToBall"
        return py_trees.common.Status.SUCCESS


class InPossession(py_trees.behaviour.Behaviour):
    """Succeed when the ball is close AND in front of the kicker.

    Same logic as HasBallControl in attacker.py.
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("InPossession")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        dx = snap.ball_position[0] - robot.position[0]
        dy = snap.ball_position[1] - robot.position[1]
        dist = math.hypot(dx, dy)
        config = self._tree.behavior_config
        if dist > config.possession_dist:
            return py_trees.common.Status.FAILURE

        if not self._tree._pass_committed:
            angle_to_ball = math.atan2(dy, dx)
            err = (angle_to_ball - robot.orientation + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > config.possession_heading_tol:
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS


class FindOpenTeammate(py_trees.behaviour.Behaviour):
    """Find the least-marked own robot (excl. goalie, excl. self).

    Writes pass target to tree scratch state on SUCCESS.
    Returns FAILURE if all teammates are within MARKED_THRESHOLD of an opponent.
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("FindOpenTeammate")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        best_id = None
        best_pos = None
        best_score = -1.0
        gx, gy = self._tree.goal_position
        cfg = self._tree.behavior_config
        ball_pos = snap.ball_position

        for r in snap.own_robots:
            if r.robot_id == GOALIE_ID or r.robot_id == bb.robot_id:
                continue

            # Build obstacle list: all robots except passer and this candidate.
            obstacles = list(snap.enemy_robots) + [
                other for other in snap.own_robots
                if other.robot_id not in (bb.robot_id, r.robot_id)
            ]

            # Evaluate lane with a wider radius to get a continuous quality score.
            los = evaluate_line_of_sight(
                ball_pos, r.position, obstacles,
                clearance=cfg.pass_lane_score_radius,
            )
            # Hard filter: reject if any obstacle is within the minimum corridor.
            if los.blockers and los.blockers[0].distance_to_line < cfg.pass_lane_clearance:
                continue
            # Lane quality: 1.0 when perfectly clear, falling off as obstacles
            # approach the minimum corridor.
            if los.blockers:
                lane_quality = min(
                    1.0,
                    los.blockers[0].distance_to_line / cfg.pass_lane_score_radius,
                )
            else:
                lane_quality = 1.0

            # Distance score: peaks at ideal pass distance, falls off either side.
            pass_dist = math.hypot(
                r.position[0] - ball_pos[0],
                r.position[1] - ball_pos[1],
            )
            dist_score = max(
                0.0,
                1.0 - abs(pass_dist - cfg.pass_ideal_distance) / cfg.pass_distance_window,
            )

            # Receiver openness: how far the nearest opponent is (normalized).
            if snap.enemy_robots:
                min_opp_dist = min(
                    math.hypot(r.position[0] - opp.position[0], r.position[1] - opp.position[1])
                    for opp in snap.enemy_robots
                )
                openness = min(1.0, min_opp_dist / cfg.pass_openness_radius)
            else:
                openness = 1.0

            # Goal proximity: how close the receiver is to the opponent goal.
            dist_to_goal = math.hypot(r.position[0] - gx, r.position[1] - gy)
            goal_proximity = 1.0 - min(1.0, dist_to_goal / cfg.max_field_dist)

            score = (
                cfg.pass_lane_weight     * lane_quality
                + cfg.pass_distance_weight * dist_score
                + cfg.pass_openness_weight * openness
                + cfg.pass_goal_weight     * goal_proximity
            )

            # Attacker gets a bonus to prefer feeding the ball-carrier role.
            if r.robot_id == cfg.attacker_id:
                score *= cfg.attacker_pass_bonus

            if score > best_score:
                best_score = score
                best_id = r.robot_id
                best_pos = r.position

        if best_id is None or best_score < cfg.marked_threshold:
            return py_trees.common.Status.FAILURE

        self._tree._pass_target_id = best_id
        self._tree._pass_target_pos = best_pos
        self._tree._pass_committed = True
        return py_trees.common.Status.SUCCESS


class DribbleTowardTarget(py_trees.behaviour.Behaviour):
    """Dribble-turn toward the pass target before kicking.

    Returns RUNNING (+ IntentDribble toward teammate) while the robot is
    not facing the target. The dribbler stays active, holding the ball
    during the turn. Returns SUCCESS once aligned within PASS_ORIENT_TOL,
    allowing PassToTeammate to fire on the same tick.
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("DribbleTowardTarget")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        if self._tree._pass_target_pos is None:
            return py_trees.common.Status.FAILURE
        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        tx, ty = self._tree._pass_target_pos
        angle_to_target = math.atan2(
            ty - robot.position[1],
            tx - robot.position[0],
        )
        err = (angle_to_target - robot.orientation + math.pi) % (2 * math.pi) - math.pi
        if abs(err) <= self._tree.behavior_config.pass_orient_tol:
            return py_trees.common.Status.SUCCESS

        bb.current_intent = IntentDribble(target_pos=self._tree._pass_target_pos)
        bb.intent_source = "DribbleTowardTarget"
        return py_trees.common.Status.RUNNING


class PassToTeammate(py_trees.behaviour.Behaviour):
    """Write IntentPass to the target found by FindOpenTeammate."""

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("PassToTeammate")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        if self._tree._pass_target_id is None or self._tree._pass_target_pos is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentPass(
            target_robot_id=self._tree._pass_target_id,
            target_pos=self._tree._pass_target_pos,
        )
        bb.intent_source = "PassToTeammate"
        self._tree._active_pass_target = self._tree._pass_target_id
        self._tree._active_pass_target_age = 0
        return py_trees.common.Status.SUCCESS


class ShootIfClose(py_trees.behaviour.Behaviour):
    """Write IntentKick toward opponent goal if within SHOOT_DIST_THRESHOLD."""

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("ShootIfClose")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE
        goal = self._tree.goal_position
        dist = math.hypot(robot.position[0] - goal[0], robot.position[1] - goal[1])
        if dist > self._tree.behavior_config.shoot_dist_threshold:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentKick(target_pos=goal)
        bb.intent_source = "ShootIfClose"
        return py_trees.common.Status.SUCCESS


class DribbleToGoal(py_trees.behaviour.Behaviour):
    """Write IntentDribble toward opponent goal. Always SUCCESS."""

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("DribbleToGoal")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentDribble(target_pos=self._tree.goal_position)
        bb.intent_source = "DribbleToGoal"
        return py_trees.common.Status.SUCCESS


class IsPassTarget(py_trees.behaviour.Behaviour):
    """Succeed if another supporter has signalled a pass to this robot.

    Reads ``self._tree._active_pass_target``, a persistent field on the
    shared SupporterTree instance. Set by PassToTeammate, cleared when:
    - the receiver gains possession (InPossession succeeds for it)
    - the signal times out (PASS_SIGNAL_TIMEOUT_TICKS ticks without reception)
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("IsPassTarget")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        if self._tree._active_pass_target != bb.robot_id:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class HoldForPass(py_trees.behaviour.Behaviour):
    """Wait for a pass, then actively meet the ball.

    Holds position facing the ball while the pass is still far away, but once
    the ball is within ``RECEIVE_RADIUS`` the receiver steps onto it (facing it)
    so a pass that lands a little short/wide is still collected on the dribbler
    instead of trickling past a stationary robot.
    """

    RECEIVE_RADIUS: float = 1.2

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("HoldForPass")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE
        angle_to_ball = math.atan2(
            snap.ball_position[1] - robot.position[1],
            snap.ball_position[0] - robot.position[0],
        )
        dist_to_ball = math.hypot(
            snap.ball_position[0] - robot.position[0],
            snap.ball_position[1] - robot.position[1],
        )
        if dist_to_ball <= self.RECEIVE_RADIUS:
            # Step onto the incoming ball to secure it on the dribbler.
            target = snap.ball_position
            bb.intent_source = "ReceiveMeetBall"
        else:
            target = robot.position
            bb.intent_source = "HoldForPass"
        bb.current_intent = IntentMove(
            target_pos=target,
            target_orientation=angle_to_ball,
        )
        return py_trees.common.Status.SUCCESS


class RepositionToSpace(py_trees.behaviour.Behaviour):
    """Move to the best open position on the field via grid scoring.

    Divides the attacking region into a grid and scores each cell by
    min(dist_to_nearest_opponent, dist_to_nearest_own_robot_excl_self).
    Picks the highest-scoring cell. Tie-break: closest to opponent goal.
    Orients toward ball to be ready for a pass.
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("RepositionToSpace")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        t = self._tree
        cfg = t.behavior_config
        best_pos = t._reposition_fallback
        best_score = -1.0
        best_goal_dist = float("inf")
        gx, gy = t.goal_position
        bx, by = snap.ball_position

        # Precompute obstacle list for LOS scoring (all robots except self).
        los_obstacles = list(snap.enemy_robots) + [
            r for r in snap.own_robots if r.robot_id != bb.robot_id
        ]
        los_enabled = cfg.reposition_los_weight > 0.0

        cx = t.repo_x_min
        while cx <= t.repo_x_max:
            cy = t.repo_y_min
            while cy <= t.repo_y_max:
                dist_to_ball = math.hypot(cx - bx, cy - by)

                # Hard filter: too close to ball (anti-crowding).
                if cfg.reposition_min_ball_distance > 0.0 and (
                    dist_to_ball < cfg.reposition_min_ball_distance
                ):
                    cy += cfg.grid_step
                    continue

                # Hard filter: too far from ball to be a viable pass receiver.
                if cfg.reposition_max_ball_distance > 0.0 and (
                    dist_to_ball > cfg.reposition_max_ball_distance
                ):
                    cy += cfg.grid_step
                    continue

                if snap.enemy_robots:
                    opp_score = min(
                        math.hypot(cx - opp.position[0], cy - opp.position[1])
                        for opp in snap.enemy_robots
                    )
                else:
                    opp_score = float("inf")

                own_score = float("inf")
                for r in snap.own_robots:
                    if r.robot_id == bb.robot_id:
                        continue
                    d = math.hypot(cx - r.position[0], cy - r.position[1])
                    if d < own_score:
                        own_score = d

                goal_dist = math.hypot(cx - gx, cy - gy)

                # LOS quality: how open is the pass corridor from the ball to
                # this cell? 1.0 = clear, falling off as obstacles approach the
                # corridor. Disabled (0.0 weight) by default.
                if los_enabled:
                    los = evaluate_line_of_sight(
                        (bx, by), (cx, cy), los_obstacles,
                        clearance=cfg.reposition_los_score_radius,
                    )
                    los_score = (
                        1.0 if not los.blockers
                        else min(1.0, los.blockers[0].distance_to_line
                                 / cfg.reposition_los_score_radius)
                    )
                else:
                    los_score = 0.0

                cell_score = min(opp_score, own_score) + (
                    cfg.reposition_goal_weight * max(0.0, cfg.max_field_dist - goal_dist)
                ) + (
                    cfg.reposition_ball_proximity_weight
                    * max(0.0, 1.0 - dist_to_ball / cfg.max_field_dist)
                ) + (
                    cfg.reposition_los_weight * los_score
                )

                if (cell_score > best_score
                        or (cell_score == best_score and goal_dist < best_goal_dist)):
                    best_score = cell_score
                    best_pos = (cx, cy)
                    best_goal_dist = goal_dist

                cy += cfg.grid_step
            cx += cfg.grid_step

        angle_to_ball = math.atan2(
            snap.ball_position[1] - robot.position[1],
            snap.ball_position[0] - robot.position[0],
        )
        bb.current_intent = IntentMove(
            target_pos=best_pos,
            target_orientation=angle_to_ball,
        )
        bb.intent_source = "RepositionToSpace"
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# SupporterTree
# -----------------------------------------------------------------------

class SupporterTree:
    """Wrapper around the Supporter py_trees topology.

    Usage::

        tree = SupporterTree(us_positive=True)
        tree.set_snapshot(snapshot)
        tree.tick(blackboard)
        intent = blackboard.current_intent
    """

    def __init__(
        self,
        us_positive: bool = True,
        behavior_config: SupporterBehaviorConfig | None = None,
        behavior_config_file: str = BT_TUNING_FILENAME,
    ) -> None:
        self._snapshot: Snapshot | None = None
        self._blackboard_ref: list = [None]
        self.us_positive = us_positive
        self.behavior_config = (
            behavior_config
            if behavior_config is not None
            else load_supporter_behavior_config(behavior_config_file)
        )

        self.goal_position: tuple[float, float] = (
            (-GOAL_POSITION[0], GOAL_POSITION[1]) if us_positive
            else GOAL_POSITION
        )

        # Scratch state for FindOpenTeammate → PassToTeammate (reset each tick)
        self._pass_target_id: int | None = None
        self._pass_target_pos: tuple[float, float] | None = None

        # Set by FindOpenTeammate — tells InPossession to skip the heading
        # check so the robot can dribble-turn toward the pass target.
        self._pass_committed: bool = False

        # Persistent pass signal: set by PassToTeammate, read by IsPassTarget.
        # Survives across ticks until cleared by reception, loss, or timeout.
        self._active_pass_target: int | None = None
        self._active_pass_target_age: int = 0

        # Reposition bounds (mirror x when us_positive=True)
        if us_positive:
            self.repo_x_min = -self.behavior_config.reposition_x_max
            self.repo_x_max = -self.behavior_config.reposition_x_min
        else:
            self.repo_x_min = self.behavior_config.reposition_x_min
            self.repo_x_max = self.behavior_config.reposition_x_max
        self.repo_y_min = self.behavior_config.reposition_y_min
        self.repo_y_max = self.behavior_config.reposition_y_max
        self._reposition_fallback: tuple[float, float] = (
            (self.repo_x_min + self.repo_x_max) / 2.0,
            (self.repo_y_min + self.repo_y_max) / 2.0,
        )

        self.root = self._build_tree()

    # ------------------------------------------------------------------

    def set_snapshot(self, snapshot: Snapshot) -> None:
        """Inject the current world-state snapshot before ticking."""
        self._snapshot = snapshot

    def tick(self, blackboard: RobotBlackboard) -> None:
        """Tick the tree with the given per-robot blackboard."""
        self._pass_target_id = None
        self._pass_target_pos = None
        self._pass_committed = False
        self._blackboard_ref[0] = blackboard

        # Age the pass signal and clear on timeout.
        if self._active_pass_target is not None:
            self._active_pass_target_age += 1
            if (
                self._active_pass_target_age
                > self.behavior_config.pass_signal_timeout_ticks
            ):
                self._active_pass_target = None
                self._active_pass_target_age = 0

        # Snapshot the signal before the tree runs so we can detect if the
        # receiver transitioned away from holding.
        was_target = (self._active_pass_target is not None
                      and blackboard.robot_id == self._active_pass_target)

        self.root.tick_once()

        # Clear the signal when the receiver does anything other than hold.
        # This covers: receiver gained possession (pass/shoot/dribble),
        # receiver became closest and chased, or receiver repositioned.
        if was_target and blackboard.intent_source != "HoldForPass":
            self._active_pass_target = None
            self._active_pass_target_age = 0

    # ------------------------------------------------------------------

    def _build_tree(self) -> py_trees.composites.Selector:
        # Branch 1: BallPossessionSequence — chase if closest
        chase_seq = py_trees.composites.Sequence(
            name="BallPossessionSequence", memory=False
        )
        chase_seq.add_children([
            IsClosestToBall(self),
            GoToBall(self),
        ])

        # Branch 2: ReceivePassSequence — hold position if we're the pass target
        receive_seq = py_trees.composites.Sequence(
            name="ReceivePassSequence", memory=False
        )
        receive_seq.add_children([
            IsPassTarget(self),
            HoldForPass(self),
        ])

        # Branch 3: PossessionSequence — distribute if we have the ball
        pass_seq = py_trees.composites.Sequence(
            name="PassSequence", memory=False
        )
        pass_seq.add_children([
            FindOpenTeammate(self),
            DribbleTowardTarget(self),
            PassToTeammate(self),
        ])

        distribute = py_trees.composites.Selector(
            name="DistributeSelector", memory=False
        )
        distribute.add_children([
            pass_seq,
            ShootIfClose(self),
            DribbleToGoal(self),
        ])

        possession_seq = py_trees.composites.Sequence(
            name="PossessionSequence", memory=False
        )
        possession_seq.add_children([
            InPossession(self),
            distribute,
        ])

        # Root: SupporterRoot
        root = py_trees.composites.Selector(
            name="SupporterRoot", memory=False
        )
        root.add_children([
            possession_seq,
            receive_seq,
            chase_seq,
            RepositionToSpace(self),
        ])
        return root
