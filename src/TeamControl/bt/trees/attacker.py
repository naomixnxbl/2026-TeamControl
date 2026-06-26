"""Attacker behaviour tree — R005.

Topology:

    AttackingSelector (Selector, memory=False)
    ├── PossessionSequence (Sequence)
    │   ├── HasBallControl
    │   └── PossessionAction (Selector)
    │       ├── ShootSequence (HasClearShot → ShootAtGoal)
    │       └── HoldPossession (dribble toward goal — the default)
    ├── WaitSequence (Sequence)
    │   ├── IsBallInOwnHalf
    │   └── WaitNearGoal (camp in front of enemy goal, face ball)
    └── ChaseBall (ball in enemy half; slow speed if not closest)

Priority:
    possession → dribble toward goal (default), shoot only if clear AND close
    no possession + ball in own half → wait near enemy goal
    no possession + ball in enemy half → chase ball

Known bugs / areas for improvement
-----------------------------------
1. **POSSESSION_DIST oscillation** — ``POSSESSION_DIST`` (0.122 m) is too
   tight. The robot flickers between possession and no-possession on
   consecutive ticks. Fix: add hysteresis (separate acquire/lose thresholds).
   See ``docs/future.md §0.1``.

2. **No field boundary enforcement** — target positions in ``ChaseBall``
   and ``HoldPossession`` are not clamped to the legal field rectangle.
   See ``docs/future.md §0.3``.

Design notes
------------
- Snapshot is injected via ``set_snapshot()`` before each ``tick()``.
  All condition and action nodes access world state through
  ``self._tree._snapshot`` (read-only Snapshot reference).
- Blackboard is injected via ``tick(blackboard)`` using the standard
  ``_blackboard_ref`` protocol (one-element list). Nodes write
  ``_blackboard_ref[0].current_intent`` to produce their output.
- No raw motor commands are produced anywhere in this module.
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
from TeamControl.bt.contracts.intent import IntentDribble, IntentKick, IntentMove, IntentPass
from TeamControl.bt.contracts.snapshot import Snapshot
from TeamControl.bt.tactics.line_of_sight import line_of_sight_clear

BT_TUNING_FILENAME = "bt_tuning.yaml"
LEGACY_HEURISTIC_WEIGHT_FILENAME = "heuristic_weight.yaml"

# -----------------------------------------------------------------------
# Tuneable constants
# -----------------------------------------------------------------------

BALL_IN_RANGE_THRESHOLD: float = 0.8   # metres — legacy chase threshold (unused by new tree)
SUPPORTER_ROLE_IDS: tuple[int, ...] = (3, 4)  # robot IDs with SUPPORTER role
GOAL_POSITION: tuple[float, float] = (4.5, 0.0)   # opponent goal centre

# Distance at which we consider the ball to be in the dribbler (we "have" it).
# SSL robot diameter ~0.18 m, ball ~0.043 m → centre-to-centre ~0.11 m when
# touching. 0.15 m gives a small margin so the predicate doesn't flicker.

##   note: This is really a threshold value for the robot turn bug   ##
POSSESSION_DIST: float = 0.11 # 0.11 is the sweet spot!!!!

# Maximum angular error between the robot's heading and the direction to the
# ball before we say we "have" the ball. The kicker plate is on the front of
# the robot — if the ball is behind us (or to the side), a kick command fires
# into empty space. ~17° tolerance lets us claim possession when the ball is
# roughly in front but not perfectly centred.
POSSESSION_HEADING_TOL: float = 0.3   # radians (~17 degrees)

# Half-width of the shooting corridor: an opponent within this perpendicular
# distance of the ball→goal line segment is considered to be blocking the shot.
# Robot radius is ~0.09 m, so 0.20 m means an opponent within ~one body length
# of the line counts as a block.
SHOT_CORRIDOR_RADIUS: float = 0.20
SHOT_HEADING_TOL: float = 0.4
SHOT_SETTLE_TICKS: int = 30

GOALIE_ID: int = 0
CHASE_SLOW_SPEED: float = 0.2
SHOOT_DIST_THRESHOLD: float = 2.0

PENALTY_BOX_DEPTH: float = 1.0
FIELD_HALF_X: float = 4.5
FIELD_HALF_Y: float = 3.0
WAIT_X: float = FIELD_HALF_X - PENALTY_BOX_DEPTH
PASS_MIN_DISTANCE_FRAC: float = 0.08
PASS_BACKWARD_ALLOWANCE_FRAC: float = 0.06
PASS_MARKED_DISTANCE_FRAC: float = 0.05
PASS_PRESSURE_RADIUS_FRAC: float = 0.08
PASS_LANE_CLEARANCE_FRAC: float = 0.02
PASS_ORIENT_TOL: float = 0.2


@dataclass(frozen=True)
class AttackerBehaviorConfig:
    """Configurable attacker tree values."""

    ball_in_range_threshold: float = BALL_IN_RANGE_THRESHOLD
    supporter_role_ids: tuple[int, ...] = SUPPORTER_ROLE_IDS
    possession_dist: float = POSSESSION_DIST
    possession_heading_tol: float = POSSESSION_HEADING_TOL
    shot_corridor_radius: float = SHOT_CORRIDOR_RADIUS
    shot_heading_tol: float = SHOT_HEADING_TOL
    shot_settle_ticks: int = SHOT_SETTLE_TICKS
    chase_slow_speed: float = CHASE_SLOW_SPEED
    shoot_dist_threshold: float = SHOOT_DIST_THRESHOLD
    wait_x: float = WAIT_X
    wait_y_limit: float = FIELD_HALF_Y
    pass_min_distance_frac: float = PASS_MIN_DISTANCE_FRAC
    pass_backward_allowance_frac: float = PASS_BACKWARD_ALLOWANCE_FRAC
    pass_marked_distance_frac: float = PASS_MARKED_DISTANCE_FRAC
    pass_pressure_radius_frac: float = PASS_PRESSURE_RADIUS_FRAC
    pass_lane_clearance_frac: float = PASS_LANE_CLEARANCE_FRAC
    pass_orient_tol: float = PASS_ORIENT_TOL
    pass_openness_weight: float = 0.45
    pass_forward_score_weight: float = 0.25
    pass_distance_score_weight: float = 0.20
    pass_goal_proximity_weight: float = 0.10
    pass_forward_scale_frac: float = 0.35
    pass_ideal_distance_frac: float = 0.22
    pass_distance_window_frac: float = 0.25


def load_attacker_behavior_config(
    config_filename: str | Path = BT_TUNING_FILENAME,
) -> AttackerBehaviorConfig:
    """Load attacker behavior config from yaml, preserving defaults."""

    path = _resolve_utils_config_path(config_filename)

    if not path.exists():
        return AttackerBehaviorConfig()

    with open(path, "r") as f:
        raw = yaml.load(f, Loader) or {}
    if not isinstance(raw, Mapping):
        return AttackerBehaviorConfig()

    section = _nested_mapping(raw, ("behavior_tree", "attacker"))
    if section is None:
        section = raw.get("attacker_behavior")
    if not isinstance(section, Mapping):
        return AttackerBehaviorConfig()

    defaults = AttackerBehaviorConfig()
    values = {}
    for item in fields(AttackerBehaviorConfig):
        default = getattr(defaults, item.name)
        if item.name in section:
            values[item.name] = _coerce_config_value(section[item.name], default)
        else:
            values[item.name] = default
    return AttackerBehaviorConfig(**values)


def _coerce_config_value(value, default):
    if isinstance(default, tuple):
        return tuple(int(item) for item in value)
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
# Condition / action nodes
# -----------------------------------------------------------------------

class HasBallControl(py_trees.behaviour.Behaviour):
    """Succeed when the ball is close to the robot AND in front of the kicker.

    Two checks:
      1. ``dist(robot, ball) <= POSSESSION_DIST`` — ball is within dribbler range.
      2. ``|angle(robot→ball) - robot.orientation| <= POSSESSION_HEADING_TOL`` —
         ball is in front of the kicker plate. Without this check, the tree
         would issue IntentKick while the ball is behind/beside the robot and
         the kick command would punch air with the ball untouched.
    """

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("HasBallControl")
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
            self._tree._possession_ticks_by_robot[bb.robot_id] = 0
            self._tree._possession_last_tick_by_robot[bb.robot_id] = self._tree._tick_index
            return py_trees.common.Status.FAILURE

        # Heading: ball must be in front of the kicker.
        angle_to_ball = math.atan2(dy, dx)
        err = (angle_to_ball - robot.orientation + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > config.possession_heading_tol:
            self._tree._possession_ticks_by_robot[bb.robot_id] = 0
            self._tree._possession_last_tick_by_robot[bb.robot_id] = self._tree._tick_index
            return py_trees.common.Status.FAILURE

        last_tick = self._tree._possession_last_tick_by_robot.get(bb.robot_id)
        ticks = self._tree._possession_ticks_by_robot.get(bb.robot_id, 0)
        if last_tick != self._tree._tick_index - 1:
            ticks = 0
        self._tree._possession_ticks_by_robot[bb.robot_id] = ticks + 1
        self._tree._possession_last_tick_by_robot[bb.robot_id] = self._tree._tick_index
        return py_trees.common.Status.SUCCESS


class HasSettledPossession(py_trees.behaviour.Behaviour):
    """Succeed only after brief continuous ball control."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("HasSettledPossession")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        ticks = self._tree._possession_ticks_by_robot.get(bb.robot_id, 0)
        if ticks >= self._tree.behavior_config.shot_settle_ticks:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class HasClearShot(py_trees.behaviour.Behaviour):
    """Succeed when shooting is safe: clear corridor, close enough, and facing goal.

    Three checks:
      1. No opponent within ``SHOT_CORRIDOR_RADIUS`` of the ball→goal line.
      2. Robot is within ``SHOOT_DIST_THRESHOLD`` of the goal.
      3. Robot is facing the goal within ``SHOT_HEADING_TOL``.
    """

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("HasClearShot")
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

        dist_to_goal = math.hypot(
            robot.position[0] - goal[0], robot.position[1] - goal[1]
        )
        config = self._tree.behavior_config
        if dist_to_goal > config.shoot_dist_threshold:
            return py_trees.common.Status.FAILURE

        angle_to_goal = math.atan2(
            goal[1] - robot.position[1], goal[0] - robot.position[0]
        )
        heading_err = (angle_to_goal - robot.orientation + math.pi) % (2 * math.pi) - math.pi
        if abs(heading_err) > config.shot_heading_tol:
            return py_trees.common.Status.FAILURE

        ball = snap.ball_position
        for opp in snap.opponent_robots:
            if (
                _point_to_segment_dist(opp.position, ball, goal)
                <= config.shot_corridor_radius
            ):
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS


class ChaseBall(py_trees.behaviour.Behaviour):
    """Write IntentMove(target=ball_position, orientation=toward_ball). SUCCESS.

    Fallback when we don't have ball control. Crucially, this sets
    ``target_orientation`` to the robot→ball bearing so the robot rotates to
    face the ball as it closes in. Without this, the move_to skill defaults
    target_orientation to 0.0 — the robot stays pointed east, holonomically
    drifts toward the ball with the ball arriving behind/beside the kicker,
    and HasBallControl never succeeds because its heading check fails.
    """

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("ChaseBall")
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
        speed = None
        if not self._is_closest_to_ball(snap, robot, bb.robot_id):
            speed = self._tree.behavior_config.chase_slow_speed
        bb.current_intent = IntentMove(
            target_pos=snap.ball_position,
            target_orientation=angle_to_ball,
            max_speed=speed,
        )
        bb.intent_source = "ChaseBall"
        return py_trees.common.Status.SUCCESS

    @staticmethod
    def _is_closest_to_ball(snap: Snapshot, robot, my_id: int) -> bool:
        bx, by = snap.ball_position
        my_dist = math.hypot(robot.position[0] - bx, robot.position[1] - by)
        for r in snap.own_robots:
            if r.robot_id == GOALIE_ID or r.robot_id == my_id:
                continue
            d = math.hypot(r.position[0] - bx, r.position[1] - by)
            if d < my_dist or (d == my_dist and r.robot_id < my_id):
                return False
        return True


class IsBallInRangeOrMove(py_trees.behaviour.Behaviour):
    """Check if ball is within range; write IntentMove and fail if not.

    SUCCESS → attacker is close enough to the ball; proceed to PassPlaySelector.
    FAILURE → attacker is too far; writes IntentMove(ball_position) so the
              caller knows to move, then returns FAILURE to halt the root Sequence.
    """

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("IsBallInRange")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        dist = math.hypot(
            snap.ball_position[0] - robot.position[0],
            snap.ball_position[1] - robot.position[1],
        )
        if dist <= self._tree.behavior_config.ball_in_range_threshold:
            return py_trees.common.Status.SUCCESS

        # Ball out of range — write move intent and signal failure so the
        # root Sequence stops (PassPlaySelector is not reached).
        bb.current_intent = IntentMove(
            target_pos=snap.ball_position,
            target_orientation=None,
        )
        bb.intent_source = "IsBallInRangeOrMove"
        return py_trees.common.Status.FAILURE


class IsSupporterAvailable(py_trees.behaviour.Behaviour):
    """Succeed when at least one supporter robot is in the snapshot."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("IsSupporterAvailable")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        if snap is None:
            return py_trees.common.Status.FAILURE
        for robot in snap.own_robots:
            if robot.robot_id in self._tree.behavior_config.supporter_role_ids:
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class PassToSupporter(py_trees.behaviour.Behaviour):
    """Write IntentPass targeting the first available supporter."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("PassToSupporter")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        for robot in snap.own_robots:
            if robot.robot_id in self._tree.behavior_config.supporter_role_ids:
                bb.current_intent = IntentPass(
                    target_robot_id=robot.robot_id,
                    target_pos=robot.position,
                )
                bb.intent_source = "PassToSupporter"
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class ShouldLookForPass(py_trees.behaviour.Behaviour):
    """Succeed when shooting confidence is low enough to consider a pass."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("ShouldLookForPass")
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
        field_scale = _field_scale(snap, goal)
        pressure_radius = field_scale * self._tree.behavior_config.pass_pressure_radius_frac
        under_pressure = (
            _nearest_opponent_distance(snap, robot.position) <= pressure_radius
        )
        shot_blocked = _goal_lane_blocked(
            snap,
            goal,
            self._tree.behavior_config,
        )

        if shot_blocked or under_pressure:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class FindOpenPassTarget(py_trees.behaviour.Behaviour):
    """Pick a teammate with a clear lane and enough space to receive."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("FindOpenPassTarget")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        target_id, target_pos = _find_best_pass_target(
            snap,
            bb.robot_id,
            self._tree.goal_position,
            self._tree.behavior_config,
        )
        if target_id is None or target_pos is None:
            return py_trees.common.Status.FAILURE

        self._tree._pass_target_id = target_id
        self._tree._pass_target_pos = target_pos
        return py_trees.common.Status.SUCCESS


class DribbleTowardPassTarget(py_trees.behaviour.Behaviour):
    """Hold the ball while turning toward the selected pass target."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("DribbleTowardPassTarget")
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
        err = (
            angle_to_target - robot.orientation + math.pi
        ) % (2 * math.pi) - math.pi
        if abs(err) <= self._tree.behavior_config.pass_orient_tol:
            return py_trees.common.Status.SUCCESS

        bb.current_intent = IntentDribble(target_pos=self._tree._pass_target_pos)
        bb.intent_source = "DribbleTowardPassTarget"
        return py_trees.common.Status.RUNNING


class PassToOpenTeammate(py_trees.behaviour.Behaviour):
    """Write IntentPass to the target selected by FindOpenPassTarget."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("PassToOpenTeammate")
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
        bb.intent_source = "PassToOpenTeammate"
        return py_trees.common.Status.SUCCESS


class HoldPossession(py_trees.behaviour.Behaviour):
    """Write IntentDribble toward goal when no pass is available."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("HoldPossession")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentDribble(target_pos=self._tree.goal_position)
        bb.intent_source = "HoldPossession"
        return py_trees.common.Status.SUCCESS


class IsCloseToGoal(py_trees.behaviour.Behaviour):
    """Succeed only if the robot is within SHOOT_DIST_THRESHOLD of the goal."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("IsCloseToGoal")
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
        return py_trees.common.Status.SUCCESS


class IsBallInOwnHalf(py_trees.behaviour.Behaviour):
    """Succeed when the ball is in our own half of the field."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("IsBallInOwnHalf")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        if snap is None:
            return py_trees.common.Status.FAILURE
        bx = snap.ball_position[0]
        if self._tree.us_positive:
            return py_trees.common.Status.SUCCESS if bx > 0 else py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.SUCCESS if bx < 0 else py_trees.common.Status.FAILURE


class WaitNearGoal(py_trees.behaviour.Behaviour):
    """Hold position in front of the opponent penalty box, facing the ball.

    The wait position sits at the penalty box edge (WAIT_X from centre),
    tracking the ball's y position so the attacker shifts laterally.
    Clamped to the opponent's half (between half line and penalty box).
    """

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("WaitNearGoal")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE
        wait_x = (
            -self._tree.behavior_config.wait_x
            if self._tree.us_positive
            else self._tree.behavior_config.wait_x
        )
        wait_y_limit = self._tree.behavior_config.wait_y_limit
        wait_y = max(-wait_y_limit, min(wait_y_limit, snap.ball_position[1]))
        angle_to_ball = math.atan2(
            snap.ball_position[1] - robot.position[1],
            snap.ball_position[0] - robot.position[0],
        )
        bb.current_intent = IntentMove(
            target_pos=(wait_x, wait_y),
            target_orientation=angle_to_ball,
        )
        bb.intent_source = "WaitNearGoal"
        return py_trees.common.Status.SUCCESS


class ShootAtGoal(py_trees.behaviour.Behaviour):
    """Write IntentKick toward goal."""

    def __init__(self, tree_ref: AttackerTree) -> None:
        super().__init__("ShootAtGoal")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentKick(target_pos=self._tree.goal_position)
        bb.intent_source = "ShootAtGoal"
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# AttackerTree
# -----------------------------------------------------------------------

class AttackerTree:
    """Wrapper around the Attacker py_trees topology.

    Usage::

        tree = AttackerTree()
        tree.set_snapshot(snapshot)   # inject world state
        tree.tick(blackboard)          # run tree; writes Intent to blackboard
        intent = blackboard.current_intent
    """

    def __init__(
        self,
        us_positive: bool = True,
        behavior_config: AttackerBehaviorConfig | None = None,
        behavior_config_file: str = BT_TUNING_FILENAME,
    ) -> None:
        self._snapshot: Snapshot | None = None
        # Shared mutable ref — nodes read the current blackboard without
        # being reconstructed each tick.
        self._blackboard_ref: list = [None]
        self.behavior_config = (
            behavior_config
            if behavior_config is not None
            else load_attacker_behavior_config(behavior_config_file)
        )
        # Convention: us_positive=True means we are on +x, so the opponent
        # goal is at -x. GOAL_POSITION = (4.5, 0) is the un-mirrored "opp
        # goal" used when us_positive=False; negate x when us_positive=True.
        self.us_positive = us_positive
        self.goal_position: tuple[float, float] = (
            (-GOAL_POSITION[0], GOAL_POSITION[1]) if us_positive
            else GOAL_POSITION
        )
        self._pass_target_id: int | None = None
        self._pass_target_pos: tuple[float, float] | None = None
        self._tick_index: int = 0
        self._possession_ticks_by_robot: dict[int, int] = {}
        self._possession_last_tick_by_robot: dict[int, int] = {}
        self.root = self._build_tree()

    # ------------------------------------------------------------------

    def set_snapshot(self, snapshot: Snapshot) -> None:
        """Inject the current world-state snapshot before ticking."""
        self._snapshot = snapshot

    def tick(self, blackboard: RobotBlackboard) -> None:
        """Tick the tree with the given per-robot blackboard.

        After this call, ``blackboard.current_intent`` contains the Intent
        produced for this tick.
        """
        self._blackboard_ref[0] = blackboard
        self._pass_target_id = None
        self._pass_target_pos = None
        self._tick_index += 1
        self.root.tick_once()

    # ------------------------------------------------------------------

    def _build_tree(self) -> py_trees.composites.Selector:
        # AttackingSelector (Selector)
        # ├── PossessionSequence (Sequence)
        # │   ├── HasBallControl
        # │   └── PossessionAction (Selector)
        # │       ├── ShootSequence (HasClearShot → IsCloseToGoal → ShootAtGoal)
        # │       └── HoldPossession (dribble toward goal — the default)
        # ├── WaitSequence (Sequence)
        # │   ├── IsBallInOwnHalf
        # │   └── WaitNearGoal
        # └── ChaseBall (ball in enemy half; closest-check speed logic)

        shoot_seq = py_trees.composites.Sequence(
            name="ShootSequence", memory=False
        )
        shoot_seq.add_children([
            HasSettledPossession(self),
            HasClearShot(self),
            ShootAtGoal(self),
        ])

        pass_seq = py_trees.composites.Sequence(
            name="PassSequence", memory=False
        )
        pass_seq.add_children([
            ShouldLookForPass(self),
            FindOpenPassTarget(self),
            DribbleTowardPassTarget(self),
            PassToOpenTeammate(self),
        ])

        possession_action = py_trees.composites.Selector(
            name="PossessionAction", memory=False
        )
        possession_action.add_children([
            shoot_seq,
            pass_seq,
            HoldPossession(self),
        ])

        possession_seq = py_trees.composites.Sequence(
            name="PossessionSequence", memory=False
        )
        possession_seq.add_children([
            HasBallControl(self),
            possession_action,
        ])

        wait_seq = py_trees.composites.Sequence(
            name="WaitSequence", memory=False
        )
        wait_seq.add_children([
            IsBallInOwnHalf(self),
            WaitNearGoal(self),
        ])

        root = py_trees.composites.Selector(
            name="AttackingSelector", memory=False
        )
        root.add_children([
            possession_seq,
            wait_seq,
            ChaseBall(self),
        ])
        return root


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _find_robot(snap: Snapshot, robot_id: int):
    for r in snap.own_robots:
        if r.robot_id == robot_id:
            return r
    return None


def _find_best_pass_target(
    snap: Snapshot,
    passer_id: int,
    goal: tuple[float, float],
    config: AttackerBehaviorConfig,
) -> tuple[int | None, tuple[float, float] | None]:
    passer = _find_robot(snap, passer_id)
    if passer is None:
        return None, None

    field_scale = _field_scale(snap, goal)
    attack_sign = 1.0 if goal[0] >= snap.ball_position[0] else -1.0
    min_pass_distance = field_scale * config.pass_min_distance_frac
    backward_allowance = field_scale * config.pass_backward_allowance_frac
    marked_distance = field_scale * config.pass_marked_distance_frac
    lane_clearance = field_scale * config.pass_lane_clearance_frac

    best_id: int | None = None
    best_pos: tuple[float, float] | None = None
    best_score = -math.inf

    for teammate in snap.own_robots:
        if teammate.robot_id in (GOALIE_ID, passer_id):
            continue

        pass_distance = math.hypot(
            teammate.position[0] - snap.ball_position[0],
            teammate.position[1] - snap.ball_position[1],
        )
        if pass_distance < min_pass_distance:
            continue

        forward_progress = (
            teammate.position[0] - snap.ball_position[0]
        ) * attack_sign
        if forward_progress < -backward_allowance:
            continue

        nearest_opp = _nearest_opponent_distance(snap, teammate.position)
        if nearest_opp < marked_distance:
            continue

        obstacles = list(snap.opponent_robots)
        obstacles.extend(
            robot
            for robot in snap.own_robots
            if robot.robot_id not in (passer_id, teammate.robot_id)
        )
        if not line_of_sight_clear(
            snap.ball_position,
            teammate.position,
            obstacles,
            clearance=lane_clearance,
        ):
            continue

        openness = 1.0 if math.isinf(nearest_opp) else _clamp(nearest_opp / field_scale)
        forward_score = _clamp(
            (forward_progress + backward_allowance)
            / (field_scale * config.pass_forward_scale_frac)
        )
        ideal_pass_distance = field_scale * config.pass_ideal_distance_frac
        distance_score = 1.0 - _clamp(
            abs(pass_distance - ideal_pass_distance)
            / (field_scale * config.pass_distance_window_frac)
        )
        goal_proximity = 1.0 - _clamp(
            math.hypot(
                teammate.position[0] - goal[0],
                teammate.position[1] - goal[1],
            )
            / field_scale
        )

        score = (
            config.pass_openness_weight * openness
            + config.pass_forward_score_weight * forward_score
            + config.pass_distance_score_weight * distance_score
            + config.pass_goal_proximity_weight * goal_proximity
        )
        if score > best_score:
            best_score = score
            best_id = teammate.robot_id
            best_pos = teammate.position

    return best_id, best_pos


def _goal_lane_blocked(
    snap: Snapshot,
    goal: tuple[float, float],
    config: AttackerBehaviorConfig,
) -> bool:
    field_scale = _field_scale(snap, goal)
    clearance = max(
        config.shot_corridor_radius,
        field_scale * config.pass_lane_clearance_frac,
    )
    return any(
        _point_to_segment_dist(opp.position, snap.ball_position, goal) <= clearance
        for opp in snap.opponent_robots
    )


def _nearest_opponent_distance(
    snap: Snapshot,
    position: tuple[float, float],
) -> float:
    if not snap.opponent_robots:
        return math.inf
    return min(
        math.hypot(position[0] - opp.position[0], position[1] - opp.position[1])
        for opp in snap.opponent_robots
    )


def _field_scale(snap: Snapshot, goal: tuple[float, float]) -> float:
    points = [snap.ball_position, goal]
    points.extend(robot.position for robot in snap.own_robots)
    points.extend(robot.position for robot in snap.opponent_robots)

    max_dist = math.hypot(FIELD_HALF_X * 2.0, FIELD_HALF_Y * 2.0)
    for i, point_a in enumerate(points):
        for point_b in points[i + 1:]:
            max_dist = max(
                max_dist,
                math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]),
            )
    return max(max_dist, 1.0)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _point_to_segment_dist(
    point: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> float:
    """Shortest distance from *point* to the line segment A→B."""
    ax, ay = seg_a
    bx, by = seg_b
    px, py = point
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)
