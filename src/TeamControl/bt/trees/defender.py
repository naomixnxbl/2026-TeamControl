"""Defender behaviour tree — R006.

Topology (from docs/defending_node.png):

    DefendingSequenceNode (Sequence)
    ├── LookAtBall         → stashes angle_to_ball on the tree (no intent)
    ├── DefendZoneFallback (Selector — OR logic)
    │   ├── InDefendingZone  → IntentMove(hold pos, facing ball) when x<0
    │   └── GoToDefendZone   → IntentMove(defend_zone, facing ball) otherwise
    └── ChallengeSequence (Sequence)
        ├── IsCloseEnough    (Condition — reads Snapshot)
        └── ClearBall        → writes IntentKick(clear_direction)

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

import math
import py_trees

from TeamControl.bt.contracts.blackboard import RobotBlackboard
from TeamControl.bt.contracts.intent import IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.bt.tactics.line_of_sight import (
    Point,
    distance,
    face_angle,
    line_of_sight_clear,
)

# -----------------------------------------------------------------------
# Tuneable constants
# -----------------------------------------------------------------------

DEFEND_ZONE_POSITION: tuple[float, float] = (-3.0, 0.0)   # own half centre
CLOSE_ENOUGH_THRESHOLD: float = 0.6                         # metres to ball to challenge
CLEAR_DIRECTION: tuple[float, float] = (4.5, 0.0)           # kick toward opponent's half
DEFENDER_ROLE_ID: int = 1                                    # defender robot ID
GOALIE_ID: int = 0
FIELD_HALF_X: float = 4.5
FIELD_HALF_Y: float = 3.0
FIELD_MARGIN: float = 0.2
SHOT_BLOCK_FRACTION_FROM_GOAL: float = 0.30
PASS_BLOCK_FRACTION_FROM_CARRIER: float = 0.58
PASS_LANE_CLEARANCE: float = 0.18
OPPONENT_POSSESSION_MARGIN_RATIO: float = 0.03
DEFENDER_TEAMMATE_MIN_GAP: float = 0.45
DEFENDER_TEAMMATE_MAX_NUDGE: float = 0.45


# -----------------------------------------------------------------------
# Condition / action nodes
# -----------------------------------------------------------------------

class LookAtBall(py_trees.behaviour.Behaviour):
    """Compute the angle from defender to ball and stash it on the tree.

    Does NOT write an intent — earlier versions wrote IntentOrient here, which
    clobbered every downstream movement intent (the adapter translates
    IntentOrient as zero linear velocity). Subsequent movement nodes pick up
    this angle via ``self._tree.look_angle`` and apply it as the target
    orientation on their IntentMove.
    """

    def __init__(self, tree_ref: DefenderTree) -> None:
        super().__init__("LookAtBall")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        self._tree.look_angle = math.atan2(
            snap.ball_position[1] - robot.position[1],
            snap.ball_position[0] - robot.position[0],
        )
        return py_trees.common.Status.SUCCESS


class InDefendingZone(py_trees.behaviour.Behaviour):
    """Hold shape or block a lane when defender is in its own half.

    If an opponent is controlling/contesting the ball, write an IntentMove to
    block the shot lane or a dangerous pass lane. Otherwise hold current
    position facing the ball.
    If x >= 0 (out of zone): return FAILURE so GoToDefendZone runs next.
    """

    def __init__(self, tree_ref: DefenderTree) -> None:
        super().__init__("InDefendingZone")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        # "In zone" means we're on our own half. Under main's convention,
        # us_positive=True ⇒ own half is +x; us_positive=False ⇒ own half is -x.
        in_zone = (
            robot.position[0] > 0.0
            if self._tree.us_positive
            else robot.position[0] < 0.0
        )
        if in_zone:
            _write_defensive_position_intent(self._tree, bb, robot)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class GoToDefendZone(py_trees.behaviour.Behaviour):
    """Write IntentMove toward the tactical lane or defend zone and return SUCCESS.

    Returning SUCCESS allows the parent Selector (DefendZoneFallback) to
    succeed, which lets the root Sequence proceed to ChallengeSequence.
    """

    def __init__(self, tree_ref: DefenderTree) -> None:
        super().__init__("GoToDefendZone")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        _write_defensive_position_intent(self._tree, bb, robot)
        return py_trees.common.Status.SUCCESS


class IsCloseEnough(py_trees.behaviour.Behaviour):
    """Succeed when the distance from robot to ball is within CLOSE_ENOUGH_THRESHOLD."""

    def __init__(self, tree_ref: DefenderTree) -> None:
        super().__init__("IsCloseEnough")
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
        if dist <= CLOSE_ENOUGH_THRESHOLD:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class ClearBall(py_trees.behaviour.Behaviour):
    """Write IntentKick toward CLEAR_DIRECTION and return SUCCESS."""

    def __init__(self, tree_ref: DefenderTree) -> None:
        super().__init__("ClearBall")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE

        bb.current_intent = IntentKick(target_pos=self._tree.clear_direction)
        bb.intent_source = "ClearBall"
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# DefenderTree
# -----------------------------------------------------------------------

class DefenderTree:
    """Wrapper around the Defender py_trees topology.

    Usage::

        tree = DefenderTree()
        tree.set_snapshot(snapshot)   # inject world state
        tree.tick(blackboard)          # run tree; writes Intent to blackboard
        intent = blackboard.current_intent
    """

    def __init__(self, us_positive: bool = True) -> None:
        self._snapshot: Snapshot | None = None
        # Shared mutable ref — nodes read the current blackboard without
        # being reconstructed each tick.
        self._blackboard_ref: list = [None]
        # Updated each tick by LookAtBall; consumed by movement nodes as the
        # target_orientation for their IntentMove. 0.0 is a safe initial value
        # in case a movement node runs before LookAtBall (shouldn't happen
        # given the sequence order, but defensive).
        self.look_angle: float = 0.0
        # Mirror side-dependent constants onto the half we're actually
        # defending. Module constants assume us_positive=True (own goal at
        # negative x). When we attack the negative half instead, negate x so
        # the defender parks in OUR half and clears toward the OPPONENT goal.
        self.us_positive = us_positive
        # Convention: us_positive=True means we are on +x; own half is +x,
        # opp goal is at -x. The module constants are authored for the
        # us_positive=False case, so negate x when us_positive=True.
        self.defend_zone_position: tuple[float, float] = (
            (-DEFEND_ZONE_POSITION[0], DEFEND_ZONE_POSITION[1]) if us_positive
            else DEFEND_ZONE_POSITION
        )
        self.clear_direction: tuple[float, float] = (
            (-CLEAR_DIRECTION[0], CLEAR_DIRECTION[1]) if us_positive
            else CLEAR_DIRECTION
        )
        self.own_goal_position: tuple[float, float] = (
            (FIELD_HALF_X, 0.0) if us_positive
            else (-FIELD_HALF_X, 0.0)
        )
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
        self.root.tick_once()

    # ------------------------------------------------------------------

    def _build_tree(self) -> py_trees.composites.Sequence:
        # DefendZoneFallback — Selector: succeed if in zone, else move to zone.
        defend_zone_fallback = py_trees.composites.Selector(
            name="DefendZoneFallback", memory=False
        )
        defend_zone_fallback.add_children([
            InDefendingZone(self),
            GoToDefendZone(self),
        ])

        # ChallengeSequence — Sequence: only clear ball if close enough.
        challenge_sequence = py_trees.composites.Sequence(
            name="ChallengeSequence", memory=False
        )
        challenge_sequence.add_children([
            IsCloseEnough(self),
            ClearBall(self),
        ])

        # Root: DefendingSequenceNode
        root = py_trees.composites.Sequence(
            name="DefendingSequenceNode", memory=False
        )
        root.add_children([
            LookAtBall(self),
            defend_zone_fallback,
            challenge_sequence,
        ])
        return root


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _find_robot(snap: Snapshot, robot_id: int) -> RobotState | None:
    for r in snap.own_robots:
        if r.robot_id == robot_id:
            return r
    return None


def _write_defensive_position_intent(
    tree: DefenderTree,
    bb: RobotBlackboard,
    robot: RobotState,
) -> None:
    snap = tree._snapshot
    if snap is None:
        return

    carrier = _opponent_ball_carrier(snap, tree)
    if carrier is None:
        target, source = _fallback_defend_target(tree, robot)
        target = _space_from_teammates(target, snap, robot, tree.us_positive)
        bb.current_intent = IntentMove(
            target_pos=target,
            target_orientation=tree.look_angle,
        )
        bb.intent_source = source
        return

    if _defensive_rank(snap, robot.robot_id, tree.own_goal_position) == 0:
        target = _shot_block_target(carrier.position, tree.own_goal_position)
        source = "BlockShotLane"
    else:
        receiver = _dangerous_receiver(
            snap,
            carrier,
            tree.own_goal_position,
            robot.robot_id,
        )
        if receiver is None:
            target = _shot_block_target(carrier.position, tree.own_goal_position)
            source = "BlockShotLane"
        else:
            target = _pass_block_target(carrier.position, receiver.position)
            source = "BlockPassLane"

    target = _clamp_defensive_target(target, tree.us_positive)
    target = _space_from_teammates(target, snap, robot, tree.us_positive)
    bb.current_intent = IntentMove(
        target_pos=target,
        target_orientation=face_angle(robot.position, carrier.position),
    )
    bb.intent_source = source


def _opponent_ball_carrier(
    snap: Snapshot,
    tree: DefenderTree,
) -> RobotState | None:
    if not snap.opponent_robots:
        return None

    carrier = min(
        snap.opponent_robots,
        key=lambda r: distance(r.position, snap.ball_position),
    )
    opponent_dist = distance(carrier.position, snap.ball_position)
    own_dist = min(
        (distance(r.position, snap.ball_position) for r in snap.own_robots),
        default=math.inf,
    )
    possession_margin = _field_scale(tree) * OPPONENT_POSSESSION_MARGIN_RATIO
    if opponent_dist <= own_dist + possession_margin:
        return carrier
    return None


def _defensive_rank(
    snap: Snapshot,
    robot_id: int,
    own_goal: Point,
) -> int | None:
    candidates = sorted(
        (r for r in snap.own_robots if r.robot_id != GOALIE_ID),
        key=lambda r: (distance(r.position, own_goal), r.robot_id),
    )
    for index, candidate in enumerate(candidates):
        if candidate.robot_id == robot_id:
            return index
    return None


def _dangerous_receiver(
    snap: Snapshot,
    carrier: RobotState,
    own_goal: Point,
    self_robot_id: int,
) -> RobotState | None:
    candidates = [
        r for r in snap.opponent_robots
        if r.robot_id != carrier.robot_id
    ]
    if not candidates:
        return None

    blockers = [
        r for r in snap.own_robots
        if r.robot_id != self_robot_id
    ]

    def receiver_score(receiver: RobotState) -> tuple[int, float, float, int]:
        lane_open = line_of_sight_clear(
            carrier.position,
            receiver.position,
            blockers,
            clearance=PASS_LANE_CLEARANCE,
        )
        return (
            0 if lane_open else 1,
            distance(receiver.position, own_goal),
            distance(carrier.position, receiver.position),
            receiver.robot_id,
        )

    return min(candidates, key=receiver_score)


def _fallback_defend_target(
    tree: DefenderTree,
    robot: RobotState,
) -> tuple[Point, str]:
    if _is_in_own_half(robot.position, tree.us_positive):
        return robot.position, "HoldDefendZone"
    return tree.defend_zone_position, "GoToDefendZone"


def _shot_block_target(carrier: Point, own_goal: Point) -> Point:
    return _interpolate(own_goal, carrier, SHOT_BLOCK_FRACTION_FROM_GOAL)


def _pass_block_target(carrier: Point, receiver: Point) -> Point:
    return _interpolate(carrier, receiver, PASS_BLOCK_FRACTION_FROM_CARRIER)


def _space_from_teammates(
    target: Point,
    snap: Snapshot,
    robot: RobotState,
    us_positive: bool,
) -> Point:
    push_x = 0.0
    push_y = 0.0

    for teammate in snap.own_robots:
        if teammate.robot_id == robot.robot_id:
            continue

        dx = target[0] - teammate.position[0]
        dy = target[1] - teammate.position[1]
        dist = math.hypot(dx, dy)
        if dist >= DEFENDER_TEAMMATE_MIN_GAP:
            continue

        if dist < 1e-9:
            angle = robot.robot_id * 2.399963229728653
            dx = math.cos(angle)
            dy = math.sin(angle)
            dist = 1.0

        strength = DEFENDER_TEAMMATE_MIN_GAP - dist
        push_x += (dx / dist) * strength
        push_y += (dy / dist) * strength

    push_mag = math.hypot(push_x, push_y)
    if push_mag < 1e-9:
        return target

    if push_mag > DEFENDER_TEAMMATE_MAX_NUDGE:
        scale = DEFENDER_TEAMMATE_MAX_NUDGE / push_mag
        push_x *= scale
        push_y *= scale

    return _clamp_defensive_target(
        (target[0] + push_x, target[1] + push_y),
        us_positive,
    )


def _interpolate(start: Point, end: Point, fraction: float) -> Point:
    return (
        start[0] + ((end[0] - start[0]) * fraction),
        start[1] + ((end[1] - start[1]) * fraction),
    )


def _clamp_defensive_target(target: Point, us_positive: bool) -> Point:
    x = max(-FIELD_HALF_X + FIELD_MARGIN, min(FIELD_HALF_X - FIELD_MARGIN, target[0]))
    y = max(-FIELD_HALF_Y + FIELD_MARGIN, min(FIELD_HALF_Y - FIELD_MARGIN, target[1]))

    # Defenders hold the defensive half. If a pass lane extends into the
    # attacking half, meet it near midfield instead of abandoning shape.
    if us_positive:
        x = max(0.0, x)
    else:
        x = min(0.0, x)
    return (x, y)


def _is_in_own_half(point: Point, us_positive: bool) -> bool:
    return point[0] > 0.0 if us_positive else point[0] < 0.0


def _field_scale(tree: DefenderTree) -> float:
    return max(distance(tree.own_goal_position, tree.clear_direction), 1.0)
