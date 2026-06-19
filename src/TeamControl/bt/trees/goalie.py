"""Goalie behaviour tree.

Topology:

    GoalieRoot (Selector)
    ├── KickSequence (Sequence)
    │   ├── IsKickReady              — robot within KICK_DIST of ball
    │   └── KickTargetSelector (Selector)
    │       ├── PassSequence (Sequence)
    │       │   ├── FindOpenMate     — teammate: no nearby enemy, past box in attack direction
    │       │   └── PassToMate       → IntentKick(mate_pos)
    │       ├── KickFieldCenter      → IntentKick toward safe field zone (away from box)
    │       └── KickSideline         → IntentKick toward nearest sideline (OUT of field, y dir)
    ├── PushSequence (Sequence)      — kicker not available: physically ram ball out of box
    │   ├── IsBallClose              — ball within RUSH_DIST of own goal
    │   └── PushToBoxEdge            → IntentMove(box_far_x, ball_y) — drives through ball
    └── HoldBarrier                  → IntentMove(goal_line + INTERCEPT_OFFSET, ball_y)
"""
from __future__ import annotations

import math
import py_trees

from TeamControl.bt.contracts.blackboard import RobotBlackboard
from TeamControl.bt.contracts.intent import IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import Snapshot
from TeamControl.world.field_config import (
    DEFENCE_X_MM, DEFENCE_Y_MM, FIELD_LENGTH_MM, FIELD_WIDTH_MM, GOAL_HALF_WIDTH_MM,
)

_HALF_LEN_M: float  = FIELD_LENGTH_MM / 2.0 / 1000.0
_HALF_WID_M: float  = FIELD_WIDTH_MM  / 2.0 / 1000.0
_GOAL_HW_M: float   = GOAL_HALF_WIDTH_MM / 1000.0
_BOX_DEPTH_M: float = DEFENCE_X_MM / 1000.0
_BOX_HW_M: float    = DEFENCE_Y_MM / 1000.0

# -----------------------------------------------------------------------
# Tuneable constants
# -----------------------------------------------------------------------

GOALIE_ROBOT_ID: int = 0

# Goalie holds at this distance inside the goal line when the ball is far.
INTERCEPT_OFFSET_M: float = 0.5

# Rush / push when ball is within this distance of our own goal.
RUSH_DIST_M: float = 1.2

# Kick triggers when robot centre is within this distance of the ball.
KICK_DIST_M: float = 0.15

# Teammate is "marked" (skip for pass) if any enemy is within this radius.
OPEN_MATE_ENEMY_RADIUS_M: float = 0.6

# Teammate must be at least this far past our box (in attack direction) to be
# a valid pass target — avoids passing back into our own danger zone.
OPEN_MATE_MIN_ATTACK_DIST_M: float = 0.5

# Sideline kick overshoots the field edge by this much to guarantee out-of-bounds.
SIDELINE_OVERSHOOT_M: float = 0.3


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _find_robot(snap: Snapshot, robot_id: int):
    for r in snap.own_robots:
        if r.robot_id == robot_id:
            return r
    return None


def _find_open_mate(snap: Snapshot, goalie_id: int, attack_sign: int):
    """Return position of the most advanced open teammate, or None."""
    best_pos = None
    best_score = -float("inf")
    for mate in snap.own_robots:
        if mate.robot_id == goalie_id:
            continue
        marked = any(
            math.hypot(
                mate.position[0] - e.position[0],
                mate.position[1] - e.position[1],
            ) < OPEN_MATE_ENEMY_RADIUS_M
            for e in snap.enemy_robots
        )
        if marked:
            continue
        attack_dist = mate.position[0] * attack_sign
        if attack_dist < OPEN_MATE_MIN_ATTACK_DIST_M:
            continue
        if attack_dist > best_score:
            best_score = attack_dist
            best_pos = mate.position
    return best_pos


def _face_angle(from_pos, to_pos) -> float:
    return math.atan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])


# -----------------------------------------------------------------------
# Condition nodes
# -----------------------------------------------------------------------

class IsKickReady(py_trees.behaviour.Behaviour):
    """Succeed when the goalie is within KICK_DIST_M of the ball."""

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("IsKickReady")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb   = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE
        dist = math.hypot(
            robot.position[0] - snap.ball_position[0],
            robot.position[1] - snap.ball_position[1],
        )
        return (
            py_trees.common.Status.SUCCESS
            if dist <= KICK_DIST_M
            else py_trees.common.Status.FAILURE
        )


class IsBallClose(py_trees.behaviour.Behaviour):
    """Succeed when the ball is within RUSH_DIST_M of our own goal."""

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("IsBallClose")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        if snap is None:
            return py_trees.common.Status.FAILURE
        dist = math.hypot(
            snap.ball_position[0] - self._tree._own_goal_x,
            snap.ball_position[1],
        )
        return (
            py_trees.common.Status.SUCCESS
            if dist <= RUSH_DIST_M
            else py_trees.common.Status.FAILURE
        )


class FindOpenMate(py_trees.behaviour.Behaviour):
    """Find the best open teammate; store position in tree._pass_target.

    FAILURE when no open mate exists.
    """

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("FindOpenMate")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb   = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        pos = _find_open_mate(snap, bb.robot_id, self._tree._attack_sign)
        if pos is None:
            return py_trees.common.Status.FAILURE
        self._tree._pass_target = pos
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# Action nodes
# -----------------------------------------------------------------------

class PassToMate(py_trees.behaviour.Behaviour):
    """Kick toward the open teammate stored by FindOpenMate."""

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("PassToMate")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None or self._tree._pass_target is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentKick(target_pos=self._tree._pass_target)
        return py_trees.common.Status.SUCCESS


class KickFieldCenter(py_trees.behaviour.Behaviour):
    """Kick the ball away from the box toward the safe field zone.

    Target: midfield area on the opposite side from the ball's current y
    (avoid returning it toward a packed side).  Always returns SUCCESS.
    """

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("KickFieldCenter")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb   = self._tree._blackboard_ref[0]
        snap = self._tree._snapshot
        if bb is None:
            return py_trees.common.Status.FAILURE
        ball_y = snap.ball_position[1] if snap else 0.0
        # Kick to the far side of midfield from where the ball currently is.
        clear_y = -math.copysign(_HALF_WID_M * 0.35, ball_y)
        bb.current_intent = IntentKick(
            target_pos=(self._tree._clear_target_x, clear_y)
        )
        return py_trees.common.Status.SUCCESS


class KickSideline(py_trees.behaviour.Behaviour):
    """Kick the ball OUT of the field via the nearest sideline (y direction).

    Target is past the sideline on whichever side the ball is on.
    This is NOT toward the goal line — it exits through the side wall.
    Always returns SUCCESS (last-resort kick).
    """

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("KickSideline")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb   = self._tree._blackboard_ref[0]
        snap = self._tree._snapshot
        if bb is None:
            return py_trees.common.Status.FAILURE
        ball_y = snap.ball_position[1] if snap else 0.0
        ball_x = snap.ball_position[0] if snap else self._tree._barrier_x
        # Kick sideways past the sideline from the ball's x position.
        sideline_y = math.copysign(_HALF_WID_M + SIDELINE_OVERSHOOT_M, ball_y or 1.0)
        bb.current_intent = IntentKick(target_pos=(ball_x, sideline_y))
        return py_trees.common.Status.SUCCESS


class PushToBoxEdge(py_trees.behaviour.Behaviour):
    """Drive through the ball toward the far box edge to physically push it clear.

    Used when the kicker is unavailable or hasn't fired.  The robot drives
    to (box_far_x, ball_y), which puts the ball in its path and rams it
    toward the box edge.  The goalie never leaves the penalty box.
    """

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("PushToBoxEdge")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb   = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        # Push target: box far edge at the ball's y, clamped to box half-width.
        push_x = (
            self._tree._box_x_max
            if self._tree._attack_sign > 0
            else self._tree._box_x_min
        )
        push_y = max(-_BOX_HW_M, min(_BOX_HW_M, snap.ball_position[1]))
        target = (push_x, push_y)
        bb.current_intent = IntentMove(
            target_pos=target,
            target_orientation=_face_angle(snap.ball_position, target),
        )
        return py_trees.common.Status.SUCCESS


class HoldBarrier(py_trees.behaviour.Behaviour):
    """Hold INTERCEPT_OFFSET_M off the goal line, tracking ball y.

    The barrier line acts as the goalie's neutral coverage zone.
    Position is clamped to goal-mouth width in y.
    """

    def __init__(self, tree_ref: "GoalieTree") -> None:
        super().__init__("HoldBarrier")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb   = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        ball_y = snap.ball_position[1] if snap else 0.0
        clamped_y = max(-_GOAL_HW_M, min(_GOAL_HW_M, ball_y))
        target = (self._tree._barrier_x, clamped_y)
        angle = _face_angle(target, snap.ball_position) if snap else 0.0
        bb.current_intent = IntentMove(target_pos=target, target_orientation=angle)
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# GoalieTree
# -----------------------------------------------------------------------

class GoalieTree:
    """Goalie behaviour tree.

    Usage::

        tree = GoalieTree(us_positive=False)
        tree.set_snapshot(snapshot)
        tree.tick(blackboard)
        intent = blackboard.current_intent
    """

    def __init__(self, us_positive: bool = False) -> None:
        self._snapshot: Snapshot | None = None
        self._blackboard_ref: list = [None]

        self._own_goal_x: float = _HALF_LEN_M if us_positive else -_HALF_LEN_M
        self._attack_sign: int  = -1 if us_positive else 1

        # Penalty box x bounds — goalie never leaves this range.
        box_far_x = self._own_goal_x + self._attack_sign * _BOX_DEPTH_M
        self._box_x_min: float = min(self._own_goal_x, box_far_x)
        self._box_x_max: float = max(self._own_goal_x, box_far_x)

        # Barrier x: INTERCEPT_OFFSET_M inside the goal line.
        self._barrier_x: float = self._own_goal_x + self._attack_sign * INTERCEPT_OFFSET_M

        # Field-center kick target x: just past the box edge.
        self._clear_target_x: float = (
            self._own_goal_x + self._attack_sign * (_BOX_DEPTH_M * 1.10)
        )

        # Filled each tick by FindOpenMate when a pass is possible.
        self._pass_target: tuple[float, float] | None = None

        self.root = self._build_tree()

    # ------------------------------------------------------------------

    def set_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot

    def tick(self, blackboard: RobotBlackboard) -> None:
        self._blackboard_ref[0] = blackboard
        self._pass_target = None
        self.root.tick_once()

    # ------------------------------------------------------------------

    def _build_tree(self):
        # Pass sub-tree.
        pass_seq = py_trees.composites.Sequence("PassSequence", memory=False)
        pass_seq.add_children([FindOpenMate(self), PassToMate(self)])

        # Kick target priority: pass > field clear > sideline out.
        kick_target = py_trees.composites.Selector("KickTargetSelector", memory=False)
        kick_target.add_children([
            pass_seq,
            KickFieldCenter(self),
            KickSideline(self),      # always succeeds — ultimate kick fallback
        ])

        # Kick only fires when robot is at the ball.
        kick_seq = py_trees.composites.Sequence("KickSequence", memory=False)
        kick_seq.add_children([IsKickReady(self), kick_target])

        # Physical push when kicker unavailable — drives robot through ball.
        push_seq = py_trees.composites.Sequence("PushSequence", memory=False)
        push_seq.add_children([IsBallClose(self), PushToBoxEdge(self)])

        # Root: kick > push > hold.
        root = py_trees.composites.Selector("GoalieRoot", memory=False)
        root.add_children([kick_seq, push_seq, HoldBarrier(self)])
        return root
