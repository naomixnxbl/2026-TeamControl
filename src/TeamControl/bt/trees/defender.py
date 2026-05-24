"""Defender behaviour tree — R006.

Topology (from docs/defending_node.png):

    DefendingSequenceNode (Sequence)
    ├── LookAtBall         → writes IntentOrient(angle_to_ball)
    ├── DefendZoneFallback (Selector — OR logic)
    │   ├── InDefendingZone  (Condition — reads Snapshot)
    │   └── GoToDefendZone   → writes IntentMove(defend_zone_pos) on failure
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
from TeamControl.bt.contracts.intent import IntentKick, IntentMove, IntentOrient
from TeamControl.bt.contracts.snapshot import Snapshot

# -----------------------------------------------------------------------
# Tuneable constants
# -----------------------------------------------------------------------

DEFEND_ZONE_POSITION: tuple[float, float] = (-3.0, 0.0)   # own half centre
CLOSE_ENOUGH_THRESHOLD: float = 0.6                         # metres to ball to challenge
CLEAR_DIRECTION: tuple[float, float] = (4.5, 0.0)           # kick toward opponent's half
DEFENDER_ROLE_ID: int = 1                                    # defender robot ID


# -----------------------------------------------------------------------
# Condition / action nodes
# -----------------------------------------------------------------------

class LookAtBall(py_trees.behaviour.Behaviour):
    """Compute the angle to the ball and write IntentOrient; always returns SUCCESS.

    This node runs first every tick so the robot always faces the ball.
    Later nodes may overwrite current_intent with a more specific action.
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

        angle = math.atan2(
            snap.ball_position[1] - robot.position[1],
            snap.ball_position[0] - robot.position[0],
        )
        bb.current_intent = IntentOrient(target_orientation=angle)
        return py_trees.common.Status.SUCCESS


class InDefendingZone(py_trees.behaviour.Behaviour):
    """Succeed when the defender robot is in its own half (x < 0)."""

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

        if robot.position[0] < 0.0:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class GoToDefendZone(py_trees.behaviour.Behaviour):
    """Write IntentMove toward the defend zone and return SUCCESS.

    Returning SUCCESS allows the parent Selector (DefendZoneFallback) to
    succeed, which lets the root Sequence proceed to ChallengeSequence.
    """

    def __init__(self, tree_ref: DefenderTree) -> None:
        super().__init__("GoToDefendZone")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE

        bb.current_intent = IntentMove(
            target_pos=DEFEND_ZONE_POSITION,
            target_orientation=None,
        )
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

        bb.current_intent = IntentKick(target_pos=CLEAR_DIRECTION)
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

    def __init__(self) -> None:
        self._snapshot: Snapshot | None = None
        # Shared mutable ref — nodes read the current blackboard without
        # being reconstructed each tick.
        self._blackboard_ref: list = [None]
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

def _find_robot(snap: Snapshot, robot_id: int):
    for r in snap.own_robots:
        if r.robot_id == robot_id:
            return r
    return None
