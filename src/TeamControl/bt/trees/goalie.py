"""Goalie behaviour tree — R008.

Topology (from docs/goalie_node.png):

    GoalieSequenceNode (Sequence)
    ├── LookAtBall          → writes IntentOrient(target_orientation=angle_to_ball)
    ├── GoalieBallSequence (Sequence)
    │   ├── GetBallHistory  → stores snap.ball_position as single-frame history
    │   └── DoBallTrajectory → v1: sets predicted_intercept to NEUTRAL_GOAL_POSITION
    └── GoToTarget          → writes IntentMove(target_pos=predicted_intercept)
        [IsBallComing stub: always FAILURE — goalie stays at neutral position]

Design notes
------------
- Snapshot is injected via ``set_snapshot()`` before each ``tick()``.
  All condition and action nodes access world state through
  ``self._tree._snapshot`` (read-only Snapshot reference).
- Blackboard is injected via ``tick(blackboard)`` using the standard
  ``_blackboard_ref`` protocol (one-element list). Nodes write
  ``_blackboard_ref[0].current_intent`` to produce their output.
- No raw motor commands are produced anywhere in this module.
- v1 simplification: DoBallTrajectory always returns NEUTRAL_GOAL_POSITION.
  IsBallComing is stubbed to always FAILURE.
"""
from __future__ import annotations

import math
import py_trees

from TeamControl.bt.contracts.blackboard import RobotBlackboard
from TeamControl.bt.contracts.intent import IntentMove, IntentOrient
from TeamControl.bt.contracts.snapshot import Snapshot

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

NEUTRAL_GOAL_POSITION: tuple[float, float] = (-4.0, 0.0)  # default goalie position in front of own goal
GOALIE_ROBOT_ID: int = 0   # robot ID 0 is always GOALIE

# us_positive convention: this tree (and the attacker/defender/supporter)
# all assume us_positive=True means "we attack +x → own goal at -x". The
# value handed to this tree at construction is sourced from
# ``run_bt_v2_process``, which NEGATES ``wm.us_positive()`` before passing it
# in — wm reports the inverse of the codebase convention in our grSim+GC
# setup. See the comment in run_bt_v2_process.py for the full story.


# -----------------------------------------------------------------------
# Condition / action nodes
# -----------------------------------------------------------------------

class LookAtBall(py_trees.behaviour.Behaviour):
    """Compute the angle from robot to ball and write IntentOrient.

    SUCCESS → always (LookAtBall never blocks the sequence).
    Writes IntentOrient(target_orientation=atan2(ball_y - robot_y, ball_x - robot_x)).
    """

    def __init__(self, tree_ref: GoalieTree) -> None:
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


class GetBallHistory(py_trees.behaviour.Behaviour):
    """Store the current ball position as single-frame history on the tree.

    v1: stores snap.ball_position as tree.ball_history.
    SUCCESS → always.
    """

    def __init__(self, tree_ref: GoalieTree) -> None:
        super().__init__("GetBallHistory")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        if snap is None:
            return py_trees.common.Status.FAILURE

        self._tree.ball_history = snap.ball_position
        return py_trees.common.Status.SUCCESS


class DoBallTrajectory(py_trees.behaviour.Behaviour):
    """Compute predicted intercept point for the goalie.

    v1 simplification: always returns NEUTRAL_GOAL_POSITION.
    Stores the result in tree.predicted_intercept.
    SUCCESS → always.
    """

    def __init__(self, tree_ref: GoalieTree) -> None:
        super().__init__("DoBallTrajectory")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        # v1: linear extrapolation not yet implemented — use neutral position
        # mirrored to the side we're actually defending.
        self._tree.predicted_intercept = self._tree.neutral_goal_position
        return py_trees.common.Status.SUCCESS


class IsBallComing(py_trees.behaviour.Behaviour):
    """Check whether the ball is heading toward the goal.

    Stub implementation — always returns FAILURE so the goalie holds
    the neutral position.

    # TODO: wire DoBallTrajectory result to determine if ball trajectory
    # intersects the goal mouth, then return SUCCESS when it does.
    """

    def __init__(self, tree_ref: GoalieTree) -> None:
        super().__init__("IsBallComing")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        # Stub: always FAILURE — goalie stays at neutral position.
        # TODO: wire DoBallTrajectory result
        return py_trees.common.Status.FAILURE


class GoToTarget(py_trees.behaviour.Behaviour):
    """Write IntentMove(target_pos=predicted_intercept) to the blackboard.

    Uses tree.predicted_intercept set by DoBallTrajectory.
    SUCCESS → always (writing the move intent is always valid).
    """

    def __init__(self, tree_ref: GoalieTree) -> None:
        super().__init__("GoToTarget")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE

        bb.current_intent = IntentMove(
            target_pos=self._tree.predicted_intercept,
            target_orientation=None,
        )
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# GoalieTree
# -----------------------------------------------------------------------

class GoalieTree:
    """Wrapper around the Goalie py_trees topology.

    Usage::

        tree = GoalieTree()
        tree.set_snapshot(snapshot)   # inject world state
        tree.tick(blackboard)          # run tree; writes Intent to blackboard
        intent = blackboard.current_intent
    """

    def __init__(self, us_positive: bool = True) -> None:
        self._snapshot: Snapshot | None = None
        # Shared mutable ref — nodes read the current blackboard without
        # being reconstructed each tick.
        self._blackboard_ref: list = [None]
        # Mirror NEUTRAL_GOAL_POSITION onto our actual half. The constant
        # assumes us_positive=True (own goal at -x); when we attack the
        # negative half instead, negate x so the goalie parks in front of
        # OUR goal, not the opponent's. This is the same convention as the
        # attacker/defender/supporter trees; see also the inversion note in
        # run_bt_v2_process.py for why the value of us_positive itself is
        # negated at the entry point.
        self.us_positive = us_positive
        self.neutral_goal_position: tuple[float, float] = (
            NEUTRAL_GOAL_POSITION if us_positive
            else (-NEUTRAL_GOAL_POSITION[0], NEUTRAL_GOAL_POSITION[1])
        )
        # v1 state: single-frame ball history and trajectory prediction
        self.ball_history: tuple[float, float] | None = None
        self.predicted_intercept: tuple[float, float] = self.neutral_goal_position
        # Build tree and expose IsBallComing node for testability
        self.is_ball_coming_node = IsBallComing(self)
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
        # GoalieBallSequence — gets ball history then predicts intercept.
        goalie_ball_seq = py_trees.composites.Sequence(
            name="GoalieBallSequence", memory=False
        )
        goalie_ball_seq.add_children([
            GetBallHistory(self),
            DoBallTrajectory(self),
        ])

        # Root: GoalieSequenceNode
        root = py_trees.composites.Sequence(
            name="GoalieSequenceNode", memory=False
        )
        root.add_children([
            LookAtBall(self),
            goalie_ball_seq,
            GoToTarget(self),
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
