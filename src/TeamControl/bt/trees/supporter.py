"""Supporter behaviour tree — R007.

Topology (from docs/supporting_node.png):

    SupportingSelectorNode (Selector)
    ├── MoveToSpace        → writes IntentMove(open_space_pos)
    ├── ReceiveBallSequence (Sequence)
    │   ├── IsBallComing   (Condition — STUBBED, always returns FAILURE for v1)
    │   └── ReceiveBall    → writes IntentReceive()
    └── BlockOpponent      → writes IntentMove(blocking_pos)

Design notes
------------
- Snapshot is injected via ``set_snapshot()`` before each ``tick()``.
  All condition and action nodes access world state through
  ``self._tree._snapshot`` (read-only Snapshot reference).
- Blackboard is injected via ``tick(blackboard)`` using the standard
  ``_blackboard_ref`` protocol (one-element list). Nodes write
  ``_blackboard_ref[0].current_intent`` to produce their output.
- No raw motor commands are produced anywhere in this module.

v1 behaviour
------------
Since ``IsBallComing`` is always FAILURE, ``ReceiveBallSequence`` always
fails. The Selector therefore always succeeds on ``MoveToSpace`` (first
child), which writes ``IntentMove(target_pos=(1.0, 2.0))``. This means
the supporter ALWAYS produces ``IntentMove`` in v1. This is correct
per spec R007.

Known issue / planned redesign (v2)
-------------------------------------
The v1 supporter is effectively inert — it moves to one hardcoded position
and never reacts to the ball or teammates. The planned v2 topology:

    SupporterRoot (Selector)
    ├── BallPossessionSequence (Sequence)
    │   ├── IsClosestToBall    → SUCCESS only if this robot is nearest to ball
    │   └── GoToBall           → IntentMove(ball_position)
    ├── PossessionSequence (Sequence)
    │   ├── InPossession       → SUCCESS if dist(robot, ball) ≤ POSSESSION_DIST
    │   └── DistributeSelector (Selector)
    │       ├── PassSequence (Sequence)
    │       │   ├── FindOpenTeammate   → highest-space own robot (excl. goalie)
    │       │   └── PassToTeammate     → IntentPass(target)
    │       ├── ShootIfClose           → IntentKick if dist ≤ SHOOT_DIST_THRESHOLD
    │       └── DribbleToGoal          → IntentDribble(enemy_goal_pos)
    └── RepositionToSpace      → IntentMove(least-crowded open field cell)

Key new nodes:
- ``IsClosestToBall`` — prevents multiple robots from chasing the same ball.
- ``InPossession`` — share logic with ``HasBallControl`` in attacker.py.
- ``FindOpenTeammate`` — score own robots by distance-to-nearest-opponent;
  return FAILURE if all are within ``MARKED_THRESHOLD`` of an opponent.
- ``RepositionToSpace`` — grid-score field cells by separation from all
  robots; pick highest-scoring cell within a sensible attacking area.

See ``docs/future.md §0.4`` for the full design specification.
"""
from __future__ import annotations

import py_trees

from TeamControl.bt.contracts.blackboard import RobotBlackboard
from TeamControl.bt.contracts.intent import IntentMove, IntentReceive
from TeamControl.bt.contracts.snapshot import Snapshot
from TeamControl.world.field_config import FIELD_LENGTH_MM, FIELD_WIDTH_MM

_HALF_LEN_M: float = FIELD_LENGTH_MM / 2.0 / 1000.0
_HALF_WID_M: float = FIELD_WIDTH_MM / 2.0 / 1000.0

# -----------------------------------------------------------------------
# Tuneable constants
# -----------------------------------------------------------------------

# Four support slots — authored for us_positive=False (opp goal at +x).
# SupporterTree negates x when us_positive=True.
# All positions are within field bounds for any standard field size.
_SUPPORT_SLOTS: dict[int, tuple[float, float]] = {
    2: ( _HALF_LEN_M * 0.25,  _HALF_WID_M * 0.55),  # right-forward wing
    3: ( _HALF_LEN_M * 0.25, -_HALF_WID_M * 0.55),  # left-forward wing
    4: (-_HALF_LEN_M * 0.25,  _HALF_WID_M * 0.55),  # right-back
    5: (-_HALF_LEN_M * 0.25, -_HALF_WID_M * 0.55),  # left-back
}
_DEFAULT_SUPPORT_POS: tuple[float, float] = (0.0, 0.0)

BLOCKING_POSITION: tuple[float, float] = (-_HALF_LEN_M * 0.5, 0.0)


# -----------------------------------------------------------------------
# Condition / action nodes
# -----------------------------------------------------------------------

class MoveToSpace(py_trees.behaviour.Behaviour):
    """Move the supporter to an open space position.

    Always returns SUCCESS in v1, writing IntentMove(open_space_pos) to
    the blackboard. In future versions, this may check whether the space
    is actually open before committing.

    SUCCESS → writes IntentMove(target_pos=OPEN_SPACE_POSITION).
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("MoveToSpace")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        pos = self._tree.support_slots.get(bb.robot_id, self._tree.support_slots.get(2, _DEFAULT_SUPPORT_POS))
        bb.current_intent = IntentMove(
            target_pos=pos,
            target_orientation=None,
        )
        return py_trees.common.Status.SUCCESS


class IsBallComing(py_trees.behaviour.Behaviour):
    """Check whether the ball is travelling toward this robot.

    STUBBED for v1: always returns FAILURE so that ReceiveBallSequence
    never fires. Replace this stub with real trajectory analysis when
    the ball-tracking module is available.

    # TODO: implement DoBallTrajectory to compute whether ball is coming.
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("IsBallComing")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        # TODO: implement DoBallTrajectory — check whether ball trajectory
        # intersects this robot's position within a time threshold.
        return py_trees.common.Status.FAILURE


class ReceiveBall(py_trees.behaviour.Behaviour):
    """Signal readiness to receive a pass by writing IntentReceive().

    Only reached when IsBallComing returns SUCCESS (never in v1).
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("ReceiveBall")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentReceive()
        return py_trees.common.Status.SUCCESS


class BlockOpponent(py_trees.behaviour.Behaviour):
    """Move to a blocking position to obstruct an opponent.

    Fallback if neither MoveToSpace nor ReceiveBallSequence succeed.
    In v1, MoveToSpace always succeeds so this node is never reached.

    SUCCESS → writes IntentMove(target_pos=BLOCKING_POSITION).
    """

    def __init__(self, tree_ref: SupporterTree) -> None:
        super().__init__("BlockOpponent")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        bb = self._tree._blackboard_ref[0]
        if bb is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = IntentMove(
            target_pos=self._tree.blocking_position,
            target_orientation=None,
        )
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# SupporterTree
# -----------------------------------------------------------------------

class SupporterTree:
    """Wrapper around the Supporter py_trees topology.

    Usage::

        tree = SupporterTree()
        tree.set_snapshot(snapshot)   # inject world state
        tree.tick(blackboard)          # run tree; writes Intent to blackboard
        intent = blackboard.current_intent
    """

    def __init__(self, us_positive: bool = True) -> None:
        self._snapshot: Snapshot | None = None
        self._blackboard_ref: list = [None]
        self.us_positive = us_positive
        # Negate x for each slot when us_positive=True (own half is +x,
        # so forward is -x direction).
        self.support_slots: dict[int, tuple[float, float]] = {
            rid: ((-pos[0], pos[1]) if us_positive else pos)
            for rid, pos in _SUPPORT_SLOTS.items()
        }
        self.blocking_position: tuple[float, float] = (
            (-BLOCKING_POSITION[0], BLOCKING_POSITION[1]) if us_positive
            else BLOCKING_POSITION
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

    def _build_tree(self) -> py_trees.composites.Selector:
        # ReceiveBallSequence — Sequence: IsBallComing (stub) → ReceiveBall.
        # IsBallComing always returns FAILURE in v1, so this sequence never
        # produces an intent in v1.
        receive_seq = py_trees.composites.Sequence(
            name="ReceiveBallSequence", memory=False
        )
        receive_seq.add_children([
            IsBallComing(self),
            ReceiveBall(self),
        ])

        # Root: SupportingSelectorNode
        # Tries MoveToSpace first (always succeeds in v1), then
        # ReceiveBallSequence (always fails in v1), then BlockOpponent.
        root = py_trees.composites.Selector(
            name="SupportingSelectorNode", memory=False
        )
        root.add_children([
            MoveToSpace(self),
            receive_seq,
            BlockOpponent(self),
        ])
        return root
