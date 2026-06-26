"""SkillBuilder — fluent factory API for creating BT nodes without boilerplate.

Instead of writing a 20-line py_trees.Behaviour subclass for each action, use
SkillBuilder factory methods that accept a callable describing the logic:

    # Old way (20+ lines):
    class ChaseBall(py_trees.behaviour.Behaviour):
        def __init__(self, tree_ref): ...
        def update(self):
            snap = self._tree._snapshot
            bb   = self._tree._blackboard_ref[0]
            ...
            bb.current_intent = IntentMove(...)
            bb.intent_source  = "ChaseBall"
            return py_trees.common.Status.SUCCESS

    # SkillBuilder way (1 line):
    ChaseBall = SkillBuilder.move_to_ball(tree_ref)

    # Custom logic (2 lines):
    MyAction = SkillBuilder.action(
        tree_ref, "MyAction",
        lambda snap, robot, bb: IntentMove(target_pos=my_fn(snap), target_orientation=None),
    )

All factory methods return a ``py_trees.behaviour.Behaviour`` node that plugs
directly into any Selector, Sequence, or composite node. The ``tree_ref``
argument must be the tree instance that owns ``_snapshot`` and
``_blackboard_ref``.

Callables receive ``(snapshot: Snapshot, robot: RobotState | None, bb: RobotBlackboard)``:
- ``snapshot``  — frozen world state for this tick.
- ``robot``     — this robot's RobotState from snapshot.own_robots, or None
                  if the robot is not currently visible.
- ``bb``        — the robot's blackboard (do NOT write intent here; the node
                  writes it automatically from the callable's return value).

Action callable must return an ``Intent`` or ``None``.
  - Returns ``Intent``  → intent is stored, node returns SUCCESS.
  - Returns ``None``    → node returns FAILURE (robot absent / condition unmet).

Condition callable must return a ``bool``.
  - ``True``  → SUCCESS.
  - ``False`` → FAILURE.
"""
from __future__ import annotations

import math
from typing import Callable

import py_trees

from TeamControl.bt.contracts.blackboard import RobotBlackboard
from TeamControl.world.field_config import FIELD_LENGTH_MM

_HALF_LEN_M: float = FIELD_LENGTH_MM / 2.0 / 1000.0
from TeamControl.bt.contracts.intent import (
    Intent,
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentOrient,
    IntentReceive,
)
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot

# Type aliases for readability.
_ActionFn = Callable[[Snapshot, "RobotState | None", RobotBlackboard], "Intent | None"]
_CondFn   = Callable[[Snapshot, "RobotState | None", RobotBlackboard], bool]


# ── Internal node implementations ────────────────────────────────────────

class _SkillActionNode(py_trees.behaviour.Behaviour):
    """Action node backed by a user-supplied intent-producing callable."""

    def __init__(self, tree_ref, name: str, intent_fn: _ActionFn) -> None:
        super().__init__(name)
        self._tree = tree_ref
        self._intent_fn = intent_fn

    def update(self) -> py_trees.common.Status:
        snap: Snapshot | None = self._tree._snapshot
        bb: RobotBlackboard | None = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        robot: RobotState | None = next(
            (r for r in snap.own_robots if r.robot_id == bb.robot_id), None
        )
        intent = self._intent_fn(snap, robot, bb)
        if intent is None:
            return py_trees.common.Status.FAILURE
        bb.current_intent = intent
        bb.intent_source = self.name
        return py_trees.common.Status.SUCCESS


class _SkillConditionNode(py_trees.behaviour.Behaviour):
    """Condition node backed by a user-supplied predicate callable."""

    def __init__(self, tree_ref, name: str, check_fn: _CondFn) -> None:
        super().__init__(name)
        self._tree = tree_ref
        self._check_fn = check_fn

    def update(self) -> py_trees.common.Status:
        snap: Snapshot | None = self._tree._snapshot
        bb: RobotBlackboard | None = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE
        robot: RobotState | None = next(
            (r for r in snap.own_robots if r.robot_id == bb.robot_id), None
        )
        ok = self._check_fn(snap, robot, bb)
        return (
            py_trees.common.Status.SUCCESS
            if ok
            else py_trees.common.Status.FAILURE
        )


# ── Public API ────────────────────────────────────────────────────────────

class SkillBuilder:
    """Factory for creating BT nodes with minimal boilerplate.

    Every method is a ``@staticmethod`` — no instance needed:
        node = SkillBuilder.move_to_ball(tree_ref)
    """

    # ── Generic factories ─────────────────────────────────────────────

    @staticmethod
    def action(
        tree_ref,
        name: str,
        intent_fn: _ActionFn,
    ) -> py_trees.behaviour.Behaviour:
        """Create a named action node driven by *intent_fn*.

        *intent_fn(snapshot, robot, blackboard)* must return an ``Intent``
        on success or ``None`` on failure.

        Example — move to a fixed position:
            node = SkillBuilder.action(
                tree_ref, "GoToCenter",
                lambda snap, robot, bb: IntentMove((0.0, 0.0), None),
            )
        """
        return _SkillActionNode(tree_ref, name, intent_fn)

    @staticmethod
    def condition(
        tree_ref,
        name: str,
        check_fn: _CondFn,
    ) -> py_trees.behaviour.Behaviour:
        """Create a named condition node driven by *check_fn*.

        *check_fn(snapshot, robot, blackboard)* must return ``True`` (SUCCESS)
        or ``False`` (FAILURE).

        Example — succeed only when the score is level:
            node = SkillBuilder.condition(
                tree_ref, "ScoreLevel",
                lambda snap, robot, bb: snap.referee_state.score[0] == snap.referee_state.score[1],
            )
        """
        return _SkillConditionNode(tree_ref, name, check_fn)

    # ── Pre-built action nodes ────────────────────────────────────────

    @staticmethod
    def move_to_ball(
        tree_ref,
        name: str = "MoveToBall",
        max_speed: float | None = None,
    ) -> py_trees.behaviour.Behaviour:
        """Move to the ball, orienting to face it on approach."""
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> Intent | None:
            if robot is None:
                return None
            dx = snap.ball_position[0] - robot.position[0]
            dy = snap.ball_position[1] - robot.position[1]
            return IntentMove(
                target_pos=snap.ball_position,
                target_orientation=math.atan2(dy, dx),
                max_speed=max_speed,
            )
        return _SkillActionNode(tree_ref, name, _fn)

    @staticmethod
    def move_to_pos(
        tree_ref,
        pos_fn: Callable[[Snapshot, "RobotState | None"], tuple[float, float]],
        name: str = "MoveToPos",
        orientation_fn: "Callable[[Snapshot, RobotState | None], float | None] | None" = None,
        face_target: bool = False,
        max_speed: float | None = None,
    ) -> py_trees.behaviour.Behaviour:
        """Move to a dynamically computed position.

        Args:
            pos_fn:         ``(snapshot, robot) → (x, y)`` target in metres.
            orientation_fn: ``(snapshot, robot) → radians``, or ``None``.
            face_target:    If ``True`` and no *orientation_fn* given, orient
                            toward the computed target automatically.
            max_speed:      Speed cap forwarded to the motion layer.

        Example — stand 1 m behind the ball toward our own goal:
            node = SkillBuilder.move_to_pos(
                tree_ref, "BehindBall",
                lambda s, r: (s.ball_position[0] - 1.0, s.ball_position[1]),
                face_target=True,
            )
        """
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> Intent | None:
            pos = pos_fn(snap, robot)
            if orientation_fn is not None:
                ori = orientation_fn(snap, robot)
            elif face_target and robot is not None:
                ori = math.atan2(pos[1] - robot.position[1], pos[0] - robot.position[0])
            else:
                ori = None
            return IntentMove(target_pos=pos, target_orientation=ori, max_speed=max_speed)
        return _SkillActionNode(tree_ref, name, _fn)

    @staticmethod
    def kick_at(
        tree_ref,
        target_fn: "Callable[[Snapshot], tuple[float, float]] | None" = None,
        name: str = "KickAt",
    ) -> py_trees.behaviour.Behaviour:
        """Kick the ball toward a target.

        Args:
            target_fn: ``(snapshot) → (x, y)`` kick target. Defaults to the
                       opponent goal centre ``(4.5, 0.0)``.

        Example — kick at a specific ally robot:
            node = SkillBuilder.kick_at(
                tree_ref, "PassToWing",
                lambda s: next(r.position for r in s.own_robots if r.robot_id == 2),
            )
        """
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> Intent | None:
            target = target_fn(snap) if target_fn is not None else (_HALF_LEN_M, 0.0)
            return IntentKick(target_pos=target)
        return _SkillActionNode(tree_ref, name, _fn)

    @staticmethod
    def dribble_to(
        tree_ref,
        pos_fn: Callable[[Snapshot, "RobotState | None"], tuple[float, float]],
        name: str = "DribbleTo",
    ) -> py_trees.behaviour.Behaviour:
        """Dribble the ball to a dynamically computed position."""
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> Intent | None:
            return IntentDribble(target_pos=pos_fn(snap, robot))
        return _SkillActionNode(tree_ref, name, _fn)

    @staticmethod
    def face_ball(
        tree_ref,
        name: str = "FaceBall",
    ) -> py_trees.behaviour.Behaviour:
        """Rotate in place to face the ball."""
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> Intent | None:
            if robot is None:
                return None
            angle = math.atan2(
                snap.ball_position[1] - robot.position[1],
                snap.ball_position[0] - robot.position[0],
            )
            return IntentOrient(target_orientation=angle)
        return _SkillActionNode(tree_ref, name, _fn)

    @staticmethod
    def orient_to(
        tree_ref,
        angle_fn: Callable[[Snapshot, "RobotState | None"], float],
        name: str = "OrientTo",
    ) -> py_trees.behaviour.Behaviour:
        """Rotate in place to a dynamically computed angle."""
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> Intent | None:
            return IntentOrient(target_orientation=angle_fn(snap, robot))
        return _SkillActionNode(tree_ref, name, _fn)

    @staticmethod
    def receive_ball(
        tree_ref,
        name: str = "ReceiveBall",
    ) -> py_trees.behaviour.Behaviour:
        """Signal readiness to receive a pass (hold position, face ball)."""
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> Intent | None:
            return IntentReceive()
        return _SkillActionNode(tree_ref, name, _fn)

    # ── Pre-built condition nodes ─────────────────────────────────────

    @staticmethod
    def ball_within(
        tree_ref,
        dist_m: float,
        name: str | None = None,
    ) -> py_trees.behaviour.Behaviour:
        """Succeed when this robot is within *dist_m* metres of the ball."""
        label = name or f"BallWithin({dist_m:.3f}m)"

        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> bool:
            if robot is None:
                return False
            return math.hypot(
                snap.ball_position[0] - robot.position[0],
                snap.ball_position[1] - robot.position[1],
            ) <= dist_m

        return _SkillConditionNode(tree_ref, label, _fn)

    @staticmethod
    def facing_ball(
        tree_ref,
        tol_rad: float = 0.3,
        name: str = "FacingBall",
    ) -> py_trees.behaviour.Behaviour:
        """Succeed when heading is within *tol_rad* radians of the ball direction."""
        def _fn(snap: Snapshot, robot: RobotState | None, bb: RobotBlackboard) -> bool:
            if robot is None:
                return False
            angle = math.atan2(
                snap.ball_position[1] - robot.position[1],
                snap.ball_position[0] - robot.position[0],
            )
            err = (angle - robot.orientation + math.pi) % (2 * math.pi) - math.pi
            return abs(err) <= tol_rad

        return _SkillConditionNode(tree_ref, name, _fn)

    @staticmethod
    def custom_condition(
        tree_ref,
        name: str,
        check_fn: _CondFn,
    ) -> py_trees.behaviour.Behaviour:
        """Alias for ``condition()`` — prefer over raw condition for clarity."""
        return _SkillConditionNode(tree_ref, name, check_fn)
