"""Tests for the MarkerTree (man-marking role, GegenPressing strategy)."""
from __future__ import annotations

import math
import py_trees

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.trees.marker import MarkerTree

MARKER_ID = 2


def _snapshot(
    *,
    ball: tuple[float, float],
    own: list[RobotState],
    enemies: list[RobotState] | None = None,
) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=[] if enemies is None else enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _tick(snapshot: Snapshot, *, mark_target_id, us_positive: bool = False):
    tree = MarkerTree(us_positive=us_positive)
    bb = RobotBlackboard(
        robot_id=MARKER_ID,
        current_role=RoleType.MARKER,
        mark_target_id=mark_target_id,
    )
    tree.set_snapshot(snapshot)
    tree.tick(bb)
    return bb


def test_marker_topology() -> None:
    tree = MarkerTree(us_positive=False)
    assert isinstance(tree.root, py_trees.composites.Sequence)
    assert tree.root.name == "MarkingSequenceNode"
    assert [c.name for c in tree.root.children] == ["LookAtBall", "MarkFallback"]
    assert [c.name for c in tree.root.children[1].children] == [
        "ShadowOpponent",
        "ZoneCover",
    ]


def test_shadows_assigned_opponent_ball_side() -> None:
    marker = RobotState(robot_id=MARKER_ID, position=(-2.0, 0.0), orientation=0.0)
    opp = RobotState(robot_id=11, position=(-1.0, 0.0), orientation=0.0)
    snap = _snapshot(ball=(1.0, 0.0), own=[marker], enemies=[opp])

    bb = _tick(snap, mark_target_id=11)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "ShadowOpponent"
    # Standoff 0.35 from opponent toward the ball (ball-side): (-1 + 0.35, 0).
    tx, ty = bb.current_intent.target_pos
    assert math.isclose(tx, -0.65, abs_tol=1e-6)
    assert math.isclose(ty, 0.0, abs_tol=1e-6)
    # Far from the man (1 m) → no speed cap.
    assert bb.current_intent.max_speed is None
    # Faces the ball.
    assert math.isclose(bb.current_intent.target_orientation, 0.0, abs_tol=1e-6)


def test_caps_speed_when_close_to_marked_opponent() -> None:
    # Marker 0.3 m from its man — inside the crash radius → approach speed cap.
    marker = RobotState(robot_id=MARKER_ID, position=(-1.3, 0.0), orientation=0.0)
    opp = RobotState(robot_id=11, position=(-1.0, 0.0), orientation=0.0)
    snap = _snapshot(ball=(1.0, 0.0), own=[marker], enemies=[opp])

    bb = _tick(snap, mark_target_id=11)

    assert bb.intent_source == "ShadowOpponent"
    assert bb.current_intent.max_speed == 0.9


def test_zone_covers_when_no_man_assigned() -> None:
    marker = RobotState(robot_id=MARKER_ID, position=(-2.0, 0.0), orientation=0.0)
    snap = _snapshot(ball=(0.0, 0.0), own=[marker], enemies=[])

    bb = _tick(snap, mark_target_id=None)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "ZoneCover"
    # us_positive=False → own goal at x=-4.5, zone_depth 2 → zone_x = -2.5.
    tx, _ = bb.current_intent.target_pos
    assert math.isclose(tx, -2.5, abs_tol=1e-6)


def test_zone_covers_when_assigned_opponent_absent() -> None:
    marker = RobotState(robot_id=MARKER_ID, position=(-2.0, 0.0), orientation=0.0)
    # Assigned to opponent 11 but it is not in the snapshot.
    snap = _snapshot(ball=(0.0, 0.0), own=[marker], enemies=[])

    bb = _tick(snap, mark_target_id=11)

    assert bb.intent_source == "ZoneCover"
