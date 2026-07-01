"""Left↔right dribble nudge: weave the body to keep the ball on the dribbler."""
from __future__ import annotations

from TeamControl.bt.adapter import MotionExecutor
from TeamControl.bt.contracts.intent import IntentDribble, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)


def _snap(robot_pos=(0.0, 0.0), ball_pos=(0.05, 0.0)) -> Snapshot:
    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=(0.0, 0.0),
        own_robots=[RobotState(robot_id=1, position=robot_pos, orientation=0.0)],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_dribble_adds_lateral_nudge_while_carrying() -> None:
    ex = MotionExecutor()
    snap = _snap(robot_pos=(0.0, 0.0), ball_pos=(0.05, 0.0))  # ball on the dribbler
    seen_lateral = set()
    for _ in range(12):  # a few ticks spans the oscillation
        cmd = ex.resolve_command(IntentDribble(target_pos=(2.0, 0.0)), 1, snap, True)
        seen_lateral.add(round(cmd.vy, 3))
    # The lateral (vy) component varies across ticks — it's weaving, not constant.
    assert len(seen_lateral) > 1
    assert any(abs(v) > 1e-6 for v in seen_lateral)


def test_no_nudge_when_ball_is_far() -> None:
    ex = MotionExecutor()
    snap = _snap(robot_pos=(0.0, 0.0), ball_pos=(2.0, 0.0))  # ball far → approaching
    # Phase stays reset; lateral component comes only from the (straight) approach.
    cmd = ex.resolve_command(IntentDribble(target_pos=(2.0, 0.0)), 1, snap, True)
    assert ex._get_movement(1).dribble_phase == 0
    assert abs(cmd.vy) < 1e-6  # straight at the ball, no weave


def test_nudge_only_applies_to_dribble() -> None:
    ex = MotionExecutor()
    snap = _snap(robot_pos=(0.0, 0.0), ball_pos=(0.05, 0.0))
    for _ in range(5):
        ex.resolve_command(IntentMove(target_pos=(2.0, 0.0), target_orientation=0.0), 1, snap, True)
    # A plain move never advances the dribble phase.
    assert ex._get_movement(1).dribble_phase == 0
