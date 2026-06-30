"""Small skill fixes: receiver faces the ball, move_to speed gain."""
from __future__ import annotations

import math

from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.skills.move_to import move_to
from TeamControl.bt.skills.receive_ball import receive_ball


def _snap(ball, robot_pos, orientation=0.0):
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=[RobotState(robot_id=1, position=robot_pos, orientation=orientation)],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_receive_ball_faces_the_ball() -> None:
    # Ball is to the north-east of the receiver; it should orient toward it,
    # not stay at heading 0.0 (the old hard-coded value).
    snap = _snap(ball=(1.0, 1.0), robot_pos=(0.0, 0.0), orientation=0.0)
    target = receive_ball(snap, robot_id=1)
    assert target.target_velocity == (0.0, 0.0)
    assert math.isclose(target.target_orientation, math.atan2(1.0, 1.0), abs_tol=1e-6)


def test_move_to_gain_reaches_cap_sooner() -> None:
    # A 0.5 m correction: gain 1.0 crawls at 0.5 m/s; gain 4.0 saturates the cap.
    snap = _snap(ball=(9.0, 9.0), robot_pos=(0.0, 0.0))
    slow = move_to(snap, 1, (0.5, 0.0), 0.0, max_speed=2.0, gain=1.0)
    fast = move_to(snap, 1, (0.5, 0.0), 0.0, max_speed=2.0, gain=4.0)
    slow_speed = math.hypot(*slow.target_velocity)
    fast_speed = math.hypot(*fast.target_velocity)
    assert math.isclose(slow_speed, 0.5, abs_tol=1e-6)
    assert math.isclose(fast_speed, 2.0, abs_tol=1e-6)
    assert fast_speed > slow_speed
