from __future__ import annotations

import math

from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot


CHASE_BALL_SPEED_GAIN: float = 3.0


def chase_ball(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Gegenpress-style loose-ball chase: drive directly at the ball, facing it."""
    if robot is None:
        return None
    dx = snap.ball_position[0] - robot.position[0]
    dy = snap.ball_position[1] - robot.position[1]
    return IntentMove(
        target_pos=snap.ball_position,
        target_orientation=math.atan2(dy, dx),
        speed_gain=CHASE_BALL_SPEED_GAIN,
    )
