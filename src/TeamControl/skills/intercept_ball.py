from __future__ import annotations
import math
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import INTERCEPT_LOOKAHEAD_S, angle_to


def intercept_ball(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Move to where the ball will be in ~0.8 s (velocity prediction), not its current position."""
    if robot is None:
        return None
    bx, by = snap.ball_position
    vx, vy = snap.ball_velocity
    pred = (bx + vx * INTERCEPT_LOOKAHEAD_S, by + vy * INTERCEPT_LOOKAHEAD_S) \
           if math.hypot(vx, vy) > 0.1 else (bx, by)
    return IntentMove(target_pos=pred, target_orientation=angle_to(robot.position, (bx, by)))
