"""receive_pass skill — hold a chosen reception spot, face the ball, then trap it.

The counter-attack reception primitive: the robot waits at the target point
facing the incoming ball and, once the ball is close, steps onto its (predicted)
path with the dribbler on to collect it. Distinct from ``intercept_ball``, which
chases the ball's predicted point from anywhere — this one holds the spot the
caller picked and only pounces at the last moment.
"""
from __future__ import annotations

import math

from TeamControl.bt.contracts.intent import IntentDribble, IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import angle_to

# Ball within this range of the receiver → step onto it and trap (dribbler on).
RECEIVE_MEET_DIST_M: float = 1.0
# How far ahead of a moving ball to aim so we meet it rather than chase (seconds).
RECEIVE_LEAD_S: float = 0.2


def receive_pass(
    snap: Snapshot,
    robot: RobotState | None,
    target: tuple[float, float] | None,
) -> IntentMove | IntentDribble | None:
    """Hold ``target`` facing the ball; trap the ball once it is near."""
    if robot is None or target is None:
        return None
    bx, by = snap.ball_position
    face = angle_to(robot.position, (bx, by))
    dist_ball = math.hypot(bx - robot.position[0], by - robot.position[1])
    if dist_ball <= RECEIVE_MEET_DIST_M:
        vx, vy = snap.ball_velocity
        meet = (
            (bx + vx * RECEIVE_LEAD_S, by + vy * RECEIVE_LEAD_S)
            if math.hypot(vx, vy) > 0.1
            else (bx, by)
        )
        return IntentDribble(target_pos=meet)
    return IntentMove(target_pos=target, target_orientation=face)
