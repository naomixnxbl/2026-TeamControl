from __future__ import annotations
import math
from TeamControl.bt.contracts.intent import IntentMove, IntentOrient
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import (
    BALL_APPROACH_OFFSET_M, FACE_BALL_TOLERANCE_RAD,
    angle_to, pd_speed, approach_cache,
)


def move_to_ball(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | IntentOrient | None:
    """Face the ball then drive toward it, stopping 15 cm short."""
    if robot is None:
        return None
    ball = snap.ball_position
    dx   = ball[0] - robot.position[0]
    dy   = ball[1] - robot.position[1]
    dist = math.hypot(dx, dy)
    angle_to_ball = angle_to(robot.position, ball)

    if dist <= BALL_APPROACH_OFFSET_M:
        approach_cache.pop(robot.robot_id, None)
        return IntentOrient(target_orientation=angle_to_ball)

    angle_err = abs(math.remainder(robot.orientation - angle_to_ball, 2 * math.pi))
    if angle_err > FACE_BALL_TOLERANCE_RAD:
        approach_cache.pop(robot.robot_id, None)
        return IntentOrient(target_orientation=angle_to_ball)

    approach_pos = (robot.position[0] + dx * (dist - BALL_APPROACH_OFFSET_M) / dist,
                    robot.position[1] + dy * (dist - BALL_APPROACH_OFFSET_M) / dist)
    speed = pd_speed(robot.robot_id, robot.position, dx, dy, dist - BALL_APPROACH_OFFSET_M)
    return IntentMove(target_pos=approach_pos, target_orientation=angle_to_ball, max_speed=speed)
