## applies different movement strategies to move the robot to a target position
# see `docs/motion-strategy.md` for more information
import time
from enum import Enum

from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.motion.controller import RobotMotionController

SLOW = 0.8  # s
NORMAL = 0.5  # s
FAST = 0.2  # s


MODE = NORMAL


def get_deadline(m=MODE) -> float:
    return time.monotonic() + m


def option_a_movement(
    motion: RobotMotionController,
    current_pos,
    target_xy,
    target_theta,
    is_yellow,
):

    robot_id = motion.robot_id

    if not motion.is_facing_dir(current_pos[2], target_theta):
        w = motion.rotational_motion(current_pos[2], target_theta, get_deadline())
        return RobotCommand(robot_id, 0, 0, w, 0, 0, isYellow=is_yellow)

    vx, vy = motion.translational_motion(current_pos, target_xy, get_deadline())
    return RobotCommand(robot_id, vx, vy, 0, 0, 0, isYellow=is_yellow)


def option_c_movement(
    motion: RobotMotionController,
    current_pos,
    target_xy,
    target_theta,
    is_yellow,
):
    vx, vy, w = motion.general_motion(
        current_pos, target_xy, target_theta, get_deadline()
    )
    return RobotCommand(motion.robot_id, vx, vy, w, 0, 0, isYellow=is_yellow)
