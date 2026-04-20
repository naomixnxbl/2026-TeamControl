from enum import Enum, auto
import math

from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.constants import _load_tuning

LINEAR = {
    "stop": [70, 0.01],
    "slow": [100, 0.5],
    "normal": [150, 1],  # avg. threshold and avg. speed
    "fast": [300, 2],
}

def _angular_from_tuning():
    t = _load_tuning()
    return {
        "stop": [0.015, 0.01],
        "slow": [0.05, t["angular_slow_speed"]],
        "normal": [0.1, t["angular_normal_speed"]],
        "fast": [1.0, t["angular_fast_speed"]],
    }

ANGULAR = _angular_from_tuning()


class Mode(Enum):
    Percision = auto()
    Normal = auto()
    Fast = auto()


def angle_between(robot_pos, target_pos):
    direction_2d = world2robot(robot_pos, target_pos)
    angle = math.atan2(direction_2d[1], direction_2d[0])
    return angle


def clamp(low, high, value):
    return max(low, min(high, value))


def select_linear_speed(relative_target, mode: Mode):
    """Select a linear speed base on distance and Mode
    Percision : all settings applied
    Normal : stop at normal threshold, otherwise go normal speed
    Fast : overshoot with the largest speed, no threshold applied
    """
    speed = 0
    match mode:
        case Mode.Percision:
            if relative_target <= LINEAR["stop"][0]:
                speed = LINEAR["stop"][1]
            elif relative_target <= LINEAR["slow"][0]:
                speed = LINEAR["slow"][1]
            else:
                speed = LINEAR["fast"][1]

        case Mode.Normal:
            if relative_target < LINEAR["normal"][0]:
                speed = 0.0
            else:
                speed = LINEAR["normal"][1]

        case Mode.Fast:
            speed = LINEAR["fast"][1]
    return speed


def go_to_target(robot_pos, target_pos, mode=Mode.Percision) -> tuple[float, float]:
    relative_target_arr = world2robot(robot_pos, target_pos)
    relative_distance = math.hypot(relative_target_arr[0], relative_target_arr[1])
    speed = select_linear_speed(relative_distance, mode)
    return calculate_linear_velocity(relative_target_arr, speed)


def calculate_linear_velocity(
    relative_target,
    speed,
    kd_gain=1.0,
    kp_gain=1.0,
) -> tuple[float, float]:
    vx = clamp(-speed, +speed, relative_target[0] * kp_gain)
    vy = clamp(-speed, +speed, relative_target[1] * kp_gain)
    return vx, vy


def select_angular_speed(relative_angle, mode: Mode):
    """Select Angular Speed base on Mode
    Percision : all settings applied
    Normal : Stop at Normal Threshold, otherwise turn with Normal Speed
    Fast : turn with Fast_Speed, no threshold applied
    """
    speed = 0.0
    relative_angle = abs(relative_angle)
    match mode:
        case Mode.Percision:
            if relative_angle <= ANGULAR["stop"][0]:
                speed = ANGULAR["stop"][1]
            elif relative_angle <= ANGULAR["slow"][0]:
                speed = ANGULAR["slow"][1]
            else:
                speed = ANGULAR["fast"][1]

        case Mode.Normal:
            if relative_angle < ANGULAR["normal"][0]:
                speed = 0.0
            else:
                speed = ANGULAR["normal"][1]

        case Mode.Fast:
            speed = ANGULAR["fast"][1]
    return speed


def turn_to_target(robot_pos, target_pos, mode: Mode) -> float:
    angle = angle_between(robot_pos, target_pos)
    speed = select_angular_speed(angle, mode)
    return calculate_angular_velocity(relative_angle=angle, speed=speed)


def calculate_angular_velocity(relative_angle, speed, kp_gain=1.0) -> float:
    w = clamp(-speed, speed, kp_gain * relative_angle)
    # print(f"Angular velocity: {'LEFT' if w > 0 else 'RIGHT'}")
    return w


if __name__ == "__main__":
    robot_pos = (0, 0, 0.1)
    target_pos = (10, 1)
    print(go_to_target(robot_pos, target_pos, Mode.Normal))
    print(turn_to_target(robot_pos, target_pos, Mode.Normal))
