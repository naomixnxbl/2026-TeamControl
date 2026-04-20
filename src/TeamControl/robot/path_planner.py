"""
High-performance path planning and velocity computation for robot movement.

Key features:
- Full-speed operation (no artificial throttle)
- Smooth proportional angular velocity (no step-function jerkiness)
- Trapezoidal speed profiles for fast yet precise arrivals
- Curved-path support for simultaneous move+rotate
- Arc-based approach for getting behind the ball without looping
"""

import math
from typing import Tuple, Optional

from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.constants import MAX_W, PP_GAIN, PP_MIN_IMPULSE, HALF_WID


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def move_toward(
    robot_pos: Tuple[float, float, float],
    target_world: Tuple[float, float],
    max_speed: float,
    stop_radius: float = 40.0,
    ramp_dist: float = 350.0,
    min_speed: float = 0.25,
) -> Tuple[float, float]:
    """
    Compute (vx, vy) in robot frame to move toward a world-frame target.
    Trapezoidal profile: full speed when far, smooth ramp to zero near target.
    """
    rel = world2robot(robot_pos, target_world)
    dist = math.hypot(rel[0], rel[1])
    if dist < stop_radius:
        return 0.0, 0.0
    if dist < ramp_dist:
        t = (dist - stop_radius) / max(ramp_dist - stop_radius, 1.0)
        # Quadratic ramp for smoother deceleration
        speed = max(max_speed * t * t, min_speed)
    else:
        speed = max_speed
    if dist < 1e-6:
        return 0.0, 0.0
    vx = (rel[0] / dist) * speed
    vy = (rel[1] / dist) * speed
    return vx, vy


def move_toward_relative(
    rel_target: Tuple[float, float],
    max_speed: float,
    stop_radius: float = 40.0,
    ramp_dist: float = 350.0,
    min_speed: float = 0.25,
) -> Tuple[float, float]:
    """
    Same as move_toward but target is already in robot frame.
    """
    dist = math.hypot(rel_target[0], rel_target[1])
    if dist < stop_radius:
        return 0.0, 0.0
    if dist < ramp_dist:
        t = (dist - stop_radius) / max(ramp_dist - stop_radius, 1.0)
        speed = max(max_speed * t * t, min_speed)
    else:
        speed = max_speed
    if dist < 1e-6:
        return 0.0, 0.0
    vx = (rel_target[0] / dist) * speed
    vy = (rel_target[1] / dist) * speed
    return vx, vy


def turn_toward(
    rel_target: Tuple[float, float],
    epsilon: float = 0.05,
    speed_scale: float = 1.0,
    max_w: float = MAX_W,
    gain: float = PP_GAIN,
) -> float:
    """
    Smooth proportional angular velocity to face rel_target in robot frame.
    Uses proportional control with saturation instead of step function.
    """
    if rel_target is None or (abs(rel_target[0]) < 1e-9 and abs(rel_target[1]) < 1e-9):
        return 0.0
    angle = math.atan2(rel_target[1], rel_target[0])
    if abs(angle) < epsilon:
        return 0.0
    # Proportional control with minimum impulse for small angles
    w = speed_scale * gain * angle
    # Minimum angular velocity to overcome friction when close to target
    if 0 < abs(w) < PP_MIN_IMPULSE:
        w = math.copysign(PP_MIN_IMPULSE, w)
    return clamp(w, -max_w, max_w)


def move_and_face(
    robot_pos: Tuple[float, float, float],
    move_target_world: Tuple[float, float],
    face_target_world: Tuple[float, float],
    max_linear_speed: float,
    max_angular_speed: float = 9.0,
    stop_radius: float = 40.0,
    ramp_dist: float = 350.0,
    turn_epsilon: float = 0.05,
) -> Tuple[float, float, float]:
    """
    (vx, vy, w) to move toward move_target while turning to face face_target.
    """
    rel_move = world2robot(robot_pos, move_target_world)
    rel_face = world2robot(robot_pos, face_target_world)
    vx, vy = move_toward_relative(
        rel_move, max_linear_speed,
        stop_radius=stop_radius, ramp_dist=ramp_dist,
    )
    w = turn_toward(rel_face, epsilon=turn_epsilon, max_w=max_angular_speed)
    return vx, vy, w


# ═══════════════════════════════════════════════════════════════════
#  ARC-BASED APPROACH — canonical implementation lives in ball_nav.py
# ═══════════════════════════════════════════════════════════════════
from TeamControl.robot.ball_nav import compute_arc_nav  # noqa: F401 — re-export


def move_with_ramp(
    rel: Tuple[float, float],
    speed: float,
    ramp_dist: float = 350.0,
    stop_dist: float = 25.0,
    min_speed: float = 0.10,
) -> Tuple[float, float]:
    """
    Move toward a robot-local point with linear deceleration near target.
    Better than raw unit-vector * speed because it prevents overshoot.
    """
    d = math.hypot(rel[0], rel[1])
    if d < stop_dist:
        return 0.0, 0.0
    if d < ramp_dist:
        t = (d - stop_dist) / max(ramp_dist - stop_dist, 1.0)
        speed = max(speed * t, min_speed)
    return (rel[0] / d) * speed, (rel[1] / d) * speed
