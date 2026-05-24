"""
Ball trajectory prediction with physics-based friction model and wall-bounce support.

Replaces naive linear regression with a proper physics simulation that accounts for:
- Ball friction/deceleration
- Wall bounces off field boundaries
- Multiple prediction modes (goal intercept, general trajectory)
"""

import math

GOAL_WIDTH_MM = 1000
FRICTION = 0.40          # friction deceleration factor per second
WALL_RESTITUTION = 0.7   # energy retained after wall bounce


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def predict_ball_path(ball_x, ball_y, vel_x, vel_y, dt, field_width):
    """
    Predict ball position after dt seconds with friction and wall bounces.

    Returns (x, y) predicted position.
    """
    bx, by = ball_x, ball_y
    vx, vy = vel_x, vel_y
    half_w = field_width / 2
    t = 0.0
    step = 0.01

    while t < dt:
        s = min(step, dt - t)
        bx += vx * s
        by += vy * s

        # Wall bounces
        if by > half_w:
            by = 2 * half_w - by
            vy = -vy * WALL_RESTITUTION
        elif by < -half_w:
            by = -2 * half_w - by
            vy = -vy * WALL_RESTITUTION

        # Friction
        f = max(1.0 - FRICTION * s, 0.0)
        vx *= f
        vy *= f

        t += s

        # Ball stopped
        if math.hypot(vx, vy) < 30:
            break

    return bx, by


def predict_trajectory(history: list, num_samples: int, isPostive: bool, field_size: tuple):
    """
    Predict ball intercept at our goal line using physics-based trajectory.

    Args:
        history: list of (x, y) ball positions (world frame, mm)
        num_samples: number of points to use
        isPostive: True if our goal is at +x
        field_size: (field_x, field_y) in mm

    Returns:
        (goalie_pos, heading_to_goal) or (None, False):
        - goalie_pos: (goal_line_x, y) where goalie should stand
        - heading_to_goal: True if ball heading toward our goal and on target
    """
    field_x, field_y = field_size
    half_goal = GOAL_WIDTH_MM / 2

    if isPostive:
        goal_line = field_x / 2 - 200
    else:
        goal_line = -field_x / 2 + 200

    if not history:
        return None, False

    n = min(len(history), num_samples)
    if n < 2:
        y_clamp = _clamp(float(history[-1][1]), -half_goal, half_goal)
        return (goal_line, y_clamp), False

    # Estimate velocity from history endpoints
    dx = float(history[-1][0]) - float(history[0][0])
    dy = float(history[-1][1]) - float(history[0][1])

    # Check if heading toward our goal
    if isPostive:
        heading_to_goal = dx > 50
    else:
        heading_to_goal = dx < -50

    if not heading_to_goal:
        # Ball not heading our way — position at ball's y, clamped
        y_clamp = _clamp(float(history[-1][1]), -half_goal, half_goal)
        return (goal_line, y_clamp), False

    # Estimate velocity (mm per sample interval)
    # Use average of recent velocity samples for stability
    vel_samples = []
    for i in range(max(0, n - 4), n - 1):
        vx = float(history[i + 1][0]) - float(history[i][0])
        vy = float(history[i + 1][1]) - float(history[i][1])
        vel_samples.append((vx, vy))

    if not vel_samples:
        y_clamp = _clamp(float(history[-1][1]), -half_goal, half_goal)
        return (goal_line, y_clamp), False

    avg_vx = sum(v[0] for v in vel_samples) / len(vel_samples)
    avg_vy = sum(v[1] for v in vel_samples) / len(vel_samples)

    # Scale to per-second (assume ~25Hz vision = 0.04s between samples)
    frame_dt = 0.04
    vel_x = avg_vx / frame_dt
    vel_y = avg_vy / frame_dt

    ball_x = float(history[-1][0])
    ball_y = float(history[-1][1])

    # Simulate ball forward with friction and bounces
    bx, by = ball_x, ball_y
    vx, vy = vel_x, vel_y
    half_field_y = field_y / 2
    sign = 1 if isPostive else -1
    t = 0.0
    step = 0.01
    max_t = 3.0

    crossed = False
    cross_y = 0.0

    while t < max_t:
        bx += vx * step
        by += vy * step

        # Wall bounces
        if by > half_field_y:
            by = 2 * half_field_y - by
            vy = -vy * WALL_RESTITUTION
        elif by < -half_field_y:
            by = -2 * half_field_y - by
            vy = -vy * WALL_RESTITUTION

        # Friction
        f = max(1.0 - FRICTION * step, 0.0)
        vx *= f
        vy *= f
        t += step

        if math.hypot(vx, vy) < 30:
            break

        # Check goal line crossing
        if (sign > 0 and bx >= goal_line) or (sign < 0 and bx <= goal_line):
            crossed = True
            cross_y = by
            break

    if crossed:
        y_clamp = _clamp(cross_y, -half_goal, half_goal)
        within_goal = abs(cross_y) <= half_goal
        return (goal_line, y_clamp), within_goal
    else:
        # Ball won't reach goal — predict where it stops
        y_clamp = _clamp(by, -half_goal, half_goal)
        return (goal_line, y_clamp), False
