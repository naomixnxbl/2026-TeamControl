"""
Shared ball physics and pathfinding utilities.

Every robot module (goalie, striker, navigator, team) imports from here
instead of defining its own copies.  One source of truth.

Ball physics:
  - predict_ball       — position after dt seconds with friction
  - ball_velocity      — velocity from timestamped history
  - update_ball_history — append to history buffer

Movement:
  - clamp              — value clamping
  - move_toward        — deceleration-ramp movement toward a local target
  - wall_brake         — slow down near field boundaries
  - rotation_compensate — pre-rotate velocity during turning

Pathfinding:
  - compute_arc_nav    — arc approach to get behind the ball
"""

import json
import math
import os
from typing import Tuple, Optional, List

from TeamControl.robot.constants import (
    HALF_LEN, HALF_WID,
    FRICTION,
    BALL_HISTORY_SIZE,
    LOOP_RATE,
)


# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION — loaded from calibration.json, applied in move_toward
# ═══════════════════════════════════════════════════════════════════

_CAL_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "calibration.json"))

_cal = {"speed_scale": 1.0}


def _reload_calibration():
    """Reload calibration values from disk.  Called at import time
    and when the calibration UI applies new values."""
    global _cal
    try:
        with open(_CAL_PATH, "r") as f:
            data = json.load(f)
        _cal["speed_scale"] = float(data.get("speed_scale", 1.0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        _cal["speed_scale"] = 1.0


_reload_calibration()


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


# ═══════════════════════════════════════════════════════════════════
#  BALL PHYSICS
# ═══════════════════════════════════════════════════════════════════

def predict_ball(pos, vel, dt):
    """Predict ball position after *dt* seconds with linear friction.

    Args:
        pos: (x, y) current ball position in mm
        vel: (vx, vy) current ball velocity in mm/s
        dt:  seconds to simulate forward

    Returns:
        (x, y) predicted position, clamped to field.
    """
    bx, by = pos
    vx, vy = vel
    t, step = 0.0, 0.02
    while t < dt:
        s = min(step, dt - t)
        bx += vx * s
        by += vy * s
        f = max(1.0 - FRICTION * s, 0.0)
        vx *= f
        vy *= f
        t += s
        if math.hypot(vx, vy) < 30:
            break
    return (clamp(bx, -HALF_LEN, HALF_LEN),
            clamp(by, -HALF_WID, HALF_WID))


def ball_velocity(history):
    """Estimate ball velocity from timestamped history.

    Args:
        history: list of (timestamp, x, y) tuples

    Returns:
        (vx, vy, speed) in mm/s.
    """
    if len(history) < 2:
        return 0.0, 0.0, 0.0
    dt = history[-1][0] - history[0][0]
    if dt < 0.02:
        return 0.0, 0.0, 0.0
    vx = (history[-1][1] - history[0][1]) / dt
    vy = (history[-1][2] - history[0][2]) / dt
    return vx, vy, math.hypot(vx, vy)


def update_ball_history(history, now, ball, last_ball_xy,
                        max_size=BALL_HISTORY_SIZE):
    """Append ball position to history if it moved.

    Returns the new last_ball_xy value.
    """
    if last_ball_xy is None or ball != last_ball_xy:
        history.append((now, ball[0], ball[1]))
        if len(history) > max_size:
            history.pop(0)
    return ball


# ═══════════════════════════════════════════════════════════════════
#  MOVEMENT
# ═══════════════════════════════════════════════════════════════════

# Wall-braking defaults (shared by striker and navigator)
WALL_BRAKE_DIST = 400       # start braking this far from wall (mm)
WALL_BRAKE_MIN  = 0.10      # minimum speed factor near walls


def move_toward(rel, speed, ramp_dist=350.0, stop_dist=10.0, min_speed=0.06):
    """Move toward a robot-local point with linear deceleration ramp.

    Speed is scaled by the calibration factor from calibration.json
    so commanded speed matches actual robot speed.

    Args:
        rel:       (x, y) target in robot frame
        speed:     cruise speed (m/s fraction)
        ramp_dist: start decelerating at this distance (mm)
        stop_dist: stop moving below this distance (mm)
        min_speed: floor speed during deceleration

    Returns:
        (vx, vy) velocity in robot frame.
    """
    d = math.hypot(rel[0], rel[1])
    if d < stop_dist:
        return 0.0, 0.0
    # Apply calibration: if robot runs slow, scale > 1 boosts commanded speed
    cal_scale = _cal.get("speed_scale", 1.0)
    if cal_scale > 0.01:
        speed = speed / cal_scale
    if d < ramp_dist:
        t = (d - stop_dist) / max(ramp_dist - stop_dist, 1.0)
        speed = max(speed * t, min_speed)
    return (rel[0] / d) * speed, (rel[1] / d) * speed


def wall_brake(rx, ry, vx, vy,
               brake_dist=WALL_BRAKE_DIST, min_factor=WALL_BRAKE_MIN):
    """Slow velocity near field boundaries.

    Args:
        rx, ry:     robot world position
        vx, vy:     current velocity
        brake_dist: distance from wall to start braking
        min_factor: minimum speed multiplier

    Returns:
        (vx, vy) braked velocity.
    """
    dist_to_wall = min(HALF_LEN - abs(rx), HALF_WID - abs(ry))
    if dist_to_wall < brake_dist:
        factor = max(dist_to_wall / brake_dist, min_factor)
        return vx * factor, vy * factor
    return vx, vy


def rotation_compensate(vx, vy, w, dt=LOOP_RATE):
    """Pre-rotate velocity so the world-frame path stays on target
    despite simultaneous rotation.

    Uses midpoint approximation: rotate by -w*dt/2.
    """
    if abs(w) < 0.01:
        return vx, vy
    half_rot = -w * dt * 0.5
    cos_r = math.cos(half_rot)
    sin_r = math.sin(half_rot)
    return vx * cos_r - vy * sin_r, vx * sin_r + vy * cos_r


# ═══════════════════════════════════════════════════════════════════
#  ARC-BASED APPROACH — get behind the ball without looping
# ═══════════════════════════════════════════════════════════════════

def compute_arc_nav(
    robot_xy: Tuple[float, float],
    ball: Tuple[float, float],
    aim: Tuple[float, float],
    behind_dist: float,
    avoid_radius: float,
    committed_side: Optional[int],
) -> Tuple[Tuple[float, float], int, bool]:
    """Compute a navigation target for approaching behind the ball.

    Uses arc-based pathing with committed-side hysteresis to prevent
    oscillation / looping.

    Args:
        robot_xy:       (x, y) robot position in world frame
        ball:           (x, y) ball position
        aim:            (x, y) aim target (usually inside opponent goal)
        behind_dist:    how far behind the ball to line up
        avoid_radius:   minimum clearance from ball while arcing
        committed_side: +1 / -1 from previous tick, or None on first call

    Returns:
        (nav_target, updated_committed_side, robot_is_behind_ball)
    """
    # Unit vector ball → aim
    ba_dx = aim[0] - ball[0]
    ba_dy = aim[1] - ball[1]
    ba_d = max(math.hypot(ba_dx, ba_dy), 1.0)
    ux, uy = ba_dx / ba_d, ba_dy / ba_d

    # Perpendicular (90° CCW of aim direction)
    px, py = -uy, ux

    # Behind-ball point
    behind = (ball[0] - ux * behind_dist, ball[1] - uy * behind_dist)

    # Robot relative to ball, decomposed along aim axis
    rbx = robot_xy[0] - ball[0]
    rby = robot_xy[1] - ball[1]
    along = rbx * ux + rby * uy          # >0 → aim-side (wrong side)
    perp  = rbx * px + rby * py          # lateral offset
    d_ball = math.hypot(rbx, rby)

    # ── Do we need to arc? ──────────────────────────────────
    need_arc = (along > -behind_dist * 0.15) and (d_ball < avoid_radius * 3.0)
    if d_ball < avoid_radius * 0.9 and along > -behind_dist * 0.5:
        need_arc = True

    if not need_arc:
        side = committed_side if committed_side is not None else (1 if perp >= 0 else -1)
        return behind, side, True

    # ── Commit to a side with strong hysteresis ─────────────
    HYSTERESIS = avoid_radius * 0.6

    if committed_side is None:
        if abs(perp) < 80:
            if abs(robot_xy[1]) > HALF_WID - 350:
                committed_side = -1 if robot_xy[1] > 0 else 1
            else:
                committed_side = 1 if perp >= 0 else -1
        else:
            committed_side = 1 if perp > 0 else -1
    else:
        if committed_side == 1 and perp < -HYSTERESIS:
            committed_side = -1
        elif committed_side == -1 and perp > HYSTERESIS:
            committed_side = 1

    # ── Compute arc waypoint ──────────────────────────────
    wrong_ratio = clamp(
        (along + behind_dist * 0.3) / max(avoid_radius * 2.0, 1.0),
        0.0, 1.0,
    )

    arc_r = avoid_radius * 1.3

    bk_x, bk_y = -ux, -uy
    sd_x, sd_y = px * committed_side, py * committed_side

    blend = wrong_ratio * 0.85
    dir_x = bk_x * (1.0 - blend) + sd_x * blend
    dir_y = bk_y * (1.0 - blend) + sd_y * blend
    dir_d = max(math.hypot(dir_x, dir_y), 1e-9)
    dir_x /= dir_d
    dir_y /= dir_d

    nav = (ball[0] + dir_x * arc_r, ball[1] + dir_y * arc_r)
    return nav, committed_side, False
