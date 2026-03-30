"""
Ball-chasing navigation with smooth obstacle avoidance.

The robot pursues the ball at a moderate, smooth speed while curving
around any other robots in its path.  No fixed waypoints — the ball
IS the target.

Key features:
- Smooth, moderate speed (not too fast, not too slow)
- Tangential + repulsive obstacle avoidance with cosine falloff
- Exponential smoothing so velocity never jumps between ticks
- Deceleration ramp near the ball so the robot doesn't overshoot
- Faces the ball at all times
"""

import time
import math

from TeamControl.network.robot_command import RobotCommand
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import (
    clamp, move_toward, wall_brake, rotation_compensate,
)
from TeamControl.robot.constants import (
    FIELD_LENGTH, FIELD_WIDTH, HALF_LEN, HALF_WID,
    CRUISE_SPEED, SPRINT_SPEED, MAX_W, TURN_GAIN,
    LOOP_RATE, FRAME_INTERVAL,
)

# ── Obstacle avoidance tuning ────────────────────────────────────
AVOID_DIST = 550          # start avoiding at this range (mm)
AVOID_STRENGTH = 2.5      # peak avoidance magnitude
AVOID_CRITICAL = 260      # very close → max strength

# Smoothing: 0 = no smoothing (instant), 1 = never changes.
SMOOTH_ALPHA = 0.3

# Tangential vs repulsive balance.
# Low = mostly push straight away from obstacle, minimal curving.
TANGENT_RATIO = 0.3

# ── Ball-chase speed ─────────────────────────────────────────────
CHASE_SPEED = CRUISE_SPEED * 0.85   # smooth, moderate pace
NEAR_BALL_DIST = 300                # start slowing down here
STOP_DIST = 80                      # stop this close to ball

# Keep old exports so main.py import doesn't break
WAYPOINTS_A = [
    (HALF_LEN - 500, HALF_WID - 200),
    (HALF_LEN - 500, -HALF_WID + 200),
    (-HALF_LEN + 500, -HALF_WID + 200),
    (-HALF_LEN + 500, HALF_WID - 200),
]
WAYPOINTS_B = [
    (-HALF_LEN + 500, -HALF_WID + 200),
    (-HALF_LEN + 500, HALF_WID - 200),
    (HALF_LEN - 500, HALF_WID - 200),
    (HALF_LEN - 500, -HALF_WID + 200),
]


def _smooth_strength(d_obs):
    """Cosine-based falloff — strength goes from 0 at AVOID_DIST to
    AVOID_STRENGTH at AVOID_CRITICAL, with a gentle S-curve in between.
    Below AVOID_CRITICAL it stays at AVOID_STRENGTH (capped)."""
    if d_obs >= AVOID_DIST:
        return 0.0
    if d_obs <= AVOID_CRITICAL:
        return AVOID_STRENGTH
    t = (AVOID_DIST - d_obs) / (AVOID_DIST - AVOID_CRITICAL)
    return AVOID_STRENGTH * 0.5 * (1.0 - math.cos(math.pi * t))


def _compute_avoidance(rpos, frame, is_yellow, robot_id, rel_target,
                       prev_obs_pos=None):
    """
    Compute tangential + repulsive avoidance force from all other robots.
    Uses predictive avoidance and smooth cosine falloff.
    Returns (avoid_vx, avoid_vy, closest_obstacle_distance, updated_obs_pos).
    """
    avoid_vx, avoid_vy = 0.0, 0.0
    closest_obs = 9999.0
    current_obs_pos = {}
    PREDICT_T = 0.4

    for color in (True, False):
        for oid in range(16):
            if color == is_yellow and oid == robot_id:
                continue
            try:
                other = frame.get_yellow_robots(isYellow=color, robot_id=oid)
                if isinstance(other, int) or other is None:
                    continue
                op = other.position
                ox, oy = float(op[0]), float(op[1])
                obs_key = (color, oid)
                current_obs_pos[obs_key] = (ox, oy)

                pred_x, pred_y = ox, oy
                if prev_obs_pos and obs_key in prev_obs_pos:
                    px, py = prev_obs_pos[obs_key]
                    est_vx = (ox - px) / max(FRAME_INTERVAL, 0.01)
                    est_vy = (oy - py) / max(FRAME_INTERVAL, 0.01)
                    pred_x = ox + est_vx * PREDICT_T
                    pred_y = oy + est_vy * PREDICT_T

                avoid_x = ox * 0.35 + pred_x * 0.65
                avoid_y = oy * 0.35 + pred_y * 0.65

                rel_obs = world2robot(rpos, (avoid_x, avoid_y))
                d_obs = math.hypot(rel_obs[0], rel_obs[1])

                if d_obs < closest_obs:
                    closest_obs = d_obs

                strength = _smooth_strength(d_obs)
                if strength > 0.001 and d_obs > 1:
                    inv_d = 1.0 / d_obs

                    # Repulsive: push directly away from obstacle
                    rep_x = -rel_obs[0] * inv_d * strength
                    rep_y = -rel_obs[1] * inv_d * strength

                    # Tangential: curve around (perpendicular to obstacle direction)
                    tang_x =  rel_obs[1] * inv_d
                    tang_y = -rel_obs[0] * inv_d
                    # Pick the tangential direction that moves toward target
                    if tang_x * rel_target[0] + tang_y * rel_target[1] < 0:
                        tang_x, tang_y = -tang_x, -tang_y

                    avoid_vx += rep_x + tang_x * strength * TANGENT_RATIO
                    avoid_vy += rep_y + tang_y * strength * TANGENT_RATIO
            except Exception:
                continue

    return avoid_vx, avoid_vy, closest_obs, current_obs_pos


# ═════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — ball chaser with obstacle avoidance
# ═════════════════════════════════════════════════════════════════

def run_navigator(is_running, dispatch_q, wm, robot_id, is_yellow,
                  waypoints=None):
    """
    Chase the ball smoothly while avoiding every other robot on the field.
    The `waypoints` argument is accepted for API compatibility but ignored.
    """
    frame = None
    prev_obs_pos = {}
    # Exponential-smoothed output velocities
    sm_vx, sm_vy, sm_w = 0.0, 0.0, 0.0

    while is_running.is_set():
        # ── Fetch frame (every tick for fast obstacle reaction) ─
        try:
            f = wm.get_latest_frame()
            if f is not None:
                frame = f
        except Exception:
            pass

        if frame is None or frame.ball is None:
            time.sleep(LOOP_RATE)
            continue

        try:
            robot = frame.get_yellow_robots(isYellow=is_yellow,
                                            robot_id=robot_id)
        except Exception:
            time.sleep(LOOP_RATE)
            continue

        if isinstance(robot, int):
            time.sleep(LOOP_RATE)
            continue

        rp = robot.position
        bp = frame.ball.position
        rpos = (float(rp[0]), float(rp[1]), float(rp[2]))
        ball = (float(bp[0]), float(bp[1]))

        # ── Target: the ball ─────────────────────────────────
        rel_ball = world2robot(rpos, ball)
        d_ball = math.hypot(rel_ball[0], rel_ball[1])

        # ── Obstacle avoidance (predictive) ──────────────────
        avoid_vx, avoid_vy, closest_obs, prev_obs_pos = _compute_avoidance(
            rpos, frame, is_yellow, robot_id, rel_ball, prev_obs_pos)

        # ── Base navigation toward ball ──────────────────────
        # Smoothly reduce speed near obstacles using cosine interpolation
        if closest_obs < AVOID_DIST:
            if closest_obs <= AVOID_CRITICAL:
                speed_frac = 0.5
            else:
                t = (AVOID_DIST - closest_obs) / (AVOID_DIST - AVOID_CRITICAL)
                speed_frac = 1.0 - 0.5 * 0.5 * (1.0 - math.cos(math.pi * t))
            base_speed = CHASE_SPEED * speed_frac
        else:
            base_speed = CHASE_SPEED

        nav_vx, nav_vy = move_toward(rel_ball, base_speed,
                                     ramp_dist=NEAR_BALL_DIST,
                                     stop_dist=STOP_DIST,
                                     min_speed=0.06)

        # ── Blend navigation + avoidance ─────────────────────
        raw_vx = nav_vx + avoid_vx
        raw_vy = nav_vy + avoid_vy

        # Cap speed — allow a bit more when actively dodging
        speed = math.hypot(raw_vx, raw_vy)
        max_spd = CHASE_SPEED * 1.3 if closest_obs < AVOID_DIST else CHASE_SPEED * 1.1
        if speed > max_spd:
            raw_vx = raw_vx / speed * max_spd
            raw_vy = raw_vy / speed * max_spd

        # ── Wall braking ────────────────────────────────────
        raw_vx, raw_vy = wall_brake(rpos[0], rpos[1], raw_vx, raw_vy)

        # ── Face the ball ────────────────────────────────────
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])
        if abs(ang_ball) < 0.05:
            raw_w = 0.0
        else:
            raw_w = clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)

        # ── Exponential smoothing ─────────────────────────────
        # Blends previous output with new target so the robot
        # never jerks — gives a gentle, curved motion.
        a = SMOOTH_ALPHA
        sm_vx = a * sm_vx + (1.0 - a) * raw_vx
        sm_vy = a * sm_vy + (1.0 - a) * raw_vy
        sm_w  = a * sm_w  + (1.0 - a) * raw_w

        # ── Rotation compensation ───────────────────────────
        out_vx, out_vy = rotation_compensate(sm_vx, sm_vy, sm_w)

        cmd = RobotCommand(robot_id=robot_id, vx=out_vx, vy=out_vy, w=sm_w,
                           kick=0, dribble=0, isYellow=is_yellow)
        dispatch_q.put((cmd, 0.15))
        time.sleep(LOOP_RATE)
