"""
Simple goalie — position, save, clear.

Three behaviors, checked in order each tick:
  1. SAVE     — ball moving fast toward our goal → sprint to intercept point
  2. CLEAR    — ball slow inside our penalty box → go kick it out
  3. POSITION — default: sit between ball and goal center, narrowing the angle
"""

import time
import math

from TeamControl.network.robot_command import RobotCommand
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import clamp, clamp_for_role, move_toward
from TeamControl.cache import TickCache
from TeamControl.robot.constants import (
    SAVE_SPEED, POSITION_SPEED, CLEAR_SPEED,
    MAX_W, FACE_BALL_GAIN,
    SHOT_SPEED, KICK_DIST,
    LOOP_RATE, FRAME_INTERVAL,
)
from TeamControl.world.field_config import (
    FIELD_LENGTH_MM,
    FIELD_WIDTH_MM,
    GOAL_HALF_WIDTH_MM,
    DEFENCE_X_MM,
    DEFENCE_Y_MM,
)

# ── Tuning ───────────────────────────────────────────────────────
POSITION_SPD   = 0.35       # gentle repositioning
SAVE_SPD       = SAVE_SPEED # fast save sprint
CLEAR_SPD      = 0.30       # controlled approach to ball
CLEAR_KICK_SPD = 0.25       # speed when actually kicking


def _clamp_to_box(x, y, goal_x):
    """Keep position inside the penalty box.

    x-depth uses the shared clamp_for_role rule (same one
    RobotMotionController/Movement.py use) -- goal_x is passed explicitly
    since this file derives its own goal line from the live field length
    each tick, same source clamp_for_role falls back to anyway.
    y-width is goalie-specific (penalty box is wider than max-advance's
    depth alone implies) and stays local.
    """
    margin = 60
    x, _ = clamp_for_role((x, y), is_goalie=True, margin=margin,
                          own_goal_positive_side=goal_x > 0, own_goal_x=goal_x)
    penalty_hw = float(DEFENCE_Y_MM)
    y = clamp(y, -penalty_hw + margin, penalty_hw - margin)
    return x, y


def run_goalie(is_running, dispatch_q, wm, goalie_id, is_yellow):
    cache = TickCache(wm)

    while is_running.is_set():
        now = time.time()

        # ── Refresh cache from latest frame ──────────────────
        if not cache.refresh(now):
            time.sleep(LOOP_RATE)
            continue
        if not cache.ball.visible:
            time.sleep(LOOP_RATE)
            continue

        rpos = cache.robots.get_position(is_yellow, goalie_id)
        if rpos is None:
            time.sleep(LOOP_RATE)
            continue

        ball = cache.ball.position
        is_positive = cache.team.us_positive

        sign = 1 if is_positive else -1
        half_len = FIELD_LENGTH_MM / 2.0
        goal_x = sign * half_len
        goal_hw = float(GOAL_HALF_WIDTH_MM)
        penalty_depth = float(DEFENCE_X_MM)
        penalty_hw = float(DEFENCE_Y_MM)
        max_advance = float(DEFENCE_X_MM) - 50.0

        # ── Ball velocity (cached; recomputed only on new frame) ──
        bvx, bvy, bspeed = cache.ball.velocity

        # ── Distances ────────────────────────────────────────
        rel_ball = world2robot(rpos, ball)
        d_ball = math.hypot(rel_ball[0], rel_ball[1])
        ball_dist_from_goal = abs(ball[0] - goal_x)
        ball_in_box = (ball_dist_from_goal < penalty_depth
                       and abs(ball[1]) < penalty_hw)

        kick, dribble = 0, 0

        # ═════════════════════════════════════════════════════
        #  1. SAVE — ball heading toward goal fast
        # ═════════════════════════════════════════════════════
        shot_incoming = False
        pred_y = 0.0

        ball_toward_us = (bvx * sign > 80)
        if ball_toward_us and bspeed > SHOT_SPEED and abs(bvx) > 40:
            t_cross = (goal_x - ball[0]) / bvx
            if 0 < t_cross < 2.0:
                pred_y_raw = ball[1] + bvy * t_cross
                if abs(pred_y_raw) < goal_hw + 250:
                    shot_incoming = True
                    pred_y = clamp(pred_y_raw, -goal_hw, goal_hw)

        if shot_incoming:
            # Sprint to the predicted crossing point
            save_x = goal_x + (ball[0] - goal_x) * 0.08
            tx, ty = _clamp_to_box(save_x, pred_y, goal_x)
            rel_t = world2robot(rpos, (tx, ty))
            vx, vy = move_toward(rel_t, SAVE_SPD, ramp_dist=150, stop_dist=10)

        # ═════════════════════════════════════════════════════
        #  2. CLEAR — ball slow in our box, go kick it out
        # ═════════════════════════════════════════════════════
        elif ball_in_box and bspeed < 500 and d_ball < 900:
            if d_ball < KICK_DIST and rel_ball[0] > 0:
                # Close enough — kick toward sideline
                outward = -1.0 if goal_x > 0 else 1.0
                side_y = FIELD_WIDTH_MM / 2.0 if ball[1] > 0 else -FIELD_WIDTH_MM / 2.0
                clear_pt = (ball[0] + outward * 1500, side_y)
                rel_clear = world2robot(rpos, clear_pt)
                ang_clear = math.atan2(rel_clear[1], rel_clear[0])
                if abs(ang_clear) < 0.40:
                    kick = 1
                else:
                    dribble = 1
                vx, vy = move_toward(rel_ball, CLEAR_KICK_SPD, ramp_dist=120,
                                     stop_dist=10)
            else:
                # Move toward ball
                vx, vy = move_toward(rel_ball, CLEAR_SPD, ramp_dist=300,
                                     stop_dist=10)
                dribble = 1 if d_ball < 300 else 0

        # ═════════════════════════════════════════════════════
        #  3. POSITION — between ball and goal, narrow angle
        # ═════════════════════════════════════════════════════
        else:
            # Direction from goal to ball
            dx = ball[0] - goal_x
            dy = ball[1]
            dist = math.hypot(dx, dy)

            if dist > 1:
                # Advance more when ball is closer
                ratio = 1.0 - clamp(ball_dist_from_goal / half_len, 0, 1)
                advance = ratio * max_advance
                tx = goal_x + (dx / dist) * advance
                ty = (dy / dist) * advance
                ty = clamp(ty, -(goal_hw + 150), goal_hw + 150)
            else:
                tx, ty = goal_x, 0.0

            tx, ty = _clamp_to_box(tx, ty, goal_x)
            rel_t = world2robot(rpos, (tx, ty))
            vx, vy = move_toward(rel_t, POSITION_SPD, ramp_dist=200,
                                 stop_dist=10)

        # ── Always face the ball ─────────────────────────────
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])
        if abs(ang_ball) < 0.04:
            w = 0.0
        else:
            w = clamp(ang_ball * FACE_BALL_GAIN, -MAX_W, MAX_W)

        cmd = RobotCommand(robot_id=goalie_id, vx=vx, vy=vy, w=w,
                           kick=kick, dribble=dribble, isYellow=is_yellow)
        dispatch_q.put((cmd, 0.15))
        time.sleep(LOOP_RATE)
