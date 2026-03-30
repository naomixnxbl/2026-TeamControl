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
from TeamControl.robot.ball_nav import (
    clamp, move_toward, ball_velocity, update_ball_history,
)
from TeamControl.robot.constants import (
    FIELD_LENGTH, HALF_LEN, HALF_WID,
    GOAL_WIDTH, GOAL_HW,
    PENALTY_DEPTH, PENALTY_HW,
    MAX_ADVANCE,
    SAVE_SPEED, POSITION_SPEED, CLEAR_SPEED,
    MAX_W, FACE_BALL_GAIN,
    SHOT_SPEED, KICK_DIST,
    LOOP_RATE, FRAME_INTERVAL,
)

# ── Tuning ───────────────────────────────────────────────────────
POSITION_SPD   = 0.35       # gentle repositioning
SAVE_SPD       = SAVE_SPEED # fast save sprint
CLEAR_SPD      = 0.30       # controlled approach to ball
CLEAR_KICK_SPD = 0.25       # speed when actually kicking


def _clamp_to_box(x, y, goal_x):
    """Keep position inside the penalty box."""
    margin = 60
    if goal_x > 0:
        x = clamp(x, goal_x - PENALTY_DEPTH + margin, goal_x - margin)
    else:
        x = clamp(x, goal_x + margin, goal_x + PENALTY_DEPTH - margin)
    y = clamp(y, -PENALTY_HW + margin, PENALTY_HW - margin)
    return x, y


def run_goalie(is_running, dispatch_q, wm, goalie_id, is_yellow):
    frame = None
    last_ft = 0.0
    ball_history = []
    last_ball_xy = None

    while is_running.is_set():
        now = time.time()

        # ── Get frame ────────────────────────────────────────
        if now - last_ft > FRAME_INTERVAL:
            try:
                f = wm.get_latest_frame()
                if f is not None:
                    frame = f
            except Exception:
                pass
            last_ft = now

        if frame is None or frame.ball is None:
            time.sleep(LOOP_RATE)
            continue

        try:
            robot = frame.get_yellow_robots(isYellow=is_yellow,
                                            robot_id=goalie_id)
        except Exception:
            time.sleep(LOOP_RATE)
            continue
        if isinstance(robot, int):
            time.sleep(LOOP_RATE)
            continue

        try:
            is_positive = wm.us_positive()
        except Exception:
            is_positive = True

        rp = robot.position
        bp = frame.ball.position
        ball = (float(bp[0]), float(bp[1]))
        rpos = (float(rp[0]), float(rp[1]), float(rp[2]))

        sign = 1 if is_positive else -1
        goal_x = sign * (FIELD_LENGTH / 2)

        # ── Ball velocity ────────────────────────────────────
        last_ball_xy = update_ball_history(ball_history, now, ball,
                                           last_ball_xy)
        bvx, bvy, bspeed = ball_velocity(ball_history)

        # ── Distances ────────────────────────────────────────
        rel_ball = world2robot(rpos, ball)
        d_ball = math.hypot(rel_ball[0], rel_ball[1])
        ball_dist_from_goal = abs(ball[0] - goal_x)
        ball_in_box = (ball_dist_from_goal < PENALTY_DEPTH
                       and abs(ball[1]) < PENALTY_HW)

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
                if abs(pred_y_raw) < GOAL_HW + 250:
                    shot_incoming = True
                    pred_y = clamp(pred_y_raw, -GOAL_HW, GOAL_HW)

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
                side_y = HALF_WID if ball[1] > 0 else -HALF_WID
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
                ratio = 1.0 - clamp(ball_dist_from_goal / (FIELD_LENGTH * 0.5), 0, 1)
                advance = ratio * MAX_ADVANCE
                tx = goal_x + (dx / dist) * advance
                ty = (dy / dist) * advance
                ty = clamp(ty, -(GOAL_HW + 150), GOAL_HW + 150)
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
