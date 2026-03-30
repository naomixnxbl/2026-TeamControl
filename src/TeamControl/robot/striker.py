"""
Simple striker — approach, align, kick.

Three behaviors, checked in order each tick:
  1. WAIT     — ball inside a penalty box → hold outside, face ball
  2. KICK     — ball close and in front → dribble, align to goal, kick
  3. APPROACH — default: arc around to get behind ball, drive toward it
"""

import time
import math

from TeamControl.network.robot_command import RobotCommand
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import (
    clamp, move_toward, compute_arc_nav, wall_brake, rotation_compensate,
)
from TeamControl.robot.navigator import _compute_avoidance
from TeamControl.robot.constants import (
    FIELD_LENGTH, HALF_LEN, HALF_WID,
    GOAL_WIDTH, GOAL_HW, GOAL_DEPTH,
    PENALTY_DEPTH, PENALTY_HW,
    BEHIND_DIST, KICK_RANGE, BALL_NEAR, AVOID_RADIUS,
    CRUISE_SPEED, CHARGE_SPEED, DRIBBLE_SPEED,
    MAX_W, TURN_GAIN,
    KICK_COOLDOWN, LOOP_RATE, FRAME_INTERVAL,
    DEFENSE_DEPTH,
)

# ── Tuning ───────────────────────────────────────────────────────
APPROACH_SPD   = CRUISE_SPEED        # fast approach
DRIBBLE_SPD    = DRIBBLE_SPEED      # gentle when close
WAIT_SPD       = CRUISE_SPEED * 0.6 # repositioning outside box
DWELL_TIME     = 0.08               # seconds ball must be stable before kick
KICK_ALIGN_TOL = 0.18               # rad — alignment tolerance for dwell
FORCE_KICK_TIME = 0.6               # if near ball this long, just kick
SHOOT_RANGE    = 350                # mm — take a shot from this far if aimed
SHOOT_ALIGN    = 0.18               # rad — alignment needed for distance shot


def _in_penalty_box(px, py, goal_x):
    return abs(px - goal_x) < PENALTY_DEPTH and abs(py) < PENALTY_HW


def _pick_aim(ball, goal_x, frame, is_yellow):
    """Pick aim point in goal mouth, away from opponent goalie."""
    aim_inward = 1 if goal_x > 0 else -1
    aim_x = goal_x + aim_inward * (GOAL_DEPTH * 0.5)

    # Find opponent goalie
    gk_y = None
    opp_yellow = not is_yellow
    for oid in range(6):
        try:
            opp = frame.get_yellow_robots(isYellow=opp_yellow, robot_id=oid)
            if isinstance(opp, int) or opp is None:
                continue
            op = opp.position
            if abs(float(op[0]) - goal_x) < DEFENSE_DEPTH:
                gk_y = float(op[1])
                break
        except Exception:
            continue

    # Pick the side of goal furthest from goalie
    if gk_y is not None:
        aim_y = -GOAL_HW * 0.6 if gk_y > 0 else GOAL_HW * 0.6
    else:
        aim_y = 0.0

    return (aim_x, aim_y)


# ── Main loop ─────────────────────────────────────────────────────

BALL_MEMORY_TIME = 0.5    # seconds to trust last-known ball when occluded

def run_striker(is_running, dispatch_q, wm, robot_id=0, is_yellow=True):
    last_kick = 0.0
    frame = None
    last_ft = 0.0
    committed_side = None
    dwell_start = 0.0       # when alignment started
    aligned = False         # True while continuously aligned
    last_ball = None         # last known ball (x, y)
    last_ball_time = 0.0     # when ball was last seen
    last_d_ball = float('inf')  # distance to ball when last seen
    near_ball_since = 0.0    # when we first got close to ball
    prev_obs_pos = {}        # obstacle positions for predictive avoidance

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

        if frame is None:
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
        rpos = (float(rp[0]), float(rp[1]), float(rp[2]))

        # ── Resolve ball position (use memory if occluded) ───
        ball_visible = frame.ball is not None
        if ball_visible:
            bp = frame.ball.position
            ball = (float(bp[0]), float(bp[1]))
            last_ball = ball
            last_ball_time = now
        elif last_ball is not None and \
             (now - last_ball_time) < BALL_MEMORY_TIME and \
             last_d_ball < BALL_NEAR:
            # Ball disappeared while close — likely occluded by us
            ball = last_ball
        else:
            time.sleep(LOOP_RATE)
            continue

        try:
            us_positive = wm.us_positive()
        except Exception:
            us_positive = True

        # Opponent goal — where we want to score
        if is_yellow:
            goal_x = -HALF_LEN if us_positive else HALF_LEN
        else:
            goal_x = HALF_LEN if us_positive else -HALF_LEN

        our_goal_x = -goal_x

        # ── Robot-local vectors ──────────────────────────────
        rel_ball = world2robot(rpos, ball)
        d_ball = math.hypot(rel_ball[0], rel_ball[1])
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])
        if ball_visible:
            last_d_ball = d_ball

        aim = _pick_aim(ball, goal_x, frame, is_yellow)
        rel_aim = world2robot(rpos, aim)
        ang_aim = math.atan2(rel_aim[1], rel_aim[0])

        vx, vy, w = 0.0, 0.0, 0.0
        kick, dribble = 0, 0

        # ═════════════════════════════════════════════════════
        #  1. SHOOT — ball in front and aimed at goal, kick now
        # ═════════════════════════════════════════════════════
        if (d_ball < SHOOT_RANGE and rel_ball[0] > 0
              and abs(ang_aim) < SHOOT_ALIGN
              and (now - last_kick) > KICK_COOLDOWN):
            kick = 1
            dribble = 0
            # Charge forward into the ball
            vx, vy = move_toward(rel_ball, CHARGE_SPEED, ramp_dist=100,
                                 stop_dist=0)
            w = clamp(ang_aim * TURN_GAIN, -MAX_W, MAX_W)
            last_kick = now
            near_ball_since = 0.0
            committed_side = None

        # ═════════════════════════════════════════════════════
        #  3. KICK — ball close and in front, align and shoot
        # ═════════════════════════════════════════════════════
        elif d_ball < KICK_RANGE and (rel_ball[0] > 0 or not ball_visible):
            dribble = 1
            # Drive into ball with dribbler — push forward with front
            vx, vy = move_toward(rel_ball, DRIBBLE_SPD, ramp_dist=150,
                                 stop_dist=10)
            committed_side = None

            # Rotate toward aim target
            w = clamp(ang_aim * TURN_GAIN, -MAX_W, MAX_W)

            # Track how long we've been near the ball
            if near_ball_since == 0.0:
                near_ball_since = now
            near_ball_duration = now - near_ball_since
            force_kick = near_ball_duration > FORCE_KICK_TIME

            # Ball occluded while very close — kick immediately
            if not ball_visible and (now - last_kick) > KICK_COOLDOWN:
                kick = 1
                dribble = 0
                vx = DRIBBLE_SPD
                vy = 0.0
                last_kick = now
                near_ball_since = 0.0
                aligned = False
            elif (now - last_kick) > KICK_COOLDOWN and (
                (abs(ang_aim) < KICK_ALIGN_TOL) or force_kick
            ):
                # Kick when roughly aligned OR forced after timeout
                kick = 1
                dribble = 0
                vx = DRIBBLE_SPD
                vy = 0.0
                last_kick = now
                near_ball_since = 0.0
                aligned = False

        # ═════════════════════════════════════════════════════
        #  3. APPROACH — arc around ball, get behind, drive in
        # ═════════════════════════════════════════════════════
        else:
            near_ball_since = 0.0
            nav, committed_side, is_behind = compute_arc_nav(
                robot_xy=(rpos[0], rpos[1]),
                ball=ball,
                aim=aim,
                behind_dist=BEHIND_DIST,
                avoid_radius=AVOID_RADIUS,
                committed_side=committed_side,
            )

            rel_nav = world2robot(rpos, nav)
            d_nav = math.hypot(rel_nav[0], rel_nav[1])

            if is_behind and d_nav < 300 and d_ball < BALL_NEAR:
                # Lined up behind ball — drive with dribbler
                dribble = 1
                vx, vy = move_toward(rel_ball, DRIBBLE_SPD,
                                     ramp_dist=300, stop_dist=10)
                w = clamp(ang_aim * TURN_GAIN * 0.7, -MAX_W, MAX_W)
            else:
                # Navigate toward arc waypoint or behind-ball point
                vx, vy = move_toward(rel_nav, APPROACH_SPD, ramp_dist=400,
                                     stop_dist=10)
                w = clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)

        # ── Obstacle avoidance (opponent + teammate) ──────
        avoid_vx, avoid_vy, _closest, prev_obs_pos = _compute_avoidance(
            rpos, frame, is_yellow, robot_id, rel_ball, prev_obs_pos)
        vx += avoid_vx
        vy += avoid_vy

        # ── Wall braking ───────────────────────────────────
        vx, vy = wall_brake(rpos[0], rpos[1], vx, vy)

        # ── Rotation compensation ──────────────────────────
        vx, vy = rotation_compensate(vx, vy, w)

        # ── Always face the ball ───────────────────────────
        if kick == 0 and dribble == 0:
            ang_ball = math.atan2(rel_ball[1], rel_ball[0])
            if abs(ang_ball) < 0.04:
                w = 0.0
            else:
                w = clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)

        cmd = RobotCommand(robot_id=robot_id, vx=vx, vy=vy, w=w,
                           kick=kick, dribble=dribble,
                           isYellow=is_yellow)
        dispatch_q.put((cmd, 0.15))
        time.sleep(LOOP_RATE)
