"""
Cooperative 2-bot striker — two robots on the same team work together to score.

Role assignment each tick (no shared state needed):
  - Closer to ball  → ATTACKER: approach ball, shoot or pass
  - Farther from ball → SUPPORT: position for a pass, take shots of opportunity

Both robots avoid every other robot on the field (teammates, opponents, obstacles).
"""

import time
import math

from TeamControl.network.robot_command import RobotCommand
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import (
    clamp, move_toward, wall_brake, rotation_compensate,
    compute_arc_nav,
)
from TeamControl.robot.navigator import _compute_avoidance, _smooth_strength
from TeamControl.robot.constants import (
    FIELD_LENGTH, HALF_LEN, HALF_WID,
    GOAL_WIDTH, GOAL_HW, GOAL_DEPTH,
    PENALTY_DEPTH, PENALTY_HW,
    BEHIND_DIST, KICK_RANGE, BALL_NEAR, AVOID_RADIUS,
    CRUISE_SPEED, CHARGE_SPEED, DRIBBLE_SPEED, SPRINT_SPEED,
    MAX_W, TURN_GAIN, FACE_BALL_GAIN,
    KICK_COOLDOWN, LOOP_RATE, FRAME_INTERVAL,
)

# ── Tuning ───────────────────────────────────────────────────────
SHOOT_RANGE     = 350       # mm — shoot from this far if aimed
SHOOT_ALIGN     = 0.18      # rad — alignment for distance shot
KICK_ALIGN      = 0.18      # rad — alignment for close kick
PASS_ALIGN      = 0.25      # rad — alignment tolerance for passing
PASS_MIN_DIST   = 400       # mm — teammate must be at least this far to pass
SUPPORT_OFFSET  = 600       # mm — how far the support positions from the ball
SUPPORT_LATERAL = 500       # mm — lateral offset from ball-goal line
FORCE_KICK_TIME = 0.8       # s  — force kick after this long near ball
POSSESSION_DIST = 350       # mm — attacker "has" the ball if this close
KEEPAWAY_RADIUS = 600       # mm — support stays at least this far from ball when attacker has possession
APPROACH_SPD    = CRUISE_SPEED
DRIBBLE_SPD     = DRIBBLE_SPEED


def _get_robot(frame, is_yellow, rid):
    """Get a robot's (x, y, theta) or None."""
    try:
        r = frame.get_yellow_robots(isYellow=is_yellow, robot_id=rid)
        if isinstance(r, int) or r is None:
            return None
        p = r.position
        return (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        return None


def _in_penalty_box(px, py, goal_x):
    return abs(px - goal_x) < PENALTY_DEPTH and abs(py) < PENALTY_HW


def run_coop(is_running, dispatch_q, wm, robot_id, teammate_id,
             is_yellow=True):
    """Cooperative striker process for one robot.

    Both robots on the team run this function (with swapped robot_id /
    teammate_id).  Each independently decides its role every tick based
    on who is closer to the ball.
    """
    frame = None
    last_ft = 0.0
    last_kick = 0.0
    committed_side = None
    near_ball_since = 0.0
    prev_obs_pos = {}

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

        me = _get_robot(frame, is_yellow, robot_id)
        mate = _get_robot(frame, is_yellow, teammate_id)
        if me is None:
            time.sleep(LOOP_RATE)
            continue

        bp = frame.ball.position
        ball = (float(bp[0]), float(bp[1]))

        try:
            us_positive = wm.us_positive()
        except Exception:
            us_positive = True

        # Opponent goal
        if is_yellow:
            goal_x = -HALF_LEN if us_positive else HALF_LEN
        else:
            goal_x = HALF_LEN if us_positive else -HALF_LEN

        # ── Decide role ──────────────────────────────────────
        my_d = math.hypot(me[0] - ball[0], me[1] - ball[1])
        if mate is not None:
            mate_d = math.hypot(mate[0] - ball[0], mate[1] - ball[1])
            # Tie-break by robot_id so they don't both pick the same role
            i_am_attacker = (my_d < mate_d) or (my_d == mate_d and robot_id < teammate_id)
        else:
            i_am_attacker = True

        # ── Possession check ─────────────────────────────────
        # Attacker "has" the ball if very close to it
        attacker_has_ball = False
        if mate is not None:
            if i_am_attacker:
                attacker_has_ball = my_d < POSSESSION_DIST
            else:
                attacker_has_ball = mate_d < POSSESSION_DIST

        # ── Common vectors ───────────────────────────────────
        rel_ball = world2robot(me, ball)
        d_ball = math.hypot(rel_ball[0], rel_ball[1])
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])

        aim = (goal_x, 0.0)
        rel_aim = world2robot(me, aim)
        ang_aim = math.atan2(rel_aim[1], rel_aim[0])

        vx, vy, w = 0.0, 0.0, 0.0
        kick, dribble = 0, 0

        if i_am_attacker:
            # ═══════════════════════════════════════════════════
            #  ATTACKER — go for the ball, shoot or pass
            # ═══════════════════════════════════════════════════

            # Check if teammate is in a better shooting position
            can_pass = False
            pass_target = None
            if mate is not None and (now - last_kick) > KICK_COOLDOWN:
                mate_to_goal = abs(mate[0] - goal_x)
                me_to_goal = abs(me[0] - goal_x)
                mate_clear = abs(mate[1]) < GOAL_HW + 200
                if (mate_to_goal < me_to_goal
                        and mate_d > PASS_MIN_DIST
                        and mate_clear):
                    rel_mate = world2robot(me, (mate[0], mate[1]))
                    ang_mate = math.atan2(rel_mate[1], rel_mate[0])
                    if abs(ang_mate) < PASS_ALIGN and rel_mate[0] > 0:
                        can_pass = True
                        pass_target = (mate[0], mate[1])

            # Penalty box — wait outside
            if _in_penalty_box(ball[0], ball[1], goal_x):
                inward = -1.0 if goal_x > 0 else 1.0
                wait_pt = (goal_x + inward * (PENALTY_DEPTH + 120),
                           clamp(ball[1], -PENALTY_HW, PENALTY_HW))
                rel_w = world2robot(me, wait_pt)
                vx, vy = move_toward(rel_w, CRUISE_SPEED * 0.6)
                w = clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)
                committed_side = None
                near_ball_since = 0.0

            # Distance shot — ball in front, aimed at goal
            elif (d_ball < SHOOT_RANGE and rel_ball[0] > 0
                  and abs(ang_aim) < SHOOT_ALIGN
                  and (now - last_kick) > KICK_COOLDOWN):
                kick = 1
                vx, vy = move_toward(rel_ball, CHARGE_SPEED,
                                     ramp_dist=100, stop_dist=0)
                w = clamp(ang_aim * TURN_GAIN, -MAX_W, MAX_W)
                last_kick = now
                near_ball_since = 0.0
                committed_side = None

            # Pass to teammate
            elif (can_pass and d_ball < KICK_RANGE * 1.5
                  and rel_ball[0] > 0
                  and (now - last_kick) > KICK_COOLDOWN):
                kick = 1
                vx, vy = move_toward(rel_ball, CHARGE_SPEED,
                                     ramp_dist=100, stop_dist=0)
                rel_pt = world2robot(me, pass_target)
                w = clamp(math.atan2(rel_pt[1], rel_pt[0]) * TURN_GAIN,
                          -MAX_W, MAX_W)
                last_kick = now
                near_ball_since = 0.0
                committed_side = None

            # Close kick
            elif d_ball < KICK_RANGE and rel_ball[0] > 0:
                dribble = 1
                vx, vy = move_toward(rel_ball, DRIBBLE_SPD,
                                     ramp_dist=150, stop_dist=10)
                w = clamp(ang_aim * TURN_GAIN, -MAX_W, MAX_W)
                committed_side = None

                if near_ball_since == 0.0:
                    near_ball_since = now
                force = (now - near_ball_since) > FORCE_KICK_TIME

                if (now - last_kick) > KICK_COOLDOWN and (
                    abs(ang_aim) < KICK_ALIGN or force
                ):
                    kick = 1
                    dribble = 0
                    vx = DRIBBLE_SPD
                    vy = 0.0
                    last_kick = now
                    near_ball_since = 0.0

            # Approach — arc behind ball
            else:
                near_ball_since = 0.0
                nav, committed_side, is_behind = compute_arc_nav(
                    robot_xy=(me[0], me[1]),
                    ball=ball, aim=aim,
                    behind_dist=BEHIND_DIST,
                    avoid_radius=AVOID_RADIUS,
                    committed_side=committed_side,
                )
                rel_nav = world2robot(me, nav)
                d_nav = math.hypot(rel_nav[0], rel_nav[1])

                if is_behind and d_nav < 300 and d_ball < BALL_NEAR:
                    dribble = 1
                    vx, vy = move_toward(rel_ball, DRIBBLE_SPD,
                                         ramp_dist=300, stop_dist=10)
                    w = clamp(ang_aim * TURN_GAIN * 0.7, -MAX_W, MAX_W)
                else:
                    vx, vy = move_toward(rel_nav, APPROACH_SPD,
                                         ramp_dist=400, stop_dist=10)
                    w = clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)

        else:
            # ═══════════════════════════════════════════════════
            #  SUPPORT — position for a pass / rebound
            # ═══════════════════════════════════════════════════
            committed_side = None
            near_ball_since = 0.0

            # Position: between ball and goal, offset laterally
            # Pick the side away from the attacker
            if mate is not None:
                side = -1.0 if mate[1] > ball[1] else 1.0
            else:
                side = 1.0 if ball[1] < 0 else -1.0

            # Direction from ball toward goal
            dx = goal_x - ball[0]
            dy = 0.0 - ball[1]
            d = max(math.hypot(dx, dy), 1.0)
            ux, uy = dx / d, dy / d
            # Perpendicular
            px, py = -uy, ux

            support_x = ball[0] + ux * SUPPORT_OFFSET + px * side * SUPPORT_LATERAL
            support_y = ball[1] + uy * SUPPORT_OFFSET + py * side * SUPPORT_LATERAL
            # Clamp to field
            support_x = clamp(support_x, -HALF_LEN + 200, HALF_LEN - 200)
            support_y = clamp(support_y, -HALF_WID + 200, HALF_WID - 200)

            # If attacker has possession, enforce keepaway from the ball
            if attacker_has_ball:
                dist_to_ball = math.hypot(support_x - ball[0],
                                          support_y - ball[1])
                if dist_to_ball < KEEPAWAY_RADIUS:
                    # Push support point away from ball
                    away_dx = support_x - ball[0]
                    away_dy = support_y - ball[1]
                    away_d = max(math.hypot(away_dx, away_dy), 1.0)
                    support_x = ball[0] + (away_dx / away_d) * KEEPAWAY_RADIUS
                    support_y = ball[1] + (away_dy / away_d) * KEEPAWAY_RADIUS
                    support_x = clamp(support_x, -HALF_LEN + 200, HALF_LEN - 200)
                    support_y = clamp(support_y, -HALF_WID + 200, HALF_WID - 200)

            target = (support_x, support_y)
            rel_t = world2robot(me, target)
            vx, vy = move_toward(rel_t, CRUISE_SPEED, ramp_dist=300,
                                 stop_dist=60)

            # Face the ball
            w = clamp(ang_ball * FACE_BALL_GAIN, -MAX_W, MAX_W)

            # Opportunistic shot — ball close and aimed at goal (only if attacker doesn't have it)
            if (not attacker_has_ball
                    and d_ball < SHOOT_RANGE and rel_ball[0] > 0
                    and abs(ang_aim) < SHOOT_ALIGN
                    and (now - last_kick) > KICK_COOLDOWN):
                kick = 1
                vx, vy = move_toward(rel_ball, CHARGE_SPEED,
                                     ramp_dist=100, stop_dist=0)
                w = clamp(ang_aim * TURN_GAIN, -MAX_W, MAX_W)
                last_kick = now

        # ── Obstacle avoidance ────────────────────────────────
        rel_target = world2robot(me, ball)
        avoid_vx, avoid_vy, closest_obs, prev_obs_pos = _compute_avoidance(
            me, frame, is_yellow, robot_id, rel_target, prev_obs_pos)
        vx += avoid_vx
        vy += avoid_vy

        # Cap speed
        spd = math.hypot(vx, vy)
        max_spd = SPRINT_SPEED if kick else CRUISE_SPEED * 1.2
        if spd > max_spd:
            vx = vx / spd * max_spd
            vy = vy / spd * max_spd

        # ── Wall braking ──────────────────────────────────────
        vx, vy = wall_brake(me[0], me[1], vx, vy)

        # ── Rotation compensation ─────────────────────────────
        vx, vy = rotation_compensate(vx, vy, w)

        cmd = RobotCommand(robot_id=robot_id, vx=vx, vy=vy, w=w,
                           kick=kick, dribble=dribble,
                           isYellow=is_yellow)
        dispatch_q.put((cmd, 0.15))
        time.sleep(LOOP_RATE)
