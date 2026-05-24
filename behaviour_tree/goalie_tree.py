"""Goalie behaviour tree — enhanced with shot detection, ball velocity tracking,
wall-bounce prediction, smart clearing, and aggressive angle narrowing.

  GetWorldPositionUpdate -> GetRobotIDPosition -> CalculateGoalieAction -> SendRobotCommand
"""

import math
import time
import py_trees
from TeamControl.robot.path_planner import move_toward_relative, turn_toward
from TeamControl.world.transform_cords import world2robot

from TeamControl.robot.constants import MAX_W, FACE_BALL_GAIN

from .common_trees import GetWorldPositionUpdate, GetRobotIDPosition, SendRobotCommand

FIELD_LENGTH = 9000
FIELD_WIDTH = 6000
HALF_LEN = FIELD_LENGTH / 2
HALF_WID = FIELD_WIDTH / 2
GOAL_WIDTH = 1000
GOAL_HW = GOAL_WIDTH / 2
GOAL_LINE_OFFSET = 200
DEFENSE_DEPTH = 1200
DEFENSE_HALF_WIDTH = 1200

# Speeds — full speed
SAVE_SPEED = 2.5
POSITION_SPEED = 1.8
CLEAR_SPEED = 1.4
RETREAT_SPEED = 2.0
KICK_DIST = 175
STOP_RADIUS = 30
RAMP_DIST = 220

# Shot detection
SHOT_SPEED_THRESH = 500     # mm/s
BALL_HISTORY_SIZE = 7
FRICTION = 0.4

# Angle narrowing
MAX_ADVANCE = 1100
DANGER_ZONE = HALF_LEN


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class CalculateGoalieAction(py_trees.behaviour.Behaviour):
    """Elite goalie: shot detection, wall-bounce saves, smart clearing, aggressive positioning."""

    def __init__(self):
        super().__init__("CalculateGoalieAction")
        self.bb = py_trees.blackboard.Client(name=self.name)
        self.ball_history = []
        self.last_ball_xy = None
        self.smooth_x = None
        self.smooth_y = None
        self.last_kick_time = 0.0

    def setup(self, **kwargs):
        self.bb.register_key(key="robot_pos", access=py_trees.common.Access.READ)
        self.bb.register_key(key="ball_pos", access=py_trees.common.Access.READ)
        self.bb.register_key(key="isPositive", access=py_trees.common.Access.READ)
        self.bb.register_key(key="vx", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="vy", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="w", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="kick", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="dribble", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        robot_pos = self.bb.robot_pos
        ball_pos = self.bb.ball_pos
        is_positive = self.bb.isPositive

        if robot_pos is None or ball_pos is None:
            self.bb.vx = self.bb.vy = self.bb.w = 0.0
            self.bb.kick = self.bb.dribble = 0
            return py_trees.common.Status.FAILURE

        rpos = (float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]))
        ball = (float(ball_pos[0]), float(ball_pos[1]))
        now = time.time()

        sign = 1 if is_positive else -1
        goal_x = sign * (FIELD_LENGTH / 2 - GOAL_LINE_OFFSET)
        goal_back_x = sign * (FIELD_LENGTH / 2)
        half_goal = GOAL_HW

        # ── Ball velocity tracking ────────────────────────────
        if (self.last_ball_xy is None or
                ball[0] != self.last_ball_xy[0] or ball[1] != self.last_ball_xy[1]):
            self.ball_history.append((now, ball[0], ball[1]))
            if len(self.ball_history) > BALL_HISTORY_SIZE:
                self.ball_history.pop(0)
            self.last_ball_xy = ball

        bvx, bvy, ball_speed = 0.0, 0.0, 0.0
        if len(self.ball_history) >= 2:
            dt = self.ball_history[-1][0] - self.ball_history[0][0]
            if dt > 0.02:
                bvx = (self.ball_history[-1][1] - self.ball_history[0][1]) / dt
                bvy = (self.ball_history[-1][2] - self.ball_history[0][2]) / dt
                ball_speed = math.hypot(bvx, bvy)

        # ── Shot detection (direct + wall bounce) ─────────────
        ball_toward_us = (bvx * sign > 80)
        shot_incoming = False
        pred_y = 0.0

        if ball_toward_us and ball_speed > SHOT_SPEED_THRESH:
            # Direct shot
            if abs(bvx) > 40:
                t_cross = (goal_back_x - ball[0]) / bvx
                if 0 < t_cross < 2.0:
                    pred_y_raw = ball[1] + bvy * t_cross
                    if abs(pred_y_raw) < half_goal + 250:
                        shot_incoming = True
                        pred_y = _clamp(pred_y_raw, -half_goal, half_goal)

            # Wall-bounce prediction
            if not shot_incoming and ball_speed > SHOT_SPEED_THRESH * 1.2:
                bx, by = ball[0], ball[1]
                vx, vy = bvx, bvy
                t = 0.0
                while t < 2.0:
                    bx += vx * 0.01
                    by += vy * 0.01
                    if by > HALF_WID:
                        by = 2 * HALF_WID - by
                        vy = -vy
                    elif by < -HALF_WID:
                        by = -2 * HALF_WID - by
                        vy = -vy
                    f = max(1.0 - FRICTION * 0.01, 0.0)
                    vx *= f
                    vy *= f
                    t += 0.01
                    if math.hypot(vx, vy) < 40:
                        break
                    crossed = (sign > 0 and bx >= goal_back_x) or (sign < 0 and bx <= goal_back_x)
                    if crossed:
                        if abs(by) < half_goal + 200:
                            shot_incoming = True
                            pred_y = _clamp(by, -half_goal, half_goal)
                        break

        # ── Distances ─────────────────────────────────────────
        rel_ball = world2robot(rpos, ball)
        d_ball = math.hypot(rel_ball[0], rel_ball[1])
        ball_dist = abs(ball[0] - goal_back_x)

        ball_in_box = (ball_dist < DEFENSE_DEPTH and abs(ball[1]) < DEFENSE_HALF_WIDTH)
        ball_slow = ball_speed < 450

        kick, dribble = 0, 0

        # ── Compute target ────────────────────────────────────
        if ball_in_box and ball_slow and d_ball < 1100:
            # CLEAR: slow ball in our box
            raw_target = ball
            speed = CLEAR_SPEED
            smooth_alpha = 0.75

            if d_ball < KICK_DIST and rel_ball[0] > 0:
                # Kick toward sideline
                outward = -1.0 if goal_back_x > 0 else 1.0
                side_y = HALF_WID if ball[1] > 0 else -HALF_WID
                clear_pt = (ball[0] + outward * 1500, side_y)
                rel_clear = world2robot(rpos, clear_pt)
                ang_clear = math.atan2(rel_clear[1], rel_clear[0])
                if abs(ang_clear) < 0.45 and (now - self.last_kick_time) > 0.3:
                    kick = 1
                    self.last_kick_time = now
                else:
                    dribble = 1
            elif d_ball < 350:
                dribble = 1

        elif shot_incoming:
            # SAVE: move to predicted crossing point FAST
            save_x = goal_x + (ball[0] - goal_x) * 0.08
            raw_target = (save_x, pred_y)
            speed = SAVE_SPEED
            smooth_alpha = 0.93

        else:
            # POSITION: narrow the shooting angle aggressively
            gm_x = ball[0] - goal_x
            gm_y = ball[1]
            gm_d = math.hypot(gm_x, gm_y)

            if gm_d > 1:
                advance_ratio = 1.0 - _clamp(ball_dist / (FIELD_LENGTH * 0.5), 0, 1)
                if ball_dist < DANGER_ZONE:
                    advance_ratio = min(1.0, advance_ratio * 1.5)
                advance = advance_ratio * MAX_ADVANCE

                target_x = goal_x + (gm_x / gm_d) * advance
                target_y = (gm_y / gm_d) * advance
                target_y = _clamp(target_y, -(half_goal + 200), half_goal + 200)
                raw_target = (target_x, target_y)
            else:
                raw_target = (goal_x, 0)

            if ball_dist < DANGER_ZONE:
                speed = RETREAT_SPEED
            else:
                speed = POSITION_SPEED
            smooth_alpha = 0.22

        # ── Smooth target ─────────────────────────────────────
        if self.smooth_x is None:
            self.smooth_x = raw_target[0]
            self.smooth_y = raw_target[1]
        else:
            self.smooth_x += smooth_alpha * (raw_target[0] - self.smooth_x)
            self.smooth_y += smooth_alpha * (raw_target[1] - self.smooth_y)
        target = (self.smooth_x, self.smooth_y)

        rel_target = world2robot(rpos, target)
        d_target = math.hypot(rel_target[0], rel_target[1])

        if d_target < 20:
            vx, vy = 0.0, 0.0
        else:
            vx, vy = move_toward_relative(
                rel_target, speed,
                stop_radius=STOP_RADIUS, ramp_dist=RAMP_DIST,
            )
            if d_target < 100:
                scale = d_target / 100.0
                vx *= scale
                vy *= scale

        # Face ball always
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])
        if abs(ang_ball) < 0.05:
            w = 0.0
        else:
            w = _clamp(ang_ball * FACE_BALL_GAIN, -MAX_W, MAX_W)

        self.bb.vx = vx
        self.bb.vy = vy
        self.bb.w = _clamp(w, -MAX_W, MAX_W)
        self.bb.kick = kick
        self.bb.dribble = dribble
        return py_trees.common.Status.SUCCESS


class GoalieRunningSeq(py_trees.composites.Sequence):
    def __init__(self, wm, dispatch_q, goalie_id, isYellow=True, isPositive=None):
        name = f"GoalieRunningSeq (RobotID:{goalie_id})"
        super().__init__(name, memory=True)
        self.wm = wm
        self.dispatch_q = dispatch_q
        self.robot_id = goalie_id
        self.isYellow = isYellow
        self.isPositive = isPositive if isPositive is not None else isYellow
        self.bb = py_trees.blackboard.Client(name=name)
        self.add_children([
            GetWorldPositionUpdate(wm),
            GetRobotIDPosition(goalie_id),
            CalculateGoalieAction(),
            SendRobotCommand(dispatch_q),
        ])

    def setup(self, **kwargs):
        self.bb.register_key(key="robot_id", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="isYellow", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="isPositive", access=py_trees.common.Access.WRITE)
        self.bb.robot_id = self.robot_id
        self.bb.isYellow = self.isYellow
        self.bb.isPositive = self.isPositive
        super().setup(**kwargs)
