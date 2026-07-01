from __future__ import annotations
import math
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import (
    GOAL_HW_M,
    GK_BASE_SPEED, GK_SPEED_SCALE, GK_MAX_SPEED,
    GK_PREDICT_HORIZON_S, GK_STANCE_RATIO,
    attack_sign, own_goal, angle_to,
)


def goalie_intercept(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Sprint to the predicted ball crossing point on our goal line.

    Speed scales with incoming ball speed (faster shot → faster response).
    When no shot is incoming, holds a narrowing stance between ball and goal.
    Side-aware via the global attack direction.
    """
    if robot is None:
        return None
    bx, by     = snap.ball_position
    vx, vy     = snap.ball_velocity
    ball_speed = math.hypot(vx, vy)
    goal_x     = own_goal()[0]
    sign       = attack_sign()

    # Ball travelling toward OUR goal: its x-velocity points in the -sign
    # direction (own goal sits at -sign * HALF_LEN).
    if vx * sign < -0.05 and ball_speed > 0.05:
        t = (goal_x - bx) / vx
        if 0 < t < GK_PREDICT_HORIZON_S:
            pred_y    = max(-GOAL_HW_M, min(GOAL_HW_M, by + vy * t))
            intercept = (goal_x + sign * 0.08, pred_y)
            speed     = min(GK_BASE_SPEED + GK_SPEED_SCALE * ball_speed, GK_MAX_SPEED)
            return IntentMove(target_pos=intercept,
                              target_orientation=angle_to(intercept, snap.ball_position),
                              max_speed=speed)

    dx, dy = bx - goal_x, by
    dist   = math.hypot(dx, dy)
    stance = (goal_x + dx * GK_STANCE_RATIO, dy * GK_STANCE_RATIO) if dist > 1e-6 \
             else (goal_x + sign * 0.2, 0.0)
    return IntentMove(target_pos=stance,
                      target_orientation=angle_to(stance, snap.ball_position))
