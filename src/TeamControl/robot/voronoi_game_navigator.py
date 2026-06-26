"""Production game navigator built on the Voronoi/Dijkstra planner.

`voronoi_navigator.py` is intentionally a bare integrator kept for testing
the planner in isolation (see docs/voronoi-navigator-stripped.md). This file
is the "match"-mode navigator: same PlannerAPI/PlannerInput loop, plus the
behaviours a real game needs:

- Penalty-box guard       — non-goalies don't path into their own box.
- Possession/steal        — temporarily ignore a possessing opponent so we
                             can challenge them (helpers ported from
                             voronoi_navigator.py, now actually wired in).
- Out-of-bounds clearance — when the ball leaves the field, back off to a
                             point >= OUT_OF_BOUNDS_CLEARANCE_MM from both
                             the ball's exit spot and the boundary line it
                             crossed (see compute_out_of_bounds_clearance
                             in ball_nav.py), instead of just stopping.
- Precision approach mode — slow to a tighter, gentler ramp near the target
                             when the planner flags endpoint_precision_mode.
- Exponential smoothing   — low-pass filter on the output command so it
                             doesn't jump frame-to-frame.
- sanitize_field_target   — defense-in-depth clamp around the one place this
                             navigator computes a movement target itself
                             without going through the planner (the
                             outside-of-field override below). The planner's
                             own clamp_to_field already covers the normal
                             path; this is just a backstop.
"""

from __future__ import annotations

import math
import time

from TeamControl.cache import TickCache
from TeamControl.network.robot_command import RobotCommand
from TeamControl.planner import PlannerAPI, PlannerInput
from TeamControl.planner.voronoi_dijkstra import is_in_penalty_box
from TeamControl.robot.ball_nav import (
    clamp,
    compute_out_of_bounds_clearance,
    rotation_compensate,
    sanitize_field_target,
)
from TeamControl.robot.constants import (
    CRUISE_SPEED,
    FACE_TARGET_ANGLE_RAD,
    FACE_TARGET_DIST_MM,
    LOOP_RATE,
    MAX_W,
    TURN_GAIN,
)
from TeamControl.robot.motion.controller import get_motion_controller
from TeamControl.world.field_config import (
    DEFENCE_X_MM,
    FIELD_X_MIN,
    FIELD_X_MAX,
    FIELD_Y_MIN,
    FIELD_Y_MAX,
    ROBOT_RADIUS_MM,
    VORONOI_CHASE_SPEED_SCALE,
    VORONOI_DENSITY_PERCENT,
    VORONOI_FIELD_TARGET_MARGIN_MM,
    VORONOI_HORIZON_MS,
    VORONOI_MAX_DENSITY_NODES,
    VORONOI_POSSESSION_ANGLE_RAD,
    VORONOI_POSSESSION_DIST_MM,
    VORONOI_PRECISION_MIN_SPEED,
    VORONOI_PRECISION_RAMP_DIST_MM,
    VORONOI_PRECISION_SPEED_SCALE,
    VORONOI_SMOOTH_ALPHA,
    VORONOI_STEAL_FRONT_ANGLE_RAD,
    VORONOI_STEAL_FRONT_DIST_MM,
    VORONOI_TARGET_OFFSET_MM,
    VORONOI_WAYPOINT_REACHED_MM,
)
from TeamControl.world.transform_cords import world2robot


CHASE_SPEED = CRUISE_SPEED * VORONOI_CHASE_SPEED_SCALE
PRECISION_APPROACH_SPEED = CRUISE_SPEED * VORONOI_PRECISION_SPEED_SCALE
WAYPOINT_REACHED_MM = VORONOI_WAYPOINT_REACHED_MM

# How far to back off from the ball's exit point/the boundary line it
# crossed once the ball has gone out of bounds. See compute_out_of_bounds_
# clearance() in ball_nav.py -- this single distance keeps us clear of both.
OUT_OF_BOUNDS_CLEARANCE_MM = 500.0

# How long to hold the clearance distance after an opponent gains
# possession before switching to an active steal attempt.
STEAL_DELAY_S = 1.0


def run_voronoi_game_navigator(
    is_running,
    dispatch_q,
    wm,
    robot_id,
    is_yellow,
    planner_path_q=None,
    is_goalie: bool = False,
):
    """Chase the ball using Voronoi/Dijkstra waypoints, with match behaviours."""
    cache = TickCache(wm)
    planner = PlannerAPI(
        density_percent=VORONOI_DENSITY_PERCENT,
        max_density_nodes=VORONOI_MAX_DENSITY_NODES,
    )
    motion_ctrl = get_motion_controller(robot_id, is_yellow)
    active_target = None

    # Exponential-smoothed output velocities — never jump between ticks.
    sm_vx, sm_vy, sm_w = 0.0, 0.0, 0.0

    # When an opponent first gains possession in front of us: timestamp,
    # cleared once they no longer qualify (lost the ball, we moved away, ...).
    possession_since = None

    while is_running.is_set():
        now = time.time()
        if not cache.refresh(now):
            time.sleep(LOOP_RATE)
            continue

        rpos = cache.robots.get_position(is_yellow, robot_id)
        if rpos is None:
            time.sleep(LOOP_RATE)
            continue

        if not cache.ball.visible:
            # Ball out of bounds: back off to a point that's >= the
            # clearance distance from both the ball's exit spot and the
            # line it crossed, instead of just freezing in place.
            if getattr(wm, "last_ball_rejection_reason", None) == "out_of_bounds":
                exit_pos = getattr(wm, "possible_ball_left_field_pos_mm", None)
                if exit_pos is not None:
                    _drive_to_clearance(
                        motion_ctrl, dispatch_q, rpos, exit_pos,
                        robot_id, is_yellow,
                    )
                    time.sleep(LOOP_RATE)
                    continue
            _send_stop(dispatch_q, robot_id, is_yellow)
            time.sleep(LOOP_RATE)
            continue

        ball = cache.ball.position
        ignore_robots = ((bool(is_yellow), int(robot_id)),)

        reached = (
            active_target is not None
            and math.hypot(active_target[0] - rpos[0], active_target[1] - rpos[1])
            <= WAYPOINT_REACHED_MM
        )

        # ── Penalty-box guard: non-goalies don't path into their own box ──
        nav_ball = ball
        if not is_goalie:
            nav_ball = _clamp_out_of_own_box(ball, is_yellow, cache)

        # ── Possession/steal: temporarily ignore a possessing opponent ───
        steal_keys = _steal_ignore_keys(
            cache,
            is_yellow=is_yellow,
            robot_id=robot_id,
            robot_pose=rpos,
            ball_pos=ball,
        )

        try:
            obstacles = wm.get_planning_obstacles(
                now_s=now,
                horizon_ms=VORONOI_HORIZON_MS,
                ignore_robots=ignore_robots,
            )
            plan = planner.plan(PlannerInput(
                robot_id=robot_id,
                is_yellow=is_yellow,
                current_pose=(float(rpos[0]), float(rpos[1]), float(rpos[2])),
                target_pose=(float(nav_ball[0]), float(nav_ball[1]), 0.0),
                obstacles=obstacles,
                clearance_mm=0.0,
                robot_reached_current_waypoint=reached,
                now_s=now,
                ignored_obstacle_keys_containing_target=steal_keys,
            ))
        except Exception:
            plan = None

        if not plan:
            _send_stop(dispatch_q, robot_id, is_yellow)
            time.sleep(LOOP_RATE)
            continue

        active_target = plan.active_target_pose
        _publish_planned_path(
            planner_path_q,
            robot_id=robot_id,
            is_yellow=is_yellow,
            robot_pose=rpos,
            plan=plan,
            now_s=now,
        )

        rx, ry = float(rpos[0]), float(rpos[1])
        x_min, x_max, y_min, y_max = float(FIELD_X_MIN), float(FIELD_X_MAX), float(FIELD_Y_MIN), float(FIELD_Y_MAX)
        outside_field = (
            rx < x_min or rx > x_max
            or ry < y_min or ry > y_max
        )

        # If outside the field, drive to the nearest point that is still
        # VORONOI_FIELD_TARGET_MARGIN_MM inside the boundary.
        # translational_motion applies the out-of-field speed scale internally.
        _m = VORONOI_FIELD_TARGET_MARGIN_MM
        movement_target = (
            (max(x_min + _m, min(x_max - _m, rx)),
             max(y_min + _m, min(y_max - _m, ry)))
            if outside_field else active_target
        )
        # Defense-in-depth: the planner already clamps to the field via
        # clamp_to_field(), but the branch above computes a target without
        # going through the planner at all — sanitize it too.
        movement_target = sanitize_field_target(
            movement_target, margin=VORONOI_FIELD_TARGET_MARGIN_MM
        )

        rel_ball = world2robot(rpos, ball)

        dist_to_move = math.hypot(
            movement_target[0] - rpos[0], movement_target[1] - rpos[1]
        )

        # ── Precision approach mode: tighter, gentler ramp near target ──
        if plan.endpoint_precision_mode:
            if dist_to_move < VORONOI_PRECISION_RAMP_DIST_MM:
                t = dist_to_move / max(VORONOI_PRECISION_RAMP_DIST_MM, 1.0)
                approach_speed = max(
                    PRECISION_APPROACH_SPEED * t, VORONOI_PRECISION_MIN_SPEED
                )
            else:
                approach_speed = PRECISION_APPROACH_SPEED
        else:
            approach_speed = CHASE_SPEED

        deadline = time.monotonic() + max(
            dist_to_move / 1000.0 / max(approach_speed, 0.01), 0.1
        )
        nav_vx, nav_vy = motion_ctrl.translational_motion(
            rpos, movement_target, deadline, stay_in_field=True
        )

        dist_to_ball = math.hypot(rel_ball[0], rel_ball[1])
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])

        # Stop once within VORONOI_TARGET_OFFSET_MM of the ball.
        if dist_to_ball < VORONOI_TARGET_OFFSET_MM:
            nav_vx, nav_vy = 0.0, 0.0

        # Face the ball before moving when within dribble range.
        if dist_to_ball < FACE_TARGET_DIST_MM and abs(ang_ball) > FACE_TARGET_ANGLE_RAD:
            nav_vx, nav_vy = 0.0, 0.0

        w = 0.0 if abs(ang_ball) < 0.05 else clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)

        # ── Exponential smoothing ────────────────────────────────────────
        a = VORONOI_SMOOTH_ALPHA
        sm_vx = a * sm_vx + (1.0 - a) * nav_vx
        sm_vy = a * sm_vy + (1.0 - a) * nav_vy
        sm_w = a * sm_w + (1.0 - a) * w

        out_vx, out_vy = rotation_compensate(sm_vx, sm_vy, sm_w)
        dispatch_q.put((
            RobotCommand(
                robot_id=robot_id,
                vx=out_vx,
                vy=out_vy,
                w=sm_w,
                kick=0,
                dribble=0,
                isYellow=is_yellow,
            ),
            0.15,
        ))
        time.sleep(LOOP_RATE)


def _clamp_out_of_own_box(ball, is_yellow, cache):
    """Push a target out of our own penalty box, along the goal axis only.

    Used so non-goalie robots don't path straight into the box when the ball
    rolls in there — they line up just outside it instead. Reuses the
    planner's own `is_in_penalty_box` geometry rather than duplicating it.
    """
    bx, by = float(ball[0]), float(ball[1])
    own_goal_x = cache.team.goal_x(is_yellow)
    positive_side = own_goal_x > 0
    if not is_in_penalty_box((bx, by), positive_side=positive_side, margin=ROBOT_RADIUS_MM):
        return (bx, by)

    defence_x = float(DEFENCE_X_MM)
    x_min, x_max = float(FIELD_X_MIN), float(FIELD_X_MAX)
    if positive_side:
        bx = min(bx, x_max - defence_x - ROBOT_RADIUS_MM)
    else:
        bx = max(bx, x_min + defence_x + ROBOT_RADIUS_MM)
    return (bx, by)


def _publish_planned_path(
    planner_path_q,
    *,
    robot_id: int,
    is_yellow: bool,
    robot_pose,
    plan,
    now_s: float,
) -> None:
    if planner_path_q is None:
        return
    points = ()
    if not plan.is_path_free and plan.waypoints:
        points = (
            (float(robot_pose[0]), float(robot_pose[1])),
            *((float(p[0]), float(p[1])) for p in plan.waypoints),
        )
    try:
        planner_path_q.put_nowait({
            "robot_id": int(robot_id),
            "is_yellow": bool(is_yellow),
            "points": points,
            "timestamp_s": float(now_s),
            "is_path_free": bool(plan.is_path_free),
            "need_reroute": bool(plan.need_reroute),
            "did_reroute": bool(plan.did_reroute),
        })
    except Exception:
        pass


def _send_stop(dispatch_q, robot_id: int, is_yellow: bool) -> None:
    dispatch_q.put((
        RobotCommand(
            robot_id=robot_id,
            vx=0.0,
            vy=0.0,
            w=0.0,
            kick=0,
            dribble=0,
            isYellow=is_yellow,
        ),
        0.15,
    ))


def _drive_to_clearance(motion_ctrl, dispatch_q, rpos, exit_pos, robot_id, is_yellow) -> None:
    """Back off to >= OUT_OF_BOUNDS_CLEARANCE_MM from the ball's exit point
    and the boundary line it crossed, instead of stopping in place."""
    target = compute_out_of_bounds_clearance(exit_pos, OUT_OF_BOUNDS_CLEARANCE_MM)
    dist = math.hypot(target[0] - rpos[0], target[1] - rpos[1])
    deadline = time.monotonic() + max(dist / 1000.0 / max(CHASE_SPEED, 0.01), 0.1)
    vx, vy = motion_ctrl.translational_motion(rpos, target, deadline, stay_in_field=True)
    dispatch_q.put((
        RobotCommand(
            robot_id=robot_id,
            vx=vx,
            vy=vy,
            w=0.0,
            kick=0,
            dribble=0,
            isYellow=is_yellow,
        ),
        0.15,
    ))


def _robot_is_in_front_of_possessor(robot_pose, possessor_pose) -> bool:
    rel = world2robot(possessor_pose, robot_pose[:2])
    dist = math.hypot(rel[0], rel[1])
    angle = math.atan2(rel[1], rel[0])
    return (
        rel[0] > 0.0
        and dist <= VORONOI_STEAL_FRONT_DIST_MM
        and abs(angle) <= VORONOI_STEAL_FRONT_ANGLE_RAD
    )


def _steal_ignore_keys(
    cache,
    *,
    is_yellow: bool,
    robot_id: int,
    robot_pose,
    ball_pos,
):
    keys = []
    opponent_is_yellow = not bool(is_yellow)
    for opponent_id, opponent_pose in cache.robots.iter_team(opponent_is_yellow):
        if opponent_is_yellow == bool(is_yellow) and int(opponent_id) == int(robot_id):
            continue
        _, dist_to_ball, angle_to_ball = cache.robots.relative_to_ball(
            opponent_is_yellow,
            opponent_id,
            ball_pos,
        )
        if (
            dist_to_ball < VORONOI_POSSESSION_DIST_MM
            and abs(angle_to_ball) <= VORONOI_POSSESSION_ANGLE_RAD
            and _robot_is_in_front_of_possessor(robot_pose, opponent_pose)
        ):
            keys.append((opponent_is_yellow, int(opponent_id)))
    return tuple(keys)
