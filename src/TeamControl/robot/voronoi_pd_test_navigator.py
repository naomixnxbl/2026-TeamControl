"""Planner + PD integration test -- single robot, RobotMotionController.

RobotMotionController (motion/controller.py) has no live gameplay caller --
only the PD calibration harness exercises it. This file is the live test
path for its rule set (field-size cache, stay_in_field/penalty-box/goal-
post, never-overshoot regulation, angular wheel-budget cap, ...): same
PlannerAPI/PlannerInput/TickCache loop shape as voronoi_navigator.py, but
driven by `get_motion_controller(...)` + `rotational_motion`/
`translational_motion(..., stay_in_field=True)` (Option A sequential, the
documented default) instead of ball_nav. Single robot only -- this is a
test path, not the production game navigator (that's voronoi_game_navigator.py).
"""

from __future__ import annotations

import math
import time

from TeamControl.cache import TickCache
from TeamControl.network.robot_command import RobotCommand
from TeamControl.planner import PlannerAPI, PlannerInput
from TeamControl.robot.constants import LOOP_RATE
from TeamControl.robot.motion import get_motion_controller
from TeamControl.world.field_config import (
    FIELD_X_MIN,
    FIELD_X_MAX,
    FIELD_Y_MIN,
    FIELD_Y_MAX,
    VORONOI_DENSITY_PERCENT,
    VORONOI_FIELD_TARGET_MARGIN_MM,
    VORONOI_HORIZON_MS,
    VORONOI_MAX_DENSITY_NODES,
    VORONOI_TARGET_OFFSET_MM,
    VORONOI_WAYPOINT_REACHED_MM,
)
from TeamControl.world.transform_cords import world2robot


def run_pd_planner_test(
    is_running,
    dispatch_q,
    wm,
    robot_id,
    is_yellow,
    planner_path_q=None,
):
    """Chase the ball using Voronoi/Dijkstra waypoints, driven by PD."""
    cache = TickCache(wm)
    planner = PlannerAPI(
        density_percent=VORONOI_DENSITY_PERCENT,
        max_density_nodes=VORONOI_MAX_DENSITY_NODES,
    )
    motion = get_motion_controller(robot_id, is_yellow)
    active_target = None

    while is_running.is_set():
        now = time.time()
        if not cache.refresh(now):
            time.sleep(LOOP_RATE)
            continue
        if not cache.ball.visible:
            _send_stop(dispatch_q, robot_id, is_yellow)
            time.sleep(LOOP_RATE)
            continue

        rpos = cache.robots.get_position(is_yellow, robot_id)
        if rpos is None:
            time.sleep(LOOP_RATE)
            continue

        ball = cache.ball.position
        ignore_robots = ((bool(is_yellow), int(robot_id)),)

        reached = (
            active_target is not None
            and math.hypot(active_target[0] - rpos[0], active_target[1] - rpos[1])
            <= VORONOI_WAYPOINT_REACHED_MM
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
                target_pose=(float(ball[0]), float(ball[1]), 0.0),
                obstacles=obstacles,
                clearance_mm=0.0,
                robot_reached_current_waypoint=reached,
                now_s=now,
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
        # translational_motion(stay_in_field=True) handles the rest.
        _m = VORONOI_FIELD_TARGET_MARGIN_MM
        movement_target = (
            (max(x_min + _m, min(x_max - _m, rx)),
             max(y_min + _m, min(y_max - _m, ry)))
            if outside_field else active_target
        )

        rel_ball = world2robot(rpos, ball)
        dist_to_ball = math.hypot(rel_ball[0], rel_ball[1])
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])
        target_theta = rpos[2] + ang_ball  # face the ball

        deadline = time.monotonic() + 0.5

        # Option A sequential (documented default): turn to face the ball
        # first if not roughly facing it, otherwise drive toward the
        # planner's waypoint.
        if not motion.is_facing_dir(rpos[2], target_theta):
            w = motion.rotational_motion(rpos[2], target_theta, deadline)
            vx, vy = 0.0, 0.0
        else:
            w = 0.0
            vx, vy = motion.translational_motion(
                rpos, movement_target, deadline, stay_in_field=True
            )

        # Stop once within VORONOI_TARGET_OFFSET_MM of the ball.
        if dist_to_ball < VORONOI_TARGET_OFFSET_MM:
            vx, vy = 0.0, 0.0

        dispatch_q.put((
            RobotCommand(
                robot_id=robot_id,
                vx=vx,
                vy=vy,
                w=w,
                kick=0,
                dribble=0,
                isYellow=is_yellow,
            ),
            0.15,
        ))
        time.sleep(LOOP_RATE)


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
