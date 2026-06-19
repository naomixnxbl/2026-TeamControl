"""Simple integrator that exercises the Voronoi/Dijkstra planner."""

from __future__ import annotations

import math
import time

from TeamControl.cache import TickCache
from TeamControl.network.robot_command import RobotCommand
from TeamControl.planner import PlannerAPI, PlannerInput
from TeamControl.robot.ball_nav import (
    apply_boundary_braking,
    clamp,
    move_toward,
    rotation_compensate,
)
from TeamControl.robot.constants import (
    CRUISE_SPEED,
    FACE_TARGET_ANGLE_RAD,
    FACE_TARGET_DIST_MM,
    LOOP_RATE,
    MAX_W,
    TURN_GAIN,
)
from TeamControl.world.field_config import (
    FIELD_X_MIN,
    FIELD_X_MAX,
    FIELD_Y_MIN,
    FIELD_Y_MAX,
    VORONOI_CHASE_SPEED_SCALE,
    VORONOI_DENSITY_PERCENT,
    VORONOI_FIELD_TARGET_MARGIN_MM,
    VORONOI_HORIZON_MS,
    VORONOI_MAX_DENSITY_NODES,
    VORONOI_POSSESSION_ANGLE_RAD,
    VORONOI_POSSESSION_DIST_MM,
    VORONOI_STEAL_FRONT_ANGLE_RAD,
    VORONOI_STEAL_FRONT_DIST_MM,
    VORONOI_TARGET_OFFSET_MM,
    VORONOI_WAYPOINT_REACHED_MM,
)
from TeamControl.world.transform_cords import world2robot


CHASE_SPEED = CRUISE_SPEED * VORONOI_CHASE_SPEED_SCALE
WAYPOINT_REACHED_MM = VORONOI_WAYPOINT_REACHED_MM


KICK_DIST_MM = 100.0   # kick when robot centre is within this distance of the ball


def run_voronoi_navigator(
    is_running,
    dispatch_q,
    wm,
    robot_id,
    is_yellow,
    planner_path_q=None,
    kick_at_ball: bool = False,
):
    """Chase the ball using Voronoi/Dijkstra waypoints."""
    cache = TickCache(wm)
    planner = PlannerAPI(
        density_percent=VORONOI_DENSITY_PERCENT,
        max_density_nodes=VORONOI_MAX_DENSITY_NODES,
    )
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
            <= WAYPOINT_REACHED_MM
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
        # apply_boundary_braking() applies the out-of-field speed scale below.
        _m = VORONOI_FIELD_TARGET_MARGIN_MM
        movement_target = (
            (max(x_min + _m, min(x_max - _m, rx)),
             max(y_min + _m, min(y_max - _m, ry)))
            if outside_field else active_target
        )

        rel_ball = world2robot(rpos, ball)

        rel_target = world2robot(rpos, movement_target)
        nav_vx, nav_vy = move_toward(rel_target, CHASE_SPEED)
        nav_vx, nav_vy = apply_boundary_braking(rpos, nav_vx, nav_vy)

        dist_to_ball = math.hypot(rel_ball[0], rel_ball[1])
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])

        kick = 0
        if kick_at_ball and dist_to_ball < KICK_DIST_MM:
            # Close enough — stop driving and fire the kicker.
            nav_vx, nav_vy = 0.0, 0.0
            kick = 1
        else:
            # Stop once within VORONOI_TARGET_OFFSET_MM of the ball.
            if dist_to_ball < VORONOI_TARGET_OFFSET_MM:
                nav_vx, nav_vy = 0.0, 0.0

            # Face the ball before moving when within dribble range.
            if dist_to_ball < FACE_TARGET_DIST_MM and abs(ang_ball) > FACE_TARGET_ANGLE_RAD:
                nav_vx, nav_vy = 0.0, 0.0

        w = 0.0 if abs(ang_ball) < 0.05 else clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)

        out_vx, out_vy = rotation_compensate(nav_vx, nav_vy, w)
        dispatch_q.put((
            RobotCommand(
                robot_id=robot_id,
                vx=out_vx,
                vy=out_vy,
                w=w,
                kick=kick,
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
