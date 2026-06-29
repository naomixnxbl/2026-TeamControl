import time

from TeamControl.planner import PlannerInput, VoronoiWaypointManager
from TeamControl.world.field_config import FIELD_X_MAX, VORONOI_FIELD_TARGET_MARGIN_MM
from TeamControl.world.map.obstacles import Obstacle
from TeamControl.world.map.world_map import WorldMap


def test_waypoint_manager_returns_original_target_when_direct_path_is_free():
    manager = VoronoiWaypointManager()

    output = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(0.0, 0.0, 0.0),
            target_pose=(1000.0, 0.0, 0.2),
        )
    )

    assert output.is_path_free is True
    assert output.need_reroute is False
    assert output.did_reroute is False
    assert output.waypoints == ()
    assert output.current_waypoint_index == 0
    assert output.active_target_pose == (1000.0, 0.0, 0.2)


def test_waypoint_manager_allows_penalty_box_target_for_now():
    manager = VoronoiWaypointManager()

    output = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2500.0, 0.0, 0.0),
            target_pose=(-4000.0, 0.0, 0.2),
        )
    )

    assert output.is_path_free is True
    assert output.active_target_pose == (-4000.0, 0.0, 0.2)


def test_waypoint_manager_reroutes_when_direct_path_is_blocked():
    now_s = time.time()
    world_map = WorldMap()
    world_map.obs = [
        Obstacle(
            timestamp=now_s,
            robot_id=1,
            isYellow=False,
            pos_mm=(0.0, 0.0, 0.0),
            received_at_s=now_s,
        )
    ]
    manager = VoronoiWaypointManager(density_percent=60, max_density_nodes=120)

    output = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2000.0, 0.0, 0.0),
            target_pose=(2000.0, 0.0, 0.5),
            clearance_mm=0.0,
            world_map=world_map,
            now_s=now_s,
        )
    )

    assert output.is_path_free is False
    assert output.need_reroute is True
    assert output.did_reroute is True
    assert output.waypoints
    assert output.active_target_pose == output.waypoints[0]


def test_waypoint_manager_pops_current_waypoint_when_pd_reaches_it():
    now_s = time.time()
    world_map = WorldMap()
    world_map.obs = [
        Obstacle(
            timestamp=now_s,
            robot_id=1,
            isYellow=False,
            pos_mm=(0.0, 0.0, 0.0),
            received_at_s=now_s,
        )
    ]
    manager = VoronoiWaypointManager(density_percent=60, max_density_nodes=120)
    first = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2000.0, 0.0, 0.0),
            target_pose=(2000.0, 0.0, 0.5),
            world_map=world_map,
            now_s=now_s,
        )
    )

    second = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2000.0, 0.0, 0.0),
            target_pose=(2000.0, 0.0, 0.5),
            robot_reached_current_waypoint=True,
            world_map=world_map,
            now_s=now_s,
        )
    )

    assert second.current_waypoint_index == 0
    assert second.waypoints == first.waypoints[1:]
    assert second.active_target_pose == second.waypoints[0]


def test_waypoint_manager_clears_waypoints_when_direct_path_reopens_then_replans():
    manager = VoronoiWaypointManager(density_percent=60, max_density_nodes=120)
    obstacle = ((0.0, 0.0, 120.0),)

    blocked = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2000.0, 0.0),
            target_pose=(2000.0, 0.0),
            obstacles=obstacle,
        )
    )
    direct = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2000.0, 0.0),
            target_pose=(2000.0, 0.0),
            obstacles=(),
        )
    )
    blocked_again = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2000.0, 0.0),
            target_pose=(2000.0, 0.0),
            obstacles=obstacle,
        )
    )

    assert blocked.is_path_free is False
    assert blocked.waypoints
    assert direct.is_path_free is True
    assert direct.waypoints == ()
    assert direct.active_target_pose == (2000.0, 0.0, 0.0)
    assert blocked_again.is_path_free is False
    assert blocked_again.did_reroute is True
    assert blocked_again.waypoints


def test_waypoint_manager_accepts_explicit_tuple_obstacles():
    manager = VoronoiWaypointManager(density_percent=60, max_density_nodes=120)

    output = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-2000.0, 0.0),
            target_pose=(2000.0, 0.0),
            obstacles=((0.0, 0.0, 120.0),),
        )
    )

    assert output.is_path_free is False
    assert output.did_reroute is True
    assert output.waypoints


def test_waypoint_manager_offsets_target_out_of_clearance_zone():
    manager = VoronoiWaypointManager(density_percent=60, max_density_nodes=120)

    output = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-1000.0, 0.0),
            target_pose=(-10.0, 0.0, 0.4),
            obstacles=((0.0, 0.0, 180.0),),
        )
    )

    assert output.is_path_free is True
    assert output.endpoint_was_adjusted is True
    assert output.endpoint_precision_mode is True
    assert output.waypoints == ()
    assert output.active_target_pose == (-275.0, 0.0, 0.4)


def test_waypoint_manager_uses_precision_mode_when_offset_stays_in_clearance():
    manager = VoronoiWaypointManager(density_percent=10, max_density_nodes=20)

    output = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(1000.0, 0.0),
            target_pose=(0.0, 0.0, 0.4),
            obstacles=((0.0, 0.0, 10000.0),),
        )
    )

    assert output.is_path_free is False
    assert output.endpoint_was_adjusted is True
    assert output.endpoint_precision_mode is True
    # endpoint is clamped to the field-target margin inset, not the raw boundary
    assert output.active_target_pose == (FIELD_X_MAX - VORONOI_FIELD_TARGET_MARGIN_MM, 0.0, 0.4)


def test_waypoint_manager_can_ignore_obstacle_that_contains_target_for_steal():
    manager = VoronoiWaypointManager(density_percent=60, max_density_nodes=120)

    blocked = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-1000.0, 0.0),
            target_pose=(0.0, 0.0),
            obstacles=((0.0, 0.0, 180.0),),
        )
    )
    manager.reset()
    steal = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-1000.0, 0.0),
            target_pose=(0.0, 0.0),
            obstacles=((0.0, 0.0, 180.0),),
            ignore_obstacles_containing_target=True,
        )
    )

    assert blocked.is_path_free is True
    assert blocked.endpoint_was_adjusted is True
    assert blocked.endpoint_precision_mode is True
    assert blocked.waypoints == ()
    assert blocked.active_target_pose == (-275.0, 0.0, 0.0)
    assert steal.is_path_free is True
    assert steal.endpoint_was_adjusted is False
    assert steal.active_target_pose == (0.0, 0.0, 0.0)


def test_waypoint_manager_steal_ignore_can_be_limited_to_specific_obstacle_key():
    now_s = time.time()
    manager = VoronoiWaypointManager(density_percent=60, max_density_nodes=120)
    obstacles = (
        Obstacle(
            timestamp=now_s,
            robot_id=1,
            isYellow=False,
            pos_mm=(0.0, 0.0, 0.0),
            received_at_s=now_s,
        ),
    )

    not_allowed = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-1000.0, 0.0),
            target_pose=(0.0, 0.0),
            obstacles=obstacles,
            ignored_obstacle_keys_containing_target=((True, 1),),
        )
    )
    manager.reset()
    allowed = manager.update(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-1000.0, 0.0),
            target_pose=(0.0, 0.0),
            obstacles=obstacles,
            ignored_obstacle_keys_containing_target=((False, 1),),
        )
    )

    assert not_allowed.is_path_free is True
    assert not_allowed.endpoint_was_adjusted is True
    assert not_allowed.endpoint_precision_mode is True
    assert not_allowed.waypoints == ()
    assert not_allowed.active_target_pose == (-215.0, 0.0, 0.0)
    assert allowed.is_path_free is True
    assert allowed.endpoint_was_adjusted is False
    assert allowed.active_target_pose == (0.0, 0.0, 0.0)
