import time

from TeamControl.planner import PlannerAPI, PlannerInput, plan
from TeamControl.world.map.obstacles import Obstacle


def test_planner_api_returns_planner_output():
    api = PlannerAPI()

    output = api.plan(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(0.0, 0.0, 0.0),
            target_pose=(1000.0, 0.0, 0.0),
        )
    )

    assert output.is_path_free is True
    assert output.active_target_pose == (1000.0, 0.0, 0.0)


def test_plan_helper_accepts_one_shot_input():
    output = plan(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(0.0, 0.0),
            target_pose=(1000.0, 0.0),
        )
    )

    assert output.active_target_pose == (1000.0, 0.0, 0.0)


def test_planner_api_keeps_pressing_possessed_target_in_precision_mode():
    api = PlannerAPI(density_percent=60, max_density_nodes=120)

    output = api.plan(
        PlannerInput(
            robot_id=0,
            is_yellow=True,
            current_pose=(-1000.0, 0.0),
            target_pose=(0.0, 0.0),
            obstacles=((0.0, 0.0, 180.0),),
        )
    )

    assert output.is_path_free is True
    assert output.endpoint_was_adjusted is True
    assert output.endpoint_precision_mode is True
    assert output.waypoints == ()
    assert output.active_target_pose == (-255.0, 0.0, 0.0)


def test_planner_api_checks_safety_and_reach_clearance_for_target():
    api = PlannerAPI()
    now_s = time.time()
    obstacles = (
        Obstacle(
            timestamp=now_s,
            robot_id=1,
            isYellow=False,
            pos_mm=(0.0, 0.0, 0.0),
            received_at_s=now_s,
        ),
    )

    outside = api.check_target_clearance((180.0, 0.0), obstacles)
    safety_only = api.check_target_clearance((150.0, 0.0), obstacles)
    close_reach = api.check_target_clearance((130.0, 0.0), obstacles)

    assert outside.in_safety_clearance is False
    assert outside.in_reach_clearance is False

    assert safety_only.in_safety_clearance is True
    assert safety_only.in_reach_clearance is False
    assert safety_only.safety_clearance_overlap_mm == 20.0

    assert close_reach.in_safety_clearance is True
    assert close_reach.in_reach_clearance is True
    assert close_reach.nearest_obstacle_key == (False, 1)
    assert close_reach.safety_clearance_radius_mm == 170.0
    assert close_reach.reach_clearance_radius_mm == 140.0
