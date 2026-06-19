import time

import pytest

from TeamControl.planner.voronoi_dijkstra import (
    PlannerState,
    VoronoiDijkstraPlanner,
    is_in_box,
    is_in_field,
    is_in_penalty_box,
)
from TeamControl.world.field_config import (
    DEFENCE_X_MM,
    DEFENCE_Y_MM,
    FIELD_X_MAX,
    FIELD_X_MIN,
    FIELD_Y_MAX,
    FIELD_Y_MIN,
    VORONOI_FIELD_TARGET_MARGIN_MM,
)
from TeamControl.world.map.obstacles import Obstacle
from TeamControl.world.map.world_map import WorldMap


def test_voronoi_planner_clamps_target_and_uses_direct_path():
    world_map = WorldMap()
    planner = VoronoiDijkstraPlanner()

    result = planner.plan(
        world_map,
        (0.0, 0.0),
        (FIELD_X_MAX + 500.0, FIELD_Y_MIN - 500.0),
    )

    assert result.used_direct_path is True
    assert result.target_mm == (FIELD_X_MAX, FIELD_Y_MIN)
    assert result.waypoints_mm == ()


def test_voronoi_planner_allows_main_field_robot_to_target_penalty_box_for_now():
    world_map = WorldMap()
    planner = VoronoiDijkstraPlanner()

    result = planner.plan(
        world_map,
        (-2500.0, 0.0),
        (-4000.0, 0.0),
    )

    assert result.used_direct_path is True
    assert result.target_mm == (-4000.0, 0.0)


def test_voronoi_planner_allows_penalty_robot_to_target_main_field_for_now():
    world_map = WorldMap()
    planner = VoronoiDijkstraPlanner()

    result = planner.plan(
        world_map,
        (-4000.0, 0.0),
        (0.0, 0.0),
    )

    assert result.used_direct_path is True
    assert result.target_mm == (0.0, 0.0)


def test_voronoi_planner_reuses_valid_previous_path_for_similar_target():
    now_s = time.time()
    world_map = WorldMap()
    world_map.obs = [
        Obstacle(
            timestamp=now_s,
            robot_id=1,
            isYellow=True,
            pos_mm=(0.0, 0.0, 0.0),
            received_at_s=now_s,
        )
    ]
    planner = VoronoiDijkstraPlanner(target_dead_zone_mm=200.0)
    state = PlannerState(
        last_target_mm=(1000.0, 0.0),
        waypoints_mm=((-1000.0, 1000.0), (1000.0, 0.0)),
    )

    result = planner.plan(
        world_map,
        (-2000.0, 0.0),
        (1050.0, 0.0),
        now_s=now_s,
        previous_state=state,
    )

    assert result.used_direct_path is False
    assert result.reused_previous is True
    assert result.waypoints_mm == state.waypoints_mm


def test_voronoi_planner_finds_detour_when_direct_path_is_blocked():
    now_s = time.time()
    world_map = WorldMap()
    world_map.obs = [
        Obstacle(
            timestamp=now_s,
            robot_id=1,
            isYellow=True,
            pos_mm=(0.0, 0.0, 0.0),
            received_at_s=now_s,
        )
    ]
    planner = VoronoiDijkstraPlanner(
        density_percent=60.0,
        max_density_nodes=120,
        connection_count=10,
    )

    result = planner.plan(
        world_map,
        (-2000.0, 0.0),
        (2000.0, 0.0),
        now_s=now_s,
    )

    assert result.used_direct_path is False
    assert result.waypoints_mm
    assert result.waypoints_mm[-1] == result.target_mm


def test_voronoi_planner_returns_escape_waypoint_when_start_inside_obstacle_clearance():
    now_s = time.time()
    world_map = WorldMap()
    world_map.obs = [
        Obstacle(
            timestamp=now_s,
            robot_id=1,
            isYellow=True,
            pos_mm=(0.0, 0.0, 0.0),
            received_at_s=now_s,
        )
    ]
    planner = VoronoiDijkstraPlanner(
        density_percent=60.0,
        max_density_nodes=120,
        connection_count=10,
    )

    result = planner.plan(
        world_map,
        (50.0, 0.0),
        (2000.0, 0.0),
        now_s=now_s,
    )

    assert result.used_direct_path is False
    assert result.waypoints_mm
    assert result.waypoints_mm[0][0] > 210.0


# ---------------------------------------------------------------------------
# is_in_box
# ---------------------------------------------------------------------------

def test_is_in_box_point_clearly_inside():
    assert is_in_box((5.0, 5.0), 0.0, 10.0, 0.0, 10.0) is True


def test_is_in_box_point_outside_on_x():
    assert is_in_box((11.0, 5.0), 0.0, 10.0, 0.0, 10.0) is False


def test_is_in_box_point_outside_on_y():
    assert is_in_box((5.0, -1.0), 0.0, 10.0, 0.0, 10.0) is False


def test_is_in_box_point_on_boundary_is_inside_with_zero_margin():
    # Boundary is inclusive at margin=0.
    assert is_in_box((10.0, 10.0), 0.0, 10.0, 0.0, 10.0) is True
    assert is_in_box((0.0, 0.0), 0.0, 10.0, 0.0, 10.0) is True


def test_is_in_box_point_exactly_at_inset_boundary_is_inside():
    # With margin=1 the effective box is [1, 9] × [1, 9]; x=9 is on the edge.
    assert is_in_box((9.0, 5.0), 0.0, 10.0, 0.0, 10.0, margin=1.0) is True


def test_is_in_box_point_just_past_inset_boundary_is_outside():
    # x=9.5 is beyond the effective x_max of 9.
    assert is_in_box((9.5, 5.0), 0.0, 10.0, 0.0, 10.0, margin=1.0) is False


def test_is_in_box_original_boundary_now_outside_with_margin():
    # The original boundary (10, 5) is outside when margin > 0.
    assert is_in_box((10.0, 5.0), 0.0, 10.0, 0.0, 10.0, margin=1.0) is False


def test_is_in_box_oversized_margin_always_false():
    # margin=6 on a box of width 10 inverts the effective range.
    assert is_in_box((5.0, 5.0), 0.0, 10.0, 0.0, 10.0, margin=6.0) is False


# ---------------------------------------------------------------------------
# is_in_field
# ---------------------------------------------------------------------------

def test_is_in_field_center_is_inside():
    assert is_in_field((0.0, 0.0)) is True


def test_is_in_field_center_with_zero_margin():
    assert is_in_field((0.0, 0.0), margin=0.0) is True


def test_is_in_field_point_outside_field_entirely():
    assert is_in_field((FIELD_X_MAX + 100.0, 0.0)) is False


def test_is_in_field_point_on_field_boundary_with_zero_margin():
    assert is_in_field((FIELD_X_MAX, 0.0), margin=0.0) is True


def test_is_in_field_point_on_boundary_excluded_by_default_margin():
    # FIELD_X_MAX is the wall; default margin=90 means it must be 90 mm inside.
    assert is_in_field((FIELD_X_MAX, 0.0)) is False


def test_is_in_field_point_just_inside_default_margin():
    # Exactly at the inset boundary (FIELD_X_MAX - margin) → still inside.
    x = FIELD_X_MAX - VORONOI_FIELD_TARGET_MARGIN_MM
    assert is_in_field((x, 0.0)) is True


def test_is_in_field_point_one_mm_past_default_margin():
    x = FIELD_X_MAX - VORONOI_FIELD_TARGET_MARGIN_MM + 1.0
    assert is_in_field((x, 0.0)) is False


def test_is_in_field_point_close_to_y_wall_excluded_by_default_margin():
    # Within 50 mm of the y wall — inside the field but within margin.
    y = FIELD_Y_MAX - 50.0
    assert is_in_field((0.0, y)) is False


def test_is_in_field_point_close_to_y_wall_included_with_zero_margin():
    y = FIELD_Y_MAX - 50.0
    assert is_in_field((0.0, y), margin=0.0) is True


# ---------------------------------------------------------------------------
# is_in_penalty_box
# ---------------------------------------------------------------------------

# Positive-side box: x ∈ [FIELD_X_MAX - DEFENCE_X_MM, FIELD_X_MAX],
#                   y ∈ [-DEFENCE_Y_MM, DEFENCE_Y_MM]
_POS_X = FIELD_X_MAX - DEFENCE_X_MM / 2   # centre of positive box in x
_NEG_X = FIELD_X_MIN + DEFENCE_X_MM / 2   # centre of negative box in x


def test_is_in_penalty_box_center_of_positive_side():
    assert is_in_penalty_box((_POS_X, 0.0), positive_side=True) is True


def test_is_in_penalty_box_positive_point_not_in_negative_side():
    assert is_in_penalty_box((_POS_X, 0.0), positive_side=False) is False


def test_is_in_penalty_box_center_of_negative_side():
    assert is_in_penalty_box((_NEG_X, 0.0), positive_side=False) is True


def test_is_in_penalty_box_negative_point_not_in_positive_side():
    assert is_in_penalty_box((_NEG_X, 0.0), positive_side=True) is False


def test_is_in_penalty_box_midfield_is_outside_both_sides():
    assert is_in_penalty_box((0.0, 0.0), positive_side=True) is False
    assert is_in_penalty_box((0.0, 0.0), positive_side=False) is False


def test_is_in_penalty_box_correct_x_but_outside_y_bounds():
    # x is within the positive box but y exceeds DEFENCE_Y_MM.
    assert is_in_penalty_box((_POS_X, DEFENCE_Y_MM + 100.0), positive_side=True) is False


def test_is_in_penalty_box_point_on_y_boundary_with_zero_margin():
    assert is_in_penalty_box((_POS_X, DEFENCE_Y_MM), positive_side=True, margin=0.0) is True


def test_is_in_penalty_box_margin_excludes_point_near_x_edge():
    # Point 50 mm inside the x-entry of the positive box; margin=200 pushes the
    # effective x_min to FIELD_X_MAX - DEFENCE_X_MM + 200, so 50 mm < 200 mm → False.
    x = FIELD_X_MAX - DEFENCE_X_MM + 50.0
    assert is_in_penalty_box((x, 0.0), positive_side=True, margin=200.0) is False


def test_is_in_penalty_box_margin_keeps_point_well_inside():
    # Point at box centre with a modest margin → still inside.
    assert is_in_penalty_box((_POS_X, 0.0), positive_side=True, margin=200.0) is True


@pytest.mark.parametrize("positive_side", [True, False])
def test_is_in_penalty_box_default_margin_is_zero(positive_side):
    # Calling without margin should behave identically to margin=0.
    pt = (_POS_X if positive_side else _NEG_X, 0.0)
    assert is_in_penalty_box(pt, positive_side=positive_side) == is_in_penalty_box(
        pt, positive_side=positive_side, margin=0.0
    )
