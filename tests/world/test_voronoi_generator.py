import warnings

import pytest
from types import SimpleNamespace

from TeamControl.world.field_config import ROBOT_RADIUS_MM, SAFE_MARGIN
from TeamControl.world.map.voronoi_generator import (
    VoronoiObstacle,
    edge_clearance_mm,
    generate_bounded_voronoi_map,
    generate_voronoi_map_from_world_map,
    obstacle_clearance_mm,
)
from TeamControl.world.map.voronoi_plot import save_voronoi_map_plot
from TeamControl.world.map.world_map import WorldMap
from TeamControl.world.snapshot import RobotSnapshot, WorldSnapshot, empty_robot_team


def _warn_if_safe_margin_is_tiny() -> None:
    if SAFE_MARGIN < 10.0:
        warnings.warn(
            f"WARNING: SAFE_MARGIN is getting lower than 10mm "
            f"({SAFE_MARGIN:.1f}mm).",
            RuntimeWarning,
            stacklevel=2,
        )


def test_generator_builds_closed_bounded_cells_for_requested_virtual_nodes():
    voronoi_map = generate_bounded_voronoi_map(
        12,
        min_clearance_mm=120.0,
        seed=7,
    )
    x_min, x_max, y_min, y_max = voronoi_map.clipped_bounds_mm

    assert len(voronoi_map.sites_mm) == 12
    assert len(voronoi_map.cells) == 12
    for cell in voronoi_map.cells:
        assert len(cell.polygon_mm) >= 3
        for x, y in cell.polygon_mm:
            assert x_min <= x <= x_max
            assert y_min <= y <= y_max


def test_navigation_edges_guarantee_requested_clearance():
    min_clearance_mm = 120.0
    voronoi_map = generate_bounded_voronoi_map(
        16,
        min_clearance_mm=min_clearance_mm,
        seed=3,
    )
    node_by_id = {node.id: node for node in voronoi_map.nodes}

    assert voronoi_map.edges
    for edge in voronoi_map.edges:
        start = node_by_id[edge.start_id]
        end = node_by_id[edge.end_id]
        clearance = edge_clearance_mm(
            (start.x, start.y),
            (end.x, end.y),
            voronoi_map.sites_mm,
            voronoi_map.bounds_mm,
        )

        assert edge.clearance == pytest.approx(clearance)
        assert clearance >= min_clearance_mm


def test_obstacles_filter_navigation_edges_by_inflated_radius():
    min_clearance_mm = 120.0
    obstacle = VoronoiObstacle((0.0, 0.0), radius_mm=300.0, label="test")
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="grid",
        grid_spacing_x_mm=2200.0,
        grid_spacing_y_mm=1450.0,
        min_clearance_mm=min_clearance_mm,
        obstacles=(obstacle,),
    )
    node_by_id = {node.id: node for node in voronoi_map.nodes}

    assert voronoi_map.obstacles == (obstacle,)
    for edge in voronoi_map.edges:
        start = node_by_id[edge.start_id]
        end = node_by_id[edge.end_id]
        clearance = obstacle_clearance_mm(
            (start.x, start.y),
            (end.x, end.y),
            voronoi_map.obstacles,
        )

        assert clearance >= min_clearance_mm


def test_adjacent_obstacles_block_edges_through_too_narrow_gap():
    min_clearance_mm = 120.0
    obstacles = (
        VoronoiObstacle((-220.0, 0.0), radius_mm=300.0, label="left"),
        VoronoiObstacle((220.0, 0.0), radius_mm=300.0, label="right"),
    )
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="grid",
        grid_spacing_x_mm=1100.0,
        grid_spacing_y_mm=725.0,
        min_clearance_mm=min_clearance_mm,
        obstacles=obstacles,
    )
    node_by_id = {node.id: node for node in voronoi_map.nodes}

    assert voronoi_map.edges
    for edge in voronoi_map.edges:
        start = node_by_id[edge.start_id]
        end = node_by_id[edge.end_id]
        clearance = obstacle_clearance_mm(
            (start.x, start.y),
            (end.x, end.y),
            voronoi_map.obstacles,
        )

        assert clearance >= min_clearance_mm
        assert not (
            start.y == pytest.approx(0.0)
            and end.y == pytest.approx(0.0)
            and min(start.x, end.x) < 0.0 < max(start.x, end.x)
        )


def test_obstacle_weighting_increases_cost_for_nearby_safe_edges():
    obstacle = VoronoiObstacle((0.0, 0.0), radius_mm=300.0, label="test")
    unweighted = generate_bounded_voronoi_map(
        placement_mode="grid",
        grid_spacing_x_mm=2200.0,
        grid_spacing_y_mm=1450.0,
        obstacles=(obstacle,),
        obstacle_cost_weight=0.0,
    )
    weighted = generate_bounded_voronoi_map(
        placement_mode="grid",
        grid_spacing_x_mm=2200.0,
        grid_spacing_y_mm=1450.0,
        obstacles=(obstacle,),
        obstacle_cost_weight=2.0,
    )

    unweighted_costs = sorted(edge.cost for edge in unweighted.edges)
    weighted_costs = sorted(edge.cost for edge in weighted.edges)

    assert len(unweighted_costs) == len(weighted_costs)
    assert max(weighted_costs) > max(unweighted_costs)


def test_world_map_planning_obstacles_can_shape_voronoi_map():
    yellow = list(empty_robot_team())
    yellow[0] = RobotSnapshot(True, 0, 0.0, 0.0, 0.0)
    world_map = WorldMap(
        snapshot=WorldSnapshot(
            version=1,
            timestamp=1.0,
            frame_number=1,
            ball=None,
            yellow=tuple(yellow),
            blue=empty_robot_team(),
            us_yellow=True,
            us_positive=True,
        )
    )

    voronoi_map = generate_voronoi_map_from_world_map(
        world_map,
        now_s=1.0,
        horizon_ms=0,
        placement_mode="density_grid",
        density_percent=10.0,
    )

    assert len(voronoi_map.virtual_sites_mm) == 8
    assert len(voronoi_map.obstacles) == 1
    assert voronoi_map.obstacles[0].label == "Y0"


def test_world_map_field_size_shapes_voronoi_bounds():
    world_map = WorldMap(field=SimpleNamespace(field_length=12000, field_width=8000))

    voronoi_map = generate_voronoi_map_from_world_map(
        world_map,
        now_s=1.0,
        horizon_ms=0,
        placement_mode="density_grid",
        density_percent=10.0,
        boundary_inset_mm=100.0,
    )

    assert voronoi_map.field_bounds_mm == (-6000.0, 6000.0, -4000.0, 4000.0)
    assert voronoi_map.bounds_mm == (-5900.0, 5900.0, -3900.0, 3900.0)


def test_navigation_bounds_are_inset_from_real_field_bounds():
    _warn_if_safe_margin_is_tiny()
    min_clearance_mm = ROBOT_RADIUS_MM + SAFE_MARGIN
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="density_grid",
        density_percent=10.0,
        boundary_inset_mm=100.0,
    )
    x_min, x_max, y_min, y_max = voronoi_map.bounds_mm

    assert voronoi_map.field_bounds_mm == (-4500.0, 4500.0, -3000.0, 3000.0)
    assert voronoi_map.bounds_mm == (-4400.0, 4400.0, -2900.0, 2900.0)
    assert voronoi_map.clipped_bounds_mm == (
        x_min + min_clearance_mm,
        x_max - min_clearance_mm,
        y_min + min_clearance_mm,
        y_max - min_clearance_mm,
    )
    assert voronoi_map.boundary_inset_mm == 100.0


def test_density_ten_percent_uses_coarse_eight_section_grid():
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="density_grid",
        density_percent=10.0,
    )

    assert len(voronoi_map.sites_mm) == 8


def test_explicit_grid_spacing_places_nodes_on_navigation_bounds():
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="grid",
        grid_spacing_x_mm=2200.0,
        grid_spacing_y_mm=1450.0,
        boundary_inset_mm=100.0,
    )
    xs = sorted({site[0] for site in voronoi_map.sites_mm})
    ys = sorted({site[1] for site in voronoi_map.sites_mm})

    assert xs[0] == pytest.approx(-4400.0)
    assert xs[-1] == pytest.approx(4400.0)
    assert ys[0] == pytest.approx(-2900.0)
    assert ys[-1] == pytest.approx(2900.0)


def test_render_layer_contains_cell_and_navigation_polylines():
    voronoi_map = generate_bounded_voronoi_map(
        6,
        min_clearance_mm=120.0,
        seed=11,
    )

    layer = voronoi_map.render_layer()

    assert layer.name == "Voronoi map"
    assert len(layer.polylines) >= len(voronoi_map.cells)


def test_clear_map_adds_safe_perimeter_navigation_edges():
    min_clearance_mm = 120.0
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="density_grid",
        density_percent=10.0,
        min_clearance_mm=min_clearance_mm,
        boundary_inset_mm=100.0,
    )
    node_by_id = {node.id: node for node in voronoi_map.nodes}
    x_min, x_max, y_min, y_max = voronoi_map.bounds_mm
    perimeter_points = {
        (x_min + min_clearance_mm, y_min + min_clearance_mm),
        (x_max - min_clearance_mm, y_min + min_clearance_mm),
        (x_max - min_clearance_mm, y_max - min_clearance_mm),
        (x_min + min_clearance_mm, y_max - min_clearance_mm),
    }
    perimeter_edges = []
    for edge in voronoi_map.edges:
        start = node_by_id[edge.start_id]
        end = node_by_id[edge.end_id]
        if (start.x, start.y) in perimeter_points and (end.x, end.y) in perimeter_points:
            perimeter_edges.append(edge)

    assert len(perimeter_edges) == 4


def test_boundary_touching_cell_edges_are_clipped_into_safe_navigation_edges():
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="density_grid",
        density_percent=10.0,
        min_clearance_mm=120.0,
        boundary_inset_mm=100.0,
    )

    assert len(voronoi_map.edges) > 6


def test_obstacle_can_block_perimeter_navigation_edge():
    min_clearance_mm = 120.0
    obstacle = VoronoiObstacle((0.0, -2780.0), radius_mm=300.0, label="bottom")
    voronoi_map = generate_bounded_voronoi_map(
        placement_mode="density_grid",
        density_percent=10.0,
        min_clearance_mm=min_clearance_mm,
        boundary_inset_mm=100.0,
        obstacles=(obstacle,),
    )
    node_by_id = {node.id: node for node in voronoi_map.nodes}
    x_min, x_max, y_min, _ = voronoi_map.bounds_mm
    bottom_y = y_min + min_clearance_mm
    bottom_start = (x_min + min_clearance_mm, bottom_y)
    bottom_end = (x_max - min_clearance_mm, bottom_y)

    assert not any(
        {
            (node_by_id[edge.start_id].x, node_by_id[edge.start_id].y),
            (node_by_id[edge.end_id].x, node_by_id[edge.end_id].y),
        }
        == {bottom_start, bottom_end}
        for edge in voronoi_map.edges
    )


def test_visual_debug_plot_can_be_saved(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    voronoi_map = generate_bounded_voronoi_map(
        16,
        min_clearance_mm=120.0,
        seed=3,
    )

    output_path = save_voronoi_map_plot(
        voronoi_map,
        tmp_path / "bounded_voronoi_map.png",
    )

    print(f"Voronoi plot written to: {output_path}")
    assert output_path.exists()
    assert output_path.stat().st_size > 0
