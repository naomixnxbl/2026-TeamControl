from multiprocessing import Event, Queue

import pytest

from TeamControl.process_workers.voronoi_map_runner import WorldMapRenderWorker
from TeamControl.world.field_config import (
    VORONOI_BOUNDARY_INSET_MM,
    VORONOI_MIN_CLEARANCE_MM,
)
from TeamControl.world.map.renderer import MapRenderData
from TeamControl.world.map.voronoi_generator import VoronoiObstacle


def test_world_map_render_worker_generates_complete_render_data():
    request_q = Queue()
    response_q = Queue()
    request_q.put(
        {
            "request_id": 1,
            "density_percent": 10,
            "max_density_nodes": 80,
            "planning_obstacles": (
                VoronoiObstacle((0.0, 0.0), radius_mm=120.0, label="obs"),
            ),
            "include_voronoi": True,
            "field_length_mm": 12000.0,
            "field_width_mm": 8000.0,
            "ball": (100.0, 200.0),
            "ball_visible": True,
            "planner_paths": (
                {
                    "robot_id": 0,
                    "is_yellow": True,
                    "points": ((0.0, 0.0), (100.0, 0.0)),
                    "timestamp_s": 1.0,
                },
                {
                    "robot_id": 1,
                    "is_yellow": False,
                    "points": (
                        (-5000.0, 5000.0),
                        (-3000.0, 5000.0),
                        (0.0, 0.0),
                    ),
                    "timestamp_s": 1.0,
                },
            ),
        }
    )

    worker = WorldMapRenderWorker(Event(), logger=None)
    worker.setup(request_q, response_q)
    worker.step()

    response = response_q.get(timeout=1)
    assert response["request_id"] == 1
    assert response["generation_ms"] >= 0.0
    assert response["voronoi_generation_ms"] >= 0.0
    assert isinstance(response["render_data"], MapRenderData)
    assert response["render_data"].layer("Robots") is not None
    assert response["render_data"].layer("Ball") is not None
    planned_paths_layer = response["render_data"].layer("Planned paths")
    assert planned_paths_layer is not None
    planned_points = [
        point
        for polyline in planned_paths_layer.polylines
        for point in polyline.points_mm
    ]
    assert planned_points
    assert all(-6000.0 <= point[0] <= 6000.0 for point in planned_points)
    assert all(-4000.0 <= point[1] <= 4000.0 for point in planned_points)

    voronoi_layer = response["render_data"].layer("Voronoi map")
    assert voronoi_layer is not None

    points = [
        point
        for polyline in voronoi_layer.polylines
        for point in polyline.points_mm
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    inset_mm = VORONOI_BOUNDARY_INSET_MM + VORONOI_MIN_CLEARANCE_MM
    assert min(xs) == pytest.approx(-6000.0 + inset_mm)
    assert max(xs) == pytest.approx(6000.0 - inset_mm)
    assert min(ys) == pytest.approx(-4000.0 + inset_mm)
    assert max(ys) == pytest.approx(4000.0 - inset_mm)
