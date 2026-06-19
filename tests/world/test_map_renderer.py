import pytest

from TeamControl.world.map.obstacles import Obstacle
from TeamControl.world.map.renderer import (
    RenderLayer,
    RenderPolyline,
    Renderer,
)
from TeamControl.world.map.world_map import WorldMap


def moving_obstacle():
    old = Obstacle(
        timestamp=9.9,
        robot_id=2,
        isYellow=True,
        pos_mm=(0.0, 0.0, 0.0),
    )
    new = Obstacle(
        timestamp=10.0,
        robot_id=2,
        isYellow=True,
        pos_mm=(100.0, 0.0, 0.0),
    )
    new.update_vel_from(old)
    return new


def test_renderer_builds_toggleable_velocity_and_prediction_layers():
    world_map = WorldMap()
    world_map.obs = [moving_obstacle()]

    data = Renderer(
        prediction_horizon_ms=250,
        velocity_vector_seconds=0.25,
    ).render(world_map, now_s=10.0)

    robots = data.layer("Robots")
    velocity = data.layer("Velocity vectors")
    prediction = data.layer("Predicted clearance")

    assert robots.robots[0].center_mm == (100.0, 0.0)
    assert robots.robots[0].orientation_rad == 0.0
    assert velocity.vectors[0].end_mm == pytest.approx((350.0, 0.0))
    assert prediction.visible_by_default is False
    assert prediction.circles[0].center_mm == pytest.approx((350.0, 0.0))
    assert prediction.circles[0].radius_mm == pytest.approx(370.0)


def test_renderer_includes_ball_velocity_vector():
    world_map = WorldMap()
    world_map.ball = (10.0, 20.0)
    world_map.ball_vel_mmps = (100.0, -40.0)
    world_map.ball_visible = True

    data = Renderer(velocity_vector_seconds=0.5).render(world_map, now_s=0.0)

    ball = data.layer("Ball")
    assert ball.circles[0].center_mm == (10.0, 20.0)
    assert ball.vectors[0].end_mm == pytest.approx((60.0, 0.0))


def test_world_map_render_data_accepts_future_overlay_layers():
    world_map = WorldMap()
    voronoi = RenderLayer(
        "Voronoi edges",
        polylines=(
            RenderPolyline(
                points_mm=((0.0, 0.0), (100.0, 100.0)),
                color="#ffffff",
            ),
        ),
    )

    data = world_map.get_render_data(now_s=0.0, extra_layers=(voronoi,))

    assert data.layer("Voronoi edges") == voronoi
