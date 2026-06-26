from types import SimpleNamespace

import pytest

from TeamControl.SSL.vision.frame import Frame
from TeamControl.world.model import WorldModel


def make_detection(
    camera_id,
    t_capture,
    t_sent,
    frame_number=7,
    robots_yellow=(),
    balls=(),
):
    return SimpleNamespace(
        camera_id=camera_id,
        frame_number=frame_number,
        t_capture=t_capture,
        t_sent=t_sent,
        balls=balls,
        robots_yellow=robots_yellow,
        robots_blue=[],
    )


def make_robot(x, y=0.0):
    return SimpleNamespace(
        robot_id=0,
        confidence=1.0,
        x=x,
        y=y,
        orientation=0.0,
        pixel_x=0.0,
        pixel_y=0.0,
        height=0.0,
    )


def make_ball(confidence, x):
    return SimpleNamespace(
        confidence=confidence,
        x=x,
        y=0.0,
        pixel_x=0.0,
        pixel_y=0.0,
    )


def test_frame_preserves_raw_ball_candidate_order_across_cameras():
    frame = Frame.from_proto(
        make_detection(0, 10.0, 10.01, balls=[make_ball(0.3, 30.0)]),
        max_cameras=2,
    )

    frame.update(
        make_detection(1, 10.02, 10.03, balls=[make_ball(0.9, 90.0)])
    )

    assert [ball.position for ball in frame.balls] == [
        (30.0, 0.0),
        (90.0, 0.0),
    ]
    assert frame.ball.position == (30.0, 0.0)


def test_frame_preserves_latest_capture_timestamp_when_combining_cameras():
    frame = Frame.from_proto(make_detection(0, 10.0, 10.01), max_cameras=2)

    frame.update(make_detection(1, 10.02, 10.03))

    assert frame.t_capture == 10.02
    assert frame.t_sent == 10.03


def test_world_model_snapshot_uses_vision_capture_timestamp():
    frame = Frame.from_proto(make_detection(0, 10.0, 10.01), max_cameras=1)
    world_model = WorldModel()

    world_model.add_new_frame(frame)

    assert world_model.snapshot().timestamp == 10.0


def test_world_model_map_selects_best_candidate_while_frame_keeps_raw_order():
    frame = Frame.from_proto(
        make_detection(
            0,
            10.0,
            10.01,
            balls=[
                make_ball(0.3, 30.0),
                make_ball(0.9, 90.0),
            ],
        ),
        max_cameras=1,
    )
    world_model = WorldModel()

    world_model.add_new_frame(frame)

    assert frame.ball.position == (30.0, 0.0)
    assert world_model.world_map.ball == (90.0, 0.0)


def test_world_model_refreshes_age_adjusted_planning_obstacles_on_new_frame():
    world_model = WorldModel()
    world_model.add_new_frame(
        Frame.from_proto(
            make_detection(0, 10.0, 10.01, 1, [make_robot(0.0)]),
            max_cameras=1,
        )
    )
    world_model.add_new_frame(
        Frame.from_proto(
            make_detection(0, 11.0, 11.01, 2, [make_robot(100.0)]),
            max_cameras=1,
        )
    )

    latest_obs = world_model.world_map.get_obstacles()[0]
    planning_obs, = world_model.get_planning_obstacles(
        now_s=latest_obs.received_at_s + 0.1,
        horizon_ms=200,
    )

    assert planning_obs.pos_mm == pytest.approx((130.0, 0.0))
    assert planning_obs.radius_mm == pytest.approx(150.0)
    assert planning_obs.vel_mmps == pytest.approx((100.0, 0.0))
