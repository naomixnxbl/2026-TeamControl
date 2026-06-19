from types import SimpleNamespace

import pytest

from TeamControl.world.map.observer import FieldAnalyzer
from TeamControl.world.map.world_map import WorldMap
from TeamControl.world.snapshot import (
    BallSnapshot,
    RobotSnapshot,
    WorldSnapshot,
    empty_robot_team,
)


def make_snapshot(
    timestamp,
    robots=(),
    ball=None,
    ball_candidates=(),
    ball_left_field=None,
):
    yellow = list(empty_robot_team())
    blue = list(empty_robot_team())
    for robot in robots:
        team = yellow if robot.isYellow else blue
        team[robot.id] = robot
    return WorldSnapshot(
        version=1,
        timestamp=timestamp,
        frame_number=1,
        ball=ball,
        yellow=tuple(yellow),
        blue=tuple(blue),
        us_yellow=True,
        us_positive=True,
        ball_candidates=ball_candidates,
        ball_left_field=ball_left_field,
    )


def robot(robot_id, is_yellow, x, y, theta=0.0):
    return RobotSnapshot(is_yellow, robot_id, x, y, theta)


def test_update_builds_obstacles_and_tracks_latest_robot_velocity():
    world_map = WorldMap()
    world_map.update(make_snapshot(1.0, [robot(0, True, 0.0, 0.0)]))
    world_map.update(make_snapshot(1.5, [robot(0, True, 100.0, 50.0)]))

    obs = world_map._get_robot_obstacle(0, True)

    assert obs.vel_mmps == pytest.approx((200.0, 100.0))
    position, velocity = world_map.get_robot_trajectory(0, True, 500)
    assert position == pytest.approx((200.0, 100.0))
    assert velocity == pytest.approx((200.0, 100.0))


def test_update_tracks_ball_trajectory():
    world_map = WorldMap()
    world_map.update(make_snapshot(1.0, ball=BallSnapshot(0.0, 0.0)))
    world_map.update(make_snapshot(1.5, ball=BallSnapshot(100.0, 50.0)))

    position, velocity = world_map.get_ball_trajectory(500)
    assert position == pytest.approx((200.0, 100.0))
    assert velocity == pytest.approx((200.0, 100.0))


def test_ball_tracking_selects_highest_confidence_plausible_candidate():
    world_map = WorldMap()
    world_map.update(make_snapshot(1.0, ball=BallSnapshot(0.0, 0.0)))

    world_map.update(
        make_snapshot(
            1.1,
            ball_candidates=(
                BallSnapshot(10.0, 0.0, confidence=0.4),
                BallSnapshot(20.0, 0.0, confidence=0.9),
                BallSnapshot(3000.0, 0.0, confidence=1.0),
            ),
        )
    )

    assert world_map.ball == (20.0, 0.0)


def test_ball_tracking_uses_prediction_distance_as_confidence_tie_breaker():
    world_map = WorldMap()
    world_map.update(make_snapshot(1.0, ball=BallSnapshot(0.0, 0.0)))

    world_map.update(
        make_snapshot(
            1.1,
            ball_candidates=(
                BallSnapshot(30.0, 0.0, confidence=0.9),
                BallSnapshot(10.0, 0.0, confidence=0.9),
            ),
        )
    )

    assert world_map.ball == (10.0, 0.0)


def test_ball_validation_rejects_implausible_jump_and_keeps_last_valid_position():
    world_map = WorldMap()
    world_map.update(make_snapshot(1.0, ball=BallSnapshot(0.0, 0.0)))
    world_map.update(make_snapshot(1.01, ball=BallSnapshot(1000.0, 0.0)))

    assert world_map.ball == (0.0, 0.0)
    assert world_map.ball_vel_mmps == (0.0, 0.0)
    assert not world_map.ball_visible
    assert world_map.last_rejected_ball_pos_mm == (1000.0, 0.0)
    assert world_map.last_ball_rejection_reason == "trajectory_error"


def test_ball_validation_rejects_low_confidence_observation():
    world_map = WorldMap()

    world_map.update(
        make_snapshot(
            1.0,
            ball=BallSnapshot(10.0, 20.0, confidence=0.05),
        )
    )

    assert world_map.ball is None
    assert world_map.last_ball_rejection_reason == "low_confidence"


def test_missing_ball_preserves_last_valid_position_but_marks_it_not_visible():
    world_map = WorldMap()
    world_map.update(
        make_snapshot(1.0, ball=BallSnapshot(10.0, 20.0)),
        received_at_s=1.0,
    )

    world_map.update(make_snapshot(1.1), received_at_s=1.1)

    assert world_map.ball == (10.0, 20.0)
    assert not world_map.ball_visible
    assert world_map.ball_age_s(now_s=1.4) == pytest.approx(0.4)


def test_out_of_bounds_ball_is_recorded_without_updating_trajectory():
    field = SimpleNamespace(field_length=9000, field_width=6000)
    world_map = WorldMap(field=field)
    world_map.update(make_snapshot(1.0, ball=BallSnapshot(0.0, 0.0)))

    world_map.update(make_snapshot(1.1, ball=BallSnapshot(4600.0, 0.0)))

    assert world_map.ball == (0.0, 0.0)
    assert world_map.possible_ball_left_field_pos_mm == (4600.0, 0.0)
    assert world_map.last_ball_rejection_reason == "out_of_bounds"


def test_confirmed_ball_left_field_location_is_copied_from_snapshot():
    world_map = WorldMap()

    world_map.update(
        make_snapshot(
            1.0,
            ball_left_field=(4500.0, 100.0),
        )
    )

    assert world_map.ball_left_field_pos_mm == (4500.0, 100.0)


def test_nearby_queries_are_sorted_and_team_filtered():
    world_map = WorldMap(
        snapshot=make_snapshot(
            1.0,
            [
                robot(0, True, 0.0, 0.0),
                robot(1, True, 300.0, 0.0),
                robot(2, True, 100.0, 0.0),
                robot(0, False, 200.0, 0.0),
            ],
        )
    )

    assert world_map.get_nearby_teammates(0, True) == [2, 1]
    assert world_map.get_nearby_enemies(0, True) == [0]


def test_analyzer_path_check_uses_future_obstacle_position():
    world_map = WorldMap(horizon_ms=1000)
    world_map.update(make_snapshot(1.0, [robot(0, False, 0.0, 400.0)]))
    world_map.update(make_snapshot(2.0, [robot(0, False, 0.0, 300.0)]))
    analyzer = FieldAnalyzer(world_map)

    assert analyzer.is_path_blocked((-500.0, 0.0), (500.0, 0.0))


def test_path_ignore_uses_team_and_robot_id():
    world_map = WorldMap(
        snapshot=make_snapshot(
            1.0,
            [
                robot(0, True, 0.0, 0.0),
                robot(0, False, 300.0, 0.0),
            ],
        )
    )

    assert not world_map.is_path_free(
        (200.0, 0.0),
        (400.0, 0.0),
        ignore_robots={(True, 0)},
        horizon_ms=0,
    )


def test_target_box_respects_offset():
    world_map = WorldMap()

    assert world_map.is_target_in_box((900.0, 400.0), 1000.0, 500.0, offset=100.0)
    assert not world_map.is_target_in_box(
        (901.0, 400.0),
        1000.0,
        500.0,
        offset=100.0,
    )


def test_planning_obstacles_compensate_for_observation_age_and_horizon():
    world_map = WorldMap()
    world_map.update(
        make_snapshot(1.0, [robot(0, False, 0.0, 0.0)]),
        received_at_s=2.0,
    )
    world_map.update(
        make_snapshot(2.0, [robot(0, False, 100.0, 0.0)]),
        received_at_s=3.0,
    )

    planning_obs, = world_map.get_planning_obstacles(
        now_s=3.1,
        horizon_ms=200,
    )

    assert planning_obs.pos_mm == pytest.approx((130.0, 0.0))
    assert planning_obs.radius_mm == pytest.approx(150.0)
    assert planning_obs.observation_age_ms == pytest.approx(100.0, abs=1e-3)
    assert planning_obs.prediction_horizon_ms == pytest.approx(300.0)


def test_planning_obstacle_age_uses_local_receipt_clock_not_capture_clock():
    world_map = WorldMap()
    world_map.update(
        make_snapshot(10.0, [robot(0, False, 0.0, 0.0)]),
        received_at_s=1_700_000_000.0,
    )
    world_map.update(
        make_snapshot(10.1, [robot(0, False, 100.0, 0.0)]),
        received_at_s=1_700_000_000.1,
    )

    planning_obs, = world_map.get_planning_obstacles(
        now_s=1_700_000_000.2,
        horizon_ms=250,
    )

    assert planning_obs.observation_age_ms == pytest.approx(100.0, abs=1e-3)
    assert planning_obs.radius_mm == pytest.approx(470.0)


def test_planning_obstacles_can_ignore_the_controlled_robot():
    world_map = WorldMap(
        snapshot=make_snapshot(
            1.0,
            [
                robot(0, True, 0.0, 0.0),
                robot(0, False, 300.0, 0.0),
            ],
        )
    )

    planning_obs = world_map.get_planning_obstacles(
        now_s=1.0,
        horizon_ms=0,
        ignore_robots={(True, 0)},
    )

    assert [(obs.isYellow, obs.robot_id) for obs in planning_obs] == [
        (False, 0)
    ]
