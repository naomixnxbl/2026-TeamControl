import pytest

from TeamControl.world.map.obstacles import Obstacle


def test_predicted_pos_defaults_to_current_position_without_velocity():
    obstacle = Obstacle(1.0, 0, True, (100.0, -50.0, 0.0))

    assert obstacle.predicted_pos(100) == pytest.approx((100.0, -50.0))


def test_new_observation_updates_velocity_from_old_observation():
    old_obstacle = Obstacle(1.0, 0, True, (100.0, -50.0, 0.0))
    new_obstacle = Obstacle(1.5, 0, True, (200.0, 50.0, 0.0))

    assert new_obstacle.update_vel_from(old_obstacle) == pytest.approx(
        (200.0, 200.0)
    )
    assert new_obstacle.predicted_pos(250) == pytest.approx((250.0, 100.0))


@pytest.mark.parametrize(
    "old_obstacle",
    [
        Obstacle(1.0, 0, True, (100.0, -50.0, 0.0)),
        Obstacle(1.5, 1, True, (100.0, -50.0, 0.0)),
        Obstacle(1.5, 0, False, (100.0, -50.0, 0.0)),
    ],
)
def test_update_vel_from_rejects_invalid_old_observation(old_obstacle):
    new_obstacle = Obstacle(1.0, 0, True, (200.0, 50.0, 0.0))

    with pytest.raises(ValueError, match="older observation"):
        new_obstacle.update_vel_from(old_obstacle)


def test_predicted_pos_rejects_negative_horizon():
    obstacle = Obstacle(1.0, 0, True, (100.0, -50.0, 0.0))

    with pytest.raises(ValueError, match="non-negative"):
        obstacle.predicted_pos(-1)


def test_future_clearance_uses_predicted_position_and_dynamic_radius():
    old_obstacle = Obstacle(1.0, 0, True, (0.0, 400.0, 0.0))
    new_obstacle = Obstacle(2.0, 0, True, (0.0, 300.0, 0.0))
    new_obstacle.update_vel_from(old_obstacle)

    assert new_obstacle.clearance_to_path_mm((-500.0, 0.0), (500.0, 0.0)) == 90.0
    assert new_obstacle.clearance_to_path_mm(
        (-500.0, 0.0),
        (500.0, 0.0),
        horizon_ms=1000,
    ) == -110.0
