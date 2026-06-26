from math import pi

import pytest

from TeamControl.world.map.geometry import (
    angle_diff,
    angular_velocity,
    distance_2_segment,
    linear_velocity,
)


def test_distance_to_segment_clamps_projection_to_endpoints():
    assert distance_2_segment((5.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == 3.0
    assert distance_2_segment((15.0, 0.0), (0.0, 0.0), (10.0, 0.0)) == 5.0
    assert distance_2_segment((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == 5.0


def test_angle_diff_wraps_across_pi_boundary():
    assert angle_diff(pi - 0.1, -pi + 0.1) == pytest.approx(0.2)
    assert angle_diff(-pi + 0.1, pi - 0.1) == pytest.approx(-0.2)


def test_velocity_helpers_reject_non_positive_elapsed_time():
    with pytest.raises(ValueError, match="positive"):
        linear_velocity(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        angular_velocity(0.0, 1.0, -1.0)
