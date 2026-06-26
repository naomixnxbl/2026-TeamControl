"""
Unit tests for the wrap_angle helper in robot/ball_nav.py.

wrap_angle maps any angle to the range (-π, π], which is what every
angle-error calculation in the codebase expects.  Getting this wrong causes
the robot to spin the wrong way or do a full circle instead of a tiny
correction.
"""
import math

import pytest

from TeamControl.robot.ball_nav import wrap_angle


class TestWrapAngle:
    def test_zero_unchanged(self):
        assert wrap_angle(0.0) == pytest.approx(0.0)

    def test_small_positive_unchanged(self):
        assert wrap_angle(0.5) == pytest.approx(0.5)

    def test_small_negative_unchanged(self):
        assert wrap_angle(-0.5) == pytest.approx(-0.5)

    def test_just_under_pi_unchanged(self):
        assert wrap_angle(math.pi - 0.01) == pytest.approx(math.pi - 0.01)

    def test_just_over_pi_wraps_to_near_negative_pi(self):
        # π + 0.1 should become -(π - 0.1)
        assert wrap_angle(math.pi + 0.1) == pytest.approx(-(math.pi - 0.1))

    def test_just_below_negative_pi_wraps_to_near_positive_pi(self):
        # -(π + 0.1) should become π - 0.1
        assert wrap_angle(-(math.pi + 0.1)) == pytest.approx(math.pi - 0.1)

    def test_two_pi_wraps_to_zero(self):
        assert wrap_angle(2.0 * math.pi) == pytest.approx(0.0)

    def test_negative_two_pi_wraps_to_zero(self):
        assert wrap_angle(-2.0 * math.pi) == pytest.approx(0.0)

    def test_three_pi_wraps_to_pi(self):
        result = wrap_angle(3.0 * math.pi)
        assert abs(result) == pytest.approx(math.pi)

    def test_output_always_within_range(self):
        # Exhaustive sweep: every 15 degrees from -720 to +720.
        for deg in range(-720, 721, 15):
            result = wrap_angle(math.radians(deg))
            assert -math.pi <= result <= math.pi, f"out of range for {deg}°: {result}"
