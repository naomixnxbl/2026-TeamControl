"""
Unit tests for wheel_kinematics.py -- 4-omniwheel inverse/forward
kinematics used by RobotMotionController's opt-in wheel-aware limiter.
"""

import math

import pytest

from TeamControl.robot.motion.wheel_kinematics import (
    DEFAULT_WHEEL_SPEC,
    WheelAccelLimiter,
    body_velocity_from_wheel_speeds,
    scale_to_wheel_speed_limit,
    wheel_speeds,
)

ANGLES = (60.0, 135.0, 225.0, 300.0)  # TurtleRabbit.ini reference layout
RADIUS_M = 0.0885


class TestRoundTrip:
    """wheel_speeds() and body_velocity_from_wheel_speeds() must be exact
    inverses of each other -- that self-consistency is what the limiter
    relies on, independent of grSim's internal sign convention."""

    @pytest.mark.parametrize(
        "vx, vy, w",
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 5.0),
            (0.7, -0.3, 2.0),
            (0.0, 0.0, 0.0),
            (-1.2, 0.4, -3.0),
        ],
    )
    def test_recovers_original_velocity(self, vx, vy, w):
        speeds = wheel_speeds(vx, vy, w, ANGLES, RADIUS_M)
        rvx, rvy, rw = body_velocity_from_wheel_speeds(speeds, ANGLES, RADIUS_M)
        assert rvx == pytest.approx(vx, abs=1e-6)
        assert rvy == pytest.approx(vy, abs=1e-6)
        assert rw == pytest.approx(w, abs=1e-6)

    def test_pure_rotation_gives_equal_speed_on_every_wheel(self):
        # Rotation about the robot's own center is symmetric regardless of
        # wheel angle -- every wheel sees the same |speed| = radius * w.
        speeds = wheel_speeds(0.0, 0.0, 4.0, ANGLES, RADIUS_M)
        assert all(s == pytest.approx(speeds[0]) for s in speeds)
        assert speeds[0] == pytest.approx(RADIUS_M * 4.0)

    def test_asymmetric_angles_give_direction_dependent_speeds(self):
        # The whole point of this module: forward vs sideways motion hits
        # different wheel speeds because the layout isn't evenly spaced.
        forward = wheel_speeds(1.0, 0.0, 0.0, ANGLES, RADIUS_M)
        sideways = wheel_speeds(0.0, 1.0, 0.0, ANGLES, RADIUS_M)
        assert max(abs(s) for s in forward) != pytest.approx(
            max(abs(s) for s in sideways)
        )


class TestScaleToWheelSpeedLimit:
    def test_unchanged_when_under_limit(self):
        vx, vy, w = scale_to_wheel_speed_limit(0.1, 0.0, 0.0, ANGLES, RADIUS_M, 5.0)
        assert (vx, vy, w) == pytest.approx((0.1, 0.0, 0.0))

    def test_disabled_when_limit_is_none(self):
        vx, vy, w = scale_to_wheel_speed_limit(50.0, 0.0, 0.0, ANGLES, RADIUS_M, None)
        assert (vx, vy, w) == (50.0, 0.0, 0.0)

    def test_disabled_when_limit_is_zero(self):
        vx, vy, w = scale_to_wheel_speed_limit(50.0, 0.0, 0.0, ANGLES, RADIUS_M, 0.0)
        assert (vx, vy, w) == (50.0, 0.0, 0.0)

    def test_scales_down_preserving_direction(self):
        vx, vy, w = scale_to_wheel_speed_limit(2.0, 0.0, 0.0, ANGLES, RADIUS_M, 1.0)
        assert vy == pytest.approx(0.0)
        assert w == pytest.approx(0.0)
        assert 0.0 < vx < 2.0

    def test_clamps_worst_wheel_to_exactly_the_limit(self):
        vx, vy, w = scale_to_wheel_speed_limit(2.0, 0.0, 0.0, ANGLES, RADIUS_M, 1.0)
        worst = max(abs(s) for s in wheel_speeds(vx, vy, w, ANGLES, RADIUS_M))
        assert worst == pytest.approx(1.0)


class TestWheelAccelLimiter:
    def test_first_call_passes_through_unchanged(self):
        lim = WheelAccelLimiter(ANGLES, RADIUS_M, max_wheel_accel_mps2=1.0)
        out = lim.limit((3.0, -1.0, 2.0), now=0.0)
        assert out == pytest.approx((3.0, -1.0, 2.0))

    def test_small_change_within_budget_passes_through(self):
        lim = WheelAccelLimiter(ANGLES, RADIUS_M, max_wheel_accel_mps2=10.0)
        lim.limit((0.0, 0.0, 0.0), now=0.0)
        out = lim.limit((0.05, 0.0, 0.0), now=1.0)
        assert out == pytest.approx((0.05, 0.0, 0.0), abs=1e-6)

    def test_large_jump_is_clamped(self):
        lim = WheelAccelLimiter(ANGLES, RADIUS_M, max_wheel_accel_mps2=1.0)
        lim.limit((0.0, 0.0, 0.0), now=0.0)
        vx, vy, w = lim.limit((10.0, 0.0, 0.0), now=1.0)
        assert vx < 10.0
        assert vy == pytest.approx(0.0, abs=1e-6)
        assert w == pytest.approx(0.0, abs=1e-6)

    def test_reset_forgets_history(self):
        lim = WheelAccelLimiter(ANGLES, RADIUS_M, max_wheel_accel_mps2=1.0)
        lim.limit((0.0, 0.0, 0.0), now=0.0)
        lim.reset()
        # After reset, next call is "first call" again -> passes through.
        out = lim.limit((10.0, 0.0, 0.0), now=1.0)
        assert out == pytest.approx((10.0, 0.0, 0.0))


def test_default_wheel_spec_matches_turtle_rabbit_ini():
    assert DEFAULT_WHEEL_SPEC["wheel1_angle_deg"] == 60.0
    assert DEFAULT_WHEEL_SPEC["wheel2_angle_deg"] == 135.0
    assert DEFAULT_WHEEL_SPEC["wheel3_angle_deg"] == 225.0
    assert DEFAULT_WHEEL_SPEC["wheel4_angle_deg"] == 300.0
    assert DEFAULT_WHEEL_SPEC["wheel_radius_mm"] == pytest.approx(32.5)
    assert DEFAULT_WHEEL_SPEC["robot_radius_mm"] == pytest.approx(88.5)
    assert DEFAULT_WHEEL_SPEC["max_wheel_speed_mps"] is None
    assert DEFAULT_WHEEL_SPEC["max_wheel_accel_mps2"] is None
