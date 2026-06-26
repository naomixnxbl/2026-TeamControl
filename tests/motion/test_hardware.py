"""
Unit tests for hardware compensation helpers (robot/motion/hardware.py).

These functions correct for real-world effects like momentum overshoot,
speed-scaling calibration, lateral drift, and static friction thresholds.
"""
import math

import pytest

from TeamControl.robot.motion.hardware import (
    apply_hardware_gains,
    apply_min_angular_command,
    apply_min_linear_command,
    shorten_target_for_overshoot,
)


class TestShortenTargetForOvershoot:
    """The robot tends to roll past the target, so we aim slightly short."""

    def test_zero_overshoot_leaves_target_unchanged(self):
        result = shorten_target_for_overshoot((3.0, 4.0), 0.0)
        assert result == pytest.approx((3.0, 4.0))

    def test_negative_overshoot_treated_as_zero(self):
        result = shorten_target_for_overshoot((3.0, 4.0), -5.0)
        assert result == pytest.approx((3.0, 4.0))

    def test_normal_overshoot_shortens_distance(self):
        # dist=5, overshoot=1 → new_dist=4, scale=4/5=0.8
        result = shorten_target_for_overshoot((3.0, 4.0), 1.0)
        assert result == pytest.approx((2.4, 3.2))

    def test_overshoot_larger_than_dist_clamps_to_zero(self):
        result = shorten_target_for_overshoot((3.0, 4.0), 10.0)
        assert result == pytest.approx((0.0, 0.0))

    def test_direction_preserved_after_shortening(self):
        x, y = shorten_target_for_overshoot((3.0, 4.0), 1.0)
        assert y / x == pytest.approx(4.0 / 3.0)

    def test_zero_target_stays_zero(self):
        result = shorten_target_for_overshoot((0.0, 0.0), 5.0)
        assert result == pytest.approx((0.0, 0.0))


class TestApplyHardwareGains:
    """Correct for measured speed-scale and lateral drift."""

    def test_speed_scale_two_halves_velocity(self):
        # Robot actually moves twice as fast as commanded → divide by 2.
        vx, vy = apply_hardware_gains(1.0, 0.0, {"speed_scale": 2.0, "lateral_drift_per_m": 0.0})
        assert vx == pytest.approx(0.5)
        assert vy == pytest.approx(0.0)

    def test_speed_scale_one_leaves_velocity_unchanged(self):
        vx, vy = apply_hardware_gains(0.5, 0.3, {"speed_scale": 1.0, "lateral_drift_per_m": 0.0})
        assert vx == pytest.approx(0.5)
        assert vy == pytest.approx(0.3)

    def test_near_zero_speed_scale_skipped_to_avoid_division(self):
        # speed_scale ≤ 0.01 is ignored to prevent divide-by-near-zero.
        vx, vy = apply_hardware_gains(1.0, 0.0, {"speed_scale": 0.0, "lateral_drift_per_m": 0.0})
        assert vx == pytest.approx(1.0)

    def test_lateral_drift_adjusts_vy(self):
        # 100mm drift per meter → ratio=0.1, correction: vy -= vx * 0.1
        vx, vy = apply_hardware_gains(1.0, 0.0, {"speed_scale": 1.0, "lateral_drift_per_m": 100.0})
        assert vx == pytest.approx(1.0)
        assert vy == pytest.approx(-0.1)

    def test_no_drift_leaves_vy_unchanged(self):
        vx, vy = apply_hardware_gains(1.0, 0.5, {"speed_scale": 1.0, "lateral_drift_per_m": 0.0})
        assert vy == pytest.approx(0.5)

    def test_missing_keys_use_safe_defaults(self):
        # Empty gains dict should not raise; speed_scale defaults to 1.0.
        vx, vy = apply_hardware_gains(0.8, 0.0, {})
        assert vx == pytest.approx(0.8)


class TestApplyMinLinearCommand:
    """Boost tiny commands so the robot overcomes static friction and actually moves."""

    def test_zero_speed_stays_zero(self):
        assert apply_min_linear_command(0.0, 0.0, 0.05) == pytest.approx((0.0, 0.0))

    def test_speed_above_min_unchanged(self):
        assert apply_min_linear_command(0.1, 0.0, 0.05) == pytest.approx((0.1, 0.0))

    def test_speed_exactly_at_min_unchanged(self):
        assert apply_min_linear_command(0.05, 0.0, 0.05) == pytest.approx((0.05, 0.0))

    def test_speed_below_min_boosted_to_min(self):
        vx, vy = apply_min_linear_command(0.03, 0.0, 0.05)
        assert math.hypot(vx, vy) == pytest.approx(0.05)

    def test_direction_preserved_when_boosted(self):
        vx, vy = apply_min_linear_command(0.03, 0.04, 0.1)  # speed=0.05 < min=0.1
        assert vy / vx == pytest.approx(0.04 / 0.03)

    def test_zero_min_v_leaves_command_unchanged(self):
        assert apply_min_linear_command(0.01, 0.0, 0.0) == pytest.approx((0.01, 0.0))


class TestApplyMinAngularCommand:
    """Boost tiny angular commands so the robot actually rotates."""

    def test_zero_w_stays_zero(self):
        assert apply_min_angular_command(0.0, 0.05) == pytest.approx(0.0)

    def test_w_above_min_unchanged(self):
        assert apply_min_angular_command(0.5, 0.05) == pytest.approx(0.5)

    def test_w_exactly_at_min_unchanged(self):
        assert apply_min_angular_command(0.05, 0.05) == pytest.approx(0.05)

    def test_small_positive_w_boosted_to_min(self):
        assert apply_min_angular_command(0.01, 0.05) == pytest.approx(0.05)

    def test_small_negative_w_boosted_preserving_sign(self):
        assert apply_min_angular_command(-0.01, 0.05) == pytest.approx(-0.05)

    def test_zero_min_w_leaves_command_unchanged(self):
        assert apply_min_angular_command(0.01, 0.0) == pytest.approx(0.01)
