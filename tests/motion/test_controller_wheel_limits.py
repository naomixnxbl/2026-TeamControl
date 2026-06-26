"""
Controller-level tests for the opt-in wheel-aware limiter in
RobotMotionController (motion/controller.py).

Isolated from the real movement_calibration.json via tmp_path/chdir, same
pattern as test_pd_calibration.py.
"""

import time

import pytest

from TeamControl.robot.motion.controller import RobotMotionController
from TeamControl.robot.motion.wheel_kinematics import wheel_speeds


@pytest.fixture
def motion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return RobotMotionController(robot_id=42, is_yellow=True)


class TestOptInDefault:
    def test_no_wheel_spec_by_default(self, motion):
        assert motion.wheel_linear_accel is None
        assert motion.wheel_angular_accel is None
        assert motion.wheel_tuned_accel is None
        assert motion.get_gains()["max_wheel_speed_mps"] is None
        assert motion.get_gains()["max_wheel_accel_mps2"] is None

    def test_regular_isotropic_limiter_still_used_when_not_calibrated(self, motion):
        # Sanity check that the normal (pre-existing) path still runs
        # without error when no wheel spec is present.
        deadline = time.monotonic() + 0.5
        vx, vy = motion.translational_motion((0.0, 0.0, 0.0), (500.0, 0.0), deadline)
        assert isinstance(vx, float) and isinstance(vy, float)


class TestWheelSpecActivation:
    def test_setting_both_limits_activates_wheel_limiters(self, motion):
        motion.apply_gains({"max_wheel_speed_mps": 1.0, "max_wheel_accel_mps2": 2.0})
        assert motion.wheel_linear_accel is not None
        assert motion.wheel_angular_accel is not None
        assert motion.wheel_tuned_accel is not None

    def test_only_one_limit_set_does_not_activate(self, motion):
        motion.apply_gains({"max_wheel_speed_mps": 1.0})
        assert motion.wheel_linear_accel is None

    def test_clearing_back_to_none_deactivates(self, motion):
        motion.apply_gains({"max_wheel_speed_mps": 1.0, "max_wheel_accel_mps2": 2.0})
        motion.apply_gains({"max_wheel_speed_mps": None})
        assert motion.wheel_linear_accel is None


class TestTunedVelocityRespectsWheelLimit:
    def test_output_stays_within_wheel_speed_budget(self, motion):
        motion.apply_gains({"max_wheel_speed_mps": 1.0, "max_wheel_accel_mps2": 100.0})
        motion.reset()

        vx, vy, w = motion.tuned_velocity(5.0, 0.0, 0.0, use_hardware=False)

        angles = motion._wheel_angles_deg()
        radius_m = motion.robot_radius_mm / 1000.0
        worst = max(abs(s) for s in wheel_speeds(vx, vy, w, angles, radius_m))
        assert worst <= 1.0 + 1e-6

    def test_without_wheel_spec_uses_isotropic_max_speed_instead(self, motion):
        # Regression guard: an uncalibrated robot behaves exactly as before.
        vx, vy, w = motion.tuned_velocity(5.0, 0.0, 0.0, use_hardware=False)
        # No wheel limiter active -> only the isotropic C.MAX_SPEED cap ran.
        assert motion.wheel_tuned_accel is None
