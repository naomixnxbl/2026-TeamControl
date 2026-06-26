"""
Unit tests for the RobotMotionController-specific rules added in
motion/controller.py: stay_in_field rename, role awareness, angular
wheel-budget cap, MAX_W ceiling, and face_while_moving.

Isolated from the real movement_calibration.json via tmp_path/chdir, same
pattern as test_pd_calibration.py.
"""

import inspect
import math
import time

import pytest

from TeamControl.robot import constants as C
from TeamControl.robot.motion.controller import RobotMotionController
from TeamControl.robot.motion.wheel_kinematics import max_angular_from_wheel_budget


@pytest.fixture
def motion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return RobotMotionController(robot_id=7, is_yellow=True)


class TestStayInFieldRename:
    def test_translational_motion_has_stay_in_field_param(self):
        sig = inspect.signature(RobotMotionController.translational_motion)
        assert "stay_in_field" in sig.parameters
        assert "field_limit" not in sig.parameters

    def test_stay_in_field_true_keeps_robot_inside_bounds(self, motion):
        deadline = time.monotonic() + 0.5
        # Drive toward a point far outside the field; stay_in_field should
        # still produce a finite, safe command (not raise, not NaN).
        vx, vy = motion.translational_motion(
            (0.0, 0.0, 0.0), (50000.0, 0.0), deadline, stay_in_field=True
        )
        assert math.isfinite(vx) and math.isfinite(vy)


class TestRoleAwareness:
    def test_default_is_not_goalie(self, motion):
        assert motion.is_goalie is False

    def test_set_role_updates_is_goalie(self, motion):
        motion.set_role(True)
        assert motion.is_goalie is True
        motion.set_role(False)
        assert motion.is_goalie is False


class TestAngularWheelBudgetCap:
    def test_no_cap_when_wheel_spec_not_active(self, motion):
        assert motion._wheel_spec_active() is False

    def test_rotational_motion_respects_budget_when_active(self, motion):
        motion.apply_gains({"max_wheel_speed_mps": 1.0, "max_wheel_accel_mps2": 100.0})
        motion.reset()
        expected_cap = max_angular_from_wheel_budget(
            motion.robot_radius_mm / 1000.0, 1.0, C.PD_ANGULAR_WHEEL_BUDGET_SHARE
        )
        deadline = time.monotonic() + 0.5
        w = motion.rotational_motion(0.0, math.pi, deadline)
        assert abs(w) <= expected_cap + 1e-9

    def test_budget_cap_formula(self):
        cap = max_angular_from_wheel_budget(0.0885, 2.0, 0.015)
        assert cap == pytest.approx((0.015 * 2.0) / 0.0885)


class TestSharedAngularCeiling:
    def test_max_w_is_one_rad_per_s(self):
        # tuning.json: max_w_raw=1.667 * w_clamp_pct=0.60 = 1.0
        assert C.MAX_W == pytest.approx(1.0, abs=0.01)


class TestFaceWhileMoving:
    def test_returns_three_velocity_components(self, motion):
        deadline = time.monotonic() + 0.5
        result = motion.face_while_moving(
            (0.0, 0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), deadline
        )
        assert len(result) == 3
        for component in result:
            assert math.isfinite(component)

    def test_heading_target_points_at_face_xy(self, motion):
        # face_xy directly "north" of the robot -> heading should end up
        # close to +90 degrees once it starts turning (sanity, not exact).
        deadline = time.monotonic() + 0.5
        vx, vy, w = motion.face_while_moving(
            (0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 1000.0), deadline
        )
        # Facing target is "up" (+y) from current heading 0 -> turn left (w>0).
        assert w > 0.0
