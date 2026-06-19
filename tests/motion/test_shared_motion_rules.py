"""
Unit tests for the shared motion-rule layer in ball_nav.py -- used by
RobotMotionController, Movement.py, and ball_nav's own move_toward so every
motion controller in the repo follows the same rule set.
"""

import math

import pytest

from TeamControl.robot import ball_nav
from TeamControl.robot.constants import ROBOT_RADIUS
from TeamControl.world.field_config import (
    DEFENCE_X_MM,
    FIELD_X_MIN,
    FIELD_X_MAX,
    GOAL_HALF_WIDTH_MM,
)


class TestFieldGeometryCache:
    def test_first_refresh_reports_changed(self):
        cache = ball_nav.FieldGeometryCache()
        assert cache.refresh() is True

    def test_second_refresh_with_no_change_reports_unchanged(self):
        cache = ball_nav.FieldGeometryCache()
        cache.refresh()
        assert cache.refresh() is False


class TestClampForRole:
    def test_goalie_clamped_to_max_advance_positive_side(self):
        goal_x = 2250.0
        max_advance = float(DEFENCE_X_MM) - 50.0
        x, y = ball_nav.clamp_for_role(
            (0.0, 0.0), is_goalie=True, own_goal_positive_side=True, own_goal_x=goal_x
        )
        assert x == pytest.approx(goal_x - max_advance)
        assert y == 0.0

    def test_goalie_clamped_to_max_advance_negative_side(self):
        goal_x = -2250.0
        max_advance = float(DEFENCE_X_MM) - 50.0
        x, y = ball_nav.clamp_for_role(
            (0.0, 0.0), is_goalie=True, own_goal_positive_side=False, own_goal_x=goal_x
        )
        assert x == pytest.approx(goal_x + max_advance)
        assert y == 0.0

    def test_goalie_near_goal_line_kept_at_margin(self):
        goal_x = 2250.0
        margin = 60.0
        x, _ = ball_nav.clamp_for_role(
            (goal_x, 0.0), is_goalie=True, margin=margin,
            own_goal_positive_side=True, own_goal_x=goal_x,
        )
        assert x == pytest.approx(goal_x - margin)

    def test_non_goalie_pushed_out_of_own_penalty_box(self):
        x_min, x_max = float(FIELD_X_MIN), float(FIELD_X_MAX)
        defence_x = float(DEFENCE_X_MM)
        # Middle of the box, safely inside the margin-inset check used by
        # is_in_penalty_box (a point right at x_min would fail that inset
        # check and not be detected as "in the box" at all).
        deep_in_box = (x_min + defence_x / 2.0, 0.0)
        x, y = ball_nav.clamp_for_role(deep_in_box, is_goalie=False, margin=ROBOT_RADIUS)
        assert x == pytest.approx(x_min + defence_x + ROBOT_RADIUS)
        assert y == 0.0

    def test_non_goalie_outside_box_is_unchanged(self):
        target = (0.0, 0.0)  # field center, nowhere near either box
        assert ball_nav.clamp_for_role(target, is_goalie=False) == target


class TestApplyBoundaryBrakingGoalPostZone:
    def test_zeroes_outward_velocity_in_goal_mouth_zone(self):
        x_max = float(FIELD_X_MAX)
        # Just past the positive end line, well within the goal mouth width.
        pos = (x_max + 50.0, 0.0, 0.0)
        vx, vy = ball_nav.apply_boundary_braking(pos, 1.0, 0.0)
        assert vx <= 0.0  # outward (+x) component zeroed; inward allowed

    def test_unaffected_outside_goal_mouth_width(self):
        x_max = float(FIELD_X_MAX)
        goal_hw = float(GOAL_HALF_WIDTH_MM)
        # Past the end line but well outside the goal mouth -- this is off
        # the field entirely (out-of-bounds crawl applies, not goal-post).
        pos = (x_max + 50.0, goal_hw + ROBOT_RADIUS + 500.0, 0.0)
        vx, vy = ball_nav.apply_boundary_braking(pos, -1.0, 0.0)
        # Out-of-field crawl scales it down but does not zero it outright.
        assert vx < 0.0


class TestRegulateSpeedToTarget:
    def test_unchanged_when_under_budget(self):
        assert ball_nav.regulate_speed_to_target(5000.0, 1.0) == 1.0

    def test_capped_to_sqrt_2ad_when_over_budget(self):
        dist_mm, speed, accel = 10.0, 5.0, 2.105
        result = ball_nav.regulate_speed_to_target(dist_mm, speed, accel)
        assert result == pytest.approx(math.sqrt(2.0 * accel * (dist_mm / 1000.0)))
        assert result < speed

    def test_zero_distance_returns_speed_unchanged(self):
        assert ball_nav.regulate_speed_to_target(0.0, 3.0) == 3.0

    def test_zero_speed_returns_unchanged(self):
        assert ball_nav.regulate_speed_to_target(500.0, 0.0) == 0.0


class TestPredictPosition:
    def test_no_rotation_moves_straight_along_heading(self):
        x, y = ball_nav.predict_position((0.0, 0.0, 0.0), 1.0, 0.0, 0.05)
        assert x == pytest.approx(50.0)  # 1.0 m/s * 0.05s * 1000 (mm)
        assert y == pytest.approx(0.0)

    def test_rotated_heading_projects_into_world_frame(self):
        x, y = ball_nav.predict_position((0.0, 0.0, math.pi / 2), 1.0, 0.0, 0.05)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(50.0)

    def test_zero_dt_is_a_no_op(self):
        x, y = ball_nav.predict_position((100.0, 200.0, 0.3), 1.0, 0.5, 0.0)
        assert (x, y) == pytest.approx((100.0, 200.0))


class TestMoveTowardRegulation:
    def test_regulation_caps_speed_near_target(self):
        # Far enough to be outside the ramp zone but very close in absolute
        # terms, so the never-overshoot cap (not the ramp) is what limits it.
        vx, vy = ball_nav.move_toward((5.0, 0.0), speed=5.0, ramp_dist=1.0, stop_dist=0.0)
        speed = math.hypot(vx, vy)
        assert speed < 5.0

    def test_regulate_false_disables_the_cap(self, monkeypatch):
        # Neutralize calibration.json's speed_scale so this test is
        # deterministic regardless of what's actually saved on disk.
        monkeypatch.setitem(ball_nav._cal, "speed_scale", 1.0)
        vx, vy = ball_nav.move_toward(
            (5.0, 0.0), speed=5.0, ramp_dist=1.0, stop_dist=0.0, regulate=False
        )
        speed = math.hypot(vx, vy)
        assert speed == pytest.approx(5.0)
