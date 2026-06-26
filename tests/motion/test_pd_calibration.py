"""
Unit tests for PDCalibration's auto-tune grid sweep (pd_calibration.py).

These don't need grSim or a real robot: FakeRobot plays both pose_source and
dispatch_q, integrating commanded velocity into a fake pose each tick. The
goal is to validate the sweep/selection LOGIC (candidate counts, winner
selection, range narrowing) -- not realistic robot physics, which is already
exercised manually via the PD Calibration page.
"""

import math
import re
from types import SimpleNamespace

import pytest

from TeamControl.robot.motion.controller import RobotMotionController
from TeamControl.robot.motion.pd_calibration import PDCalibration


class FakeRobot:
    """Pose source + dispatch sink: integrates commanded velocity each tick."""

    def __init__(self, pose=(0.0, 0.0, 0.0), dt=0.05):
        self.pose = pose
        self.dt = dt

    def get_robot_pose(self, robot_id, is_yellow):
        return self.pose

    def put(self, item):
        cmd, _runtime = item
        x, y, theta = self.pose
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dx = (cmd.vx * cos_t - cmd.vy * sin_t) * self.dt * 1000.0
        dy = (cmd.vx * sin_t + cmd.vy * cos_t) * self.dt * 1000.0
        self.pose = (x + dx, y + dy, theta + cmd.w * self.dt)


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    """A PDCalibration wired to an isolated settings file and a fake robot."""
    monkeypatch.chdir(tmp_path)
    motion = RobotMotionController(robot_id=99, is_yellow=True)
    robot = FakeRobot()
    return PDCalibration(motion, robot, robot, tick_s=0.01, command_runtime=0.01)


class TestNarrow:
    def test_centers_on_value_with_half_step_spacing(self):
        values = PDCalibration._narrow(1.0, 0.5, n=3, floor=0.0)
        assert values == pytest.approx([0.75, 1.0, 1.25])

    def test_clamps_to_floor(self):
        values = PDCalibration._narrow(0.05, 0.5, n=3, floor=0.05)
        assert min(values) == pytest.approx(0.05)


class TestBest:
    def test_picks_lowest_score(self):
        low = SimpleNamespace(score=1.0)
        high = SimpleNamespace(score=5.0)
        tried = [({"a": 1}, high), ({"a": 2}, low)]

        gains, result = PDCalibration._best(tried)

        assert gains == {"a": 2}
        assert result is low


class TestAutoTuneTurn:
    def test_runs_full_coarse_and_fine_grid(self, calibration):
        result = calibration.auto_tune_turn(
            coarse_kp=(0.5, 1.0, 1.5),
            coarse_kd=(0.0, 0.1, 0.2),
            timeout_s=0.05,
            settle_error_rad=0.0,
            deadline_s=0.2,
        )
        assert len(result.tried) == 9 + 9

    def test_best_result_has_minimum_score(self, calibration):
        result = calibration.auto_tune_turn(timeout_s=0.05, settle_error_rad=0.0)
        assert result.best_result.score == min(r.score for _, r in result.tried)

    def test_returned_gains_include_turn_keys(self, calibration):
        result = calibration.auto_tune_turn(timeout_s=0.05, settle_error_rad=0.0)
        assert "turn_kp" in result.gains
        assert "turn_kd" in result.gains

    def test_logs_a_candidate_line_per_grid_cell(self, calibration):
        # Matches "_grid_sweep"'s "<stage> <idx>/<total>: ..." format exactly,
        # so it doesn't also pick up the unrelated "fine stage did not
        # improve..." recap line (which also happens to start with "fine ").
        candidate_pattern = re.compile(r"^(coarse|fine) \d+/\d+:")
        seen = []
        calibration.auto_tune_turn(
            on_candidate=seen.append, timeout_s=0.05, settle_error_rad=0.0
        )
        candidate_lines = [l for l in seen if candidate_pattern.match(l)]
        assert len(candidate_lines) == 9 + 9


class TestAutoTuneLinear:
    def test_runs_full_coarse_and_fine_grid(self, calibration):
        result = calibration.auto_tune_linear(timeout_s=0.05, settle_error_mm=0.0)
        assert len(result.tried) == 9 + 9

    def test_returned_gains_include_linear_keys(self, calibration):
        result = calibration.auto_tune_linear(timeout_s=0.05, settle_error_mm=0.0)
        assert "linear_kp" in result.gains
        assert "linear_kd" in result.gains
