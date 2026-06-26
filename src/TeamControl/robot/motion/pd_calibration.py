"""
Small PD calibration tests for grSim or real robots.

This file does not talk directly to vision/grSim/robot UDP. Instead it uses:

- pose_source.get_robot_pose(robot_id, is_yellow) -> (x, y, theta) or None
- command_sink.send(RobotCommand)

That keeps the same tests usable for grSim and real robots.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.motion.controller import RobotMotionController


def _wrap_angle(a: float) -> float:
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class CalibrationResult:
    test_name: str
    passed: bool
    score: float
    start_pose: tuple[float, float, float]
    final_pose: tuple[float, float, float]
    target_xy: Optional[tuple[float, float]]
    target_theta: Optional[float]
    elapsed_s: float
    final_position_error_mm: float
    final_heading_error_rad: float
    max_position_error_mm: float
    max_heading_error_rad: float
    samples: int


@dataclass
class AutoTuneResult:
    """Outcome of a coarse-to-fine grid sweep over one (kp, kd) gain pair."""
    gains: dict
    best_result: CalibrationResult
    tried: list[tuple[dict, CalibrationResult]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


class PDCalibration:
    def __init__(
        self,
        motion: RobotMotionController,
        pose_source,
        dispatch_q,
        is_yellow: Optional[bool] = None,
        tick_s: float = 0.05,
        command_runtime: float = 0.15,
    ):
        self.motion = motion
        self.pose_source = pose_source
        self.dispatch_q = dispatch_q
        self.is_yellow = motion.is_yellow if is_yellow is None else bool(is_yellow)
        self.tick_s = tick_s
        self.command_runtime = command_runtime

    def _get_pose(self):
        return self.pose_source.get_robot_pose(self.motion.robot_id, self.is_yellow)

    def _send(self, vx: float, vy: float, w: float) -> None:
        cmd = RobotCommand(
            self.motion.robot_id,
            vx,
            vy,
            w,
            0,
            0,
            isYellow=self.is_yellow,
        )
        self.dispatch_q.put((cmd, self.command_runtime))

    def stop(self) -> None:
        self._send(0.0, 0.0, 0.0)
        self.motion.reset()

    def _wait_for_pose(self, timeout_s: float = 1.0):
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            pose = self._get_pose()
            if pose is not None:
                return pose
            time.sleep(self.tick_s)
        raise RuntimeError("No robot pose available for calibration")

    def run_angular_turn_test(
        self,
        angle_rad: float = math.pi / 2,
        gains: Optional[dict] = None,
        use_pd: bool = True,
        use_hardware: bool = False,
        timeout_s: float = 3.0,
        settle_error_rad: float = 0.08,
        deadline_s: float = 0.5,
    ) -> CalibrationResult:
        """Rotate in place by angle_rad and score final heading error."""
        if gains is not None:
            self.motion.apply_gains(gains)

        self.motion.reset()
        start_pose = self._wait_for_pose()
        target_theta = _wrap_angle(start_pose[2] + angle_rad)

        max_heading_error = 0.0
        samples = 0
        t0 = time.monotonic()
        end = t0 + timeout_s

        try:
            while time.monotonic() < end:
                pose = self._get_pose()
                if pose is None:
                    time.sleep(self.tick_s)
                    continue

                err = abs(_wrap_angle(target_theta - pose[2]))
                max_heading_error = max(max_heading_error, err)
                samples += 1

                if err < settle_error_rad:
                    break

                deadline = time.monotonic() + deadline_s
                w = self.motion.rotational_motion(
                    pose[2],
                    target_theta,
                    deadline,
                    use_pd=use_pd,
                    use_hardware=use_hardware,
                )
                self._send(0.0, 0.0, w)
                time.sleep(self.tick_s)
        finally:
            self.stop()

        final_pose = self._wait_for_pose()
        elapsed = time.monotonic() - t0
        final_heading_error = abs(_wrap_angle(target_theta - final_pose[2]))
        score = final_heading_error * 200.0 + elapsed * 10.0

        return CalibrationResult(
            test_name="angular_turn",
            passed=final_heading_error < settle_error_rad,
            score=score,
            start_pose=start_pose,
            final_pose=final_pose,
            target_xy=None,
            target_theta=target_theta,
            elapsed_s=elapsed,
            final_position_error_mm=0.0,
            final_heading_error_rad=final_heading_error,
            max_position_error_mm=0.0,
            max_heading_error_rad=max_heading_error,
            samples=samples,
        )

    def run_linear_forward_test(
        self,
        distance_mm: float = 1000.0,
        gains: Optional[dict] = None,
        use_pd: bool = True,
        use_hardware: bool = False,
        timeout_s: float = 4.0,
        settle_error_mm: float = 100.0,
        deadline_s: float = 0.8,
    ) -> CalibrationResult:
        """Drive forward from the current heading by distance_mm, with w=0."""
        if gains is not None:
            self.motion.apply_gains(gains)

        self.motion.reset()
        start_pose = self._wait_for_pose()
        target_xy = (
            start_pose[0] + distance_mm * math.cos(start_pose[2]),
            start_pose[1] + distance_mm * math.sin(start_pose[2]),
        )
        target_theta = start_pose[2]

        max_pos_error = 0.0
        max_heading_error = 0.0
        samples = 0
        t0 = time.monotonic()
        end = t0 + timeout_s

        try:
            while time.monotonic() < end:
                pose = self._get_pose()
                if pose is None:
                    time.sleep(self.tick_s)
                    continue

                pos_error = math.hypot(target_xy[0] - pose[0], target_xy[1] - pose[1])
                heading_error = abs(_wrap_angle(target_theta - pose[2]))
                max_pos_error = max(max_pos_error, pos_error)
                max_heading_error = max(max_heading_error, heading_error)
                samples += 1

                if pos_error < settle_error_mm:
                    break

                deadline = time.monotonic() + deadline_s
                vx, vy = self.motion.translational_motion(
                    pose,
                    target_xy,
                    deadline,
                    use_pd=use_pd,
                    use_hardware=use_hardware,
                )
                self._send(vx, vy, 0.0)
                time.sleep(self.tick_s)
        finally:
            self.stop()

        final_pose = self._wait_for_pose()
        elapsed = time.monotonic() - t0
        final_pos_error = math.hypot(
            target_xy[0] - final_pose[0],
            target_xy[1] - final_pose[1],
        )
        final_heading_error = abs(_wrap_angle(target_theta - final_pose[2]))
        score = final_pos_error + final_heading_error * 200.0 + elapsed * 10.0

        return CalibrationResult(
            test_name="linear_forward",
            passed=final_pos_error < settle_error_mm,
            score=score,
            start_pose=start_pose,
            final_pose=final_pose,
            target_xy=target_xy,
            target_theta=target_theta,
            elapsed_s=elapsed,
            final_position_error_mm=final_pos_error,
            final_heading_error_rad=final_heading_error,
            max_position_error_mm=max_pos_error,
            max_heading_error_rad=max_heading_error,
            samples=samples,
        )

    def save_result(self, result, gains):
        """
        Save gains to settings store using the result score.
        """
        return self.motion.calibrate(gains, score=result.score)

    # ════════════════════════════════════════════════════════════════
    #  AUTO-TUNE — coarse-to-fine grid sweep over one (kp, kd) pair
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _best(tried: list[tuple[dict, CalibrationResult]]) -> tuple[dict, CalibrationResult]:
        """Pick the candidate with the lowest score (lower is better)."""
        gains, result = min(tried, key=lambda t: t[1].score)
        return gains, result

    @staticmethod
    def _narrow(center: float, step: float, n: int = 3, floor: float = 0.0) -> list[float]:
        """Build a finer candidate range centered on *center*.

        Spaces *n* values half of *step* apart, clamped to *floor* so gains
        never go negative (or below some sane minimum).
        """
        half = step / 2.0
        mid = (n - 1) / 2.0
        return [max(center + (i - mid) * half, floor) for i in range(n)]

    def _grid_sweep(
        self,
        test_fn: Callable[..., CalibrationResult],
        kp_key: str,
        kd_key: str,
        kp_values: list[float],
        kd_values: list[float],
        stage_name: str,
        on_candidate: Optional[Callable[[str], None]] = None,
        **test_kwargs,
    ) -> list[tuple[dict, CalibrationResult]]:
        """Run *test_fn* once per (kp, kd) candidate in the grid, logging each."""
        tried: list[tuple[dict, CalibrationResult]] = []
        total = len(kp_values) * len(kd_values)
        idx = 0
        for kp in kp_values:
            for kd in kd_values:
                idx += 1
                gains = {kp_key: kp, kd_key: kd}
                result = test_fn(gains=gains, **test_kwargs)
                tried.append((gains, result))
                line = (
                    f"{stage_name} {idx}/{total}: "
                    f"{kp_key}={kp:.4g} {kd_key}={kd:.4g} -> score={result.score:.2f}"
                )
                if on_candidate is not None:
                    on_candidate(line)
        return tried

    @staticmethod
    def _recap_lines(
        tried: list[tuple[dict, CalibrationResult]],
        best_gains: dict,
        stage_name: str,
    ) -> list[str]:
        """Format every candidate in a stage, marking the winner."""
        lines = [f"--- {stage_name} stage results ---"]
        for gains, result in tried:
            marker = " <-- BEST" if gains is best_gains else ""
            parts = ", ".join(f"{k}={v:.4g}" for k, v in gains.items())
            lines.append(f"  {parts} -> score={result.score:.2f}{marker}")
        return lines

    def _auto_tune(
        self,
        kind: str,
        test_fn: Callable[..., CalibrationResult],
        kp_key: str,
        kd_key: str,
        coarse_kp: tuple[float, float, float],
        coarse_kd: tuple[float, float, float],
        kp_floor: float,
        kd_floor: float,
        on_candidate: Optional[Callable[[str], None]] = None,
        **test_kwargs,
    ) -> AutoTuneResult:
        """Shared coarse-to-fine grid sweep used by both turn and linear tuning."""
        log: list[str] = []

        def emit(line: str) -> None:
            log.append(line)
            if on_candidate is not None:
                on_candidate(line)

        coarse_tried = self._grid_sweep(
            test_fn, kp_key, kd_key, list(coarse_kp), list(coarse_kd),
            "coarse", emit, **test_kwargs,
        )
        coarse_gains, coarse_result = self._best(coarse_tried)
        for line in self._recap_lines(coarse_tried, coarse_gains, "coarse"):
            emit(line)

        kp_step = (coarse_kp[1] - coarse_kp[0]) if len(coarse_kp) > 1 else coarse_kp[0]
        kd_step = (coarse_kd[1] - coarse_kd[0]) if len(coarse_kd) > 1 else coarse_kd[0]
        fine_kp = self._narrow(coarse_gains[kp_key], kp_step, floor=kp_floor)
        fine_kd = self._narrow(coarse_gains[kd_key], kd_step, floor=kd_floor)

        fine_tried = self._grid_sweep(
            test_fn, kp_key, kd_key, fine_kp, fine_kd,
            "fine", emit, **test_kwargs,
        )
        fine_gains, fine_result = self._best(fine_tried)
        for line in self._recap_lines(fine_tried, fine_gains, "fine"):
            emit(line)

        if fine_result.score < coarse_result.score:
            best_gains, best_result = fine_gains, fine_result
        else:
            best_gains, best_result = coarse_gains, coarse_result
            emit("fine stage did not improve, keeping coarse result")

        # Re-apply the winner so the saved gains reflect the full current
        # state (other gains the sweep didn't touch stay as they were).
        full_gains = self.motion.apply_gains(best_gains)
        self.save_result(best_result, full_gains)
        emit(
            f"Auto-tune {kind} done: {kp_key}={full_gains[kp_key]:.4g} "
            f"{kd_key}={full_gains[kd_key]:.4g} score={best_result.score:.2f} (saved)"
        )

        return AutoTuneResult(
            gains=full_gains,
            best_result=best_result,
            tried=coarse_tried + fine_tried,
            log=log,
        )

    def auto_tune_turn(
        self,
        on_candidate: Optional[Callable[[str], None]] = None,
        coarse_kp: tuple[float, float, float] = (0.5, 1.0, 1.5),
        coarse_kd: tuple[float, float, float] = (0.0, 0.1, 0.2),
        **test_kwargs,
    ) -> AutoTuneResult:
        """Coarse-to-fine grid sweep over (turn_kp, turn_kd) via the angular turn test."""
        return self._auto_tune(
            "turn",
            self.run_angular_turn_test,
            "turn_kp",
            "turn_kd",
            coarse_kp,
            coarse_kd,
            kp_floor=0.05,
            kd_floor=0.0,
            on_candidate=on_candidate,
            **test_kwargs,
        )

    def auto_tune_linear(
        self,
        on_candidate: Optional[Callable[[str], None]] = None,
        coarse_kp: tuple[float, float, float] = (0.001, 0.002, 0.004),
        coarse_kd: tuple[float, float, float] = (0.0, 0.0005, 0.001),
        **test_kwargs,
    ) -> AutoTuneResult:
        """Coarse-to-fine grid sweep over (linear_kp, linear_kd) via the linear forward test."""
        return self._auto_tune(
            "linear",
            self.run_linear_forward_test,
            "linear_kp",
            "linear_kd",
            coarse_kp,
            coarse_kd,
            kp_floor=0.0002,
            kd_floor=0.0,
            on_candidate=on_candidate,
            **test_kwargs,
        )
