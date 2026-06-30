"""Tests for MotionExecutor — the stateful PD-backed motion layer in adapter.py.

Each test exercises one behaviour so a failing name tells you exactly what broke.
"""
from __future__ import annotations

import math
from unittest import mock

import pytest

from TeamControl.bt.adapter import MotionExecutor, dispatch_coordinator_output
from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import (
    IntentMove,
    IntentOrient,
    IntentReceive,
)
from TeamControl.bt.contracts.snapshot import GamePhase, RefereeState, RobotState, Snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(
    robot_pos: tuple[float, float] = (0.0, 0.0),
    robot_orientation: float = 0.0,
    ball_pos: tuple[float, float] = (10.0, 10.0),
    robot_id: int = 1,
) -> Snapshot:
    """One own robot, ball far away by default so guards don't fire unexpectedly."""
    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=(0.0, 0.0),
        own_robots=(
            RobotState(
                robot_id=robot_id,
                position=robot_pos,
                orientation=robot_orientation,
            ),
        ),
        enemy_robots=(),
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


# ---------------------------------------------------------------------------
# IntentMove — linear velocity
# ---------------------------------------------------------------------------

class TestIntentMoveLinear:
    def test_produces_forward_velocity_toward_target(self):
        # Robot at origin facing east (+x); target 1 m ahead.
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=None),
            1,
            _snapshot(robot_pos=(0.0, 0.0), robot_orientation=0.0),
            True,
        )
        assert cmd is not None
        assert cmd.vx > 0.0
        assert abs(cmd.vy) < 1e-6

    def test_absent_robot_returns_none(self):
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=None),
            99,  # not in snapshot
            _snapshot(robot_id=1),
            True,
        )
        assert cmd is None

    def test_kick_and_dribble_flags_are_zero(self):
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=None),
            1,
            _snapshot(),
            True,
        )
        assert cmd is not None
        assert cmd.kick == 0
        assert cmd.dribble == 0


# ---------------------------------------------------------------------------
# IntentMove — angular velocity
# ---------------------------------------------------------------------------

class TestIntentMoveAngular:
    def test_orientation_target_produces_angular_velocity(self):
        # Robot facing east (0 rad), wants to face north (π/2). Should spin CCW.
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=math.pi / 2),
            1,
            _snapshot(robot_orientation=0.0),
            True,
        )
        assert cmd is not None
        assert cmd.w > 0.0

    def test_no_orientation_target_produces_zero_angular_velocity(self):
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=None),
            1,
            _snapshot(robot_orientation=0.0),
            True,
        )
        assert cmd is not None
        assert cmd.w == 0.0

    def test_negative_orientation_error_produces_negative_angular_velocity(self):
        # Robot facing north (π/2), wants to face east (0). Should spin CW (w < 0).
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=0.0),
            1,
            _snapshot(robot_orientation=math.pi / 2),
            True,
        )
        assert cmd is not None
        assert cmd.w < 0.0


# ---------------------------------------------------------------------------
# D term
# ---------------------------------------------------------------------------

class TestDTerm:
    def test_d_term_brakes_angular_velocity_when_error_shrinks(self):
        # Tick 1 at t=0: angle error = 1.0 rad — P-only (first call, no history).
        # Tick 2 at t=0.1: angle error = 0.5 rad — D term = Kd*(0.5-1.0)/0.1 < 0.
        # Under pure P, w would halve; PD must push it below w1/2.
        executor = MotionExecutor()

        with mock.patch("TeamControl.robot.pd_controller.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            cmd1 = executor.resolve_command(
                IntentOrient(target_orientation=1.0),
                1,
                _snapshot(robot_orientation=0.0),
                True,
            )
            mock_time.monotonic.return_value = 0.1
            cmd2 = executor.resolve_command(
                IntentOrient(target_orientation=1.0),
                1,
                _snapshot(robot_orientation=0.5),  # error halved to 0.5 rad
                True,
            )

        assert cmd1 is not None and cmd2 is not None
        assert cmd1.w > 0.0
        # Pure-P would give w1/2; D braking must push the output below that.
        assert cmd2.w < cmd1.w / 2

    def test_pd_state_persists_across_ticks_for_same_robot(self):
        # After one call the controller must have stored prev_error.
        executor = MotionExecutor()
        executor.resolve_command(
            IntentOrient(target_orientation=1.0),
            1,
            _snapshot(robot_orientation=0.0),
            True,
        )
        assert executor._get_movement(1).angular_pd.prev_error is not None

    def test_separate_robots_have_independent_pd_state(self):
        # Two robots with identical pose/intent on their respective first ticks
        # must produce identical outputs (both P-only, same error).
        executor = MotionExecutor()
        snap_both = Snapshot(
            ball_position=(10.0, 10.0),
            ball_velocity=(0.0, 0.0),
            own_robots=(
                RobotState(robot_id=1, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=2, position=(0.0, 0.0), orientation=0.0),
            ),
            enemy_robots=(),
            referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
        )
        intent = IntentOrient(target_orientation=1.0)
        cmd1 = executor.resolve_command(intent, 1, snap_both, True)
        cmd2 = executor.resolve_command(intent, 2, snap_both, True)
        assert cmd1 is not None and cmd2 is not None
        assert cmd1.w == pytest.approx(cmd2.w)


# ---------------------------------------------------------------------------
# Ball-approach guard
# ---------------------------------------------------------------------------

class TestBallApproachGuard:
    def test_stops_linear_motion_when_close_and_misaligned(self):
        # Robot at origin facing east. Ball is 0.2 m to the north.
        # Heading error ≈ π/2 > 0.45 rad, distance 0.2 m < 0.35 m → full stop.
        executor = MotionExecutor()
        snapshot = _snapshot(
            robot_pos=(0.0, 0.0), robot_orientation=0.0, ball_pos=(0.0, 0.2)
        )
        cmd = executor.resolve_command(
            IntentMove(target_pos=(0.0, 0.2), target_orientation=None),
            1,
            snapshot,
            True,
        )
        assert cmd is not None
        assert cmd.vx == 0.0
        assert cmd.vy == 0.0
        assert cmd.w != 0.0  # turning to face ball

    def test_slows_linear_motion_when_misaligned_but_not_close_enough_to_stop(self):
        # Ball is 0.6 m away (> 0.35 m stop threshold) but heading error is large.
        # Slow guard fires: speed capped to BALL_APPROACH_SLOW_SPEED = 0.45 m/s.
        executor = MotionExecutor()
        snapshot = _snapshot(
            robot_pos=(0.0, 0.0), robot_orientation=0.0, ball_pos=(0.0, 0.6)
        )
        cmd = executor.resolve_command(
            IntentMove(target_pos=(0.0, 0.6), target_orientation=None),
            1,
            snapshot,
            True,
        )
        assert cmd is not None
        assert math.hypot(cmd.vx, cmd.vy) <= 0.45 + 1e-6

    def test_guard_does_not_apply_when_target_is_not_ball(self):
        # Ball is nearby and misaligned, but robot is heading somewhere else.
        # No guard — robot should move at its normal PD speed.
        executor = MotionExecutor()
        snapshot = _snapshot(
            robot_pos=(0.0, 0.0), robot_orientation=0.0, ball_pos=(0.0, 0.2)
        )
        cmd = executor.resolve_command(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=None),
            1,
            snapshot,
            True,
        )
        assert cmd is not None
        assert cmd.vx > 0.0  # moving normally toward (1, 0)


# ---------------------------------------------------------------------------
# Other intent types
# ---------------------------------------------------------------------------

class TestOtherIntents:
    def test_intent_orient_produces_only_angular_velocity(self):
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentOrient(target_orientation=math.pi / 2),
            1,
            _snapshot(robot_orientation=0.0),
            True,
        )
        assert cmd is not None
        assert cmd.vx == 0.0
        assert cmd.vy == 0.0
        assert cmd.w > 0.0

    def test_intent_orient_negative_error_spins_cw(self):
        executor = MotionExecutor()
        cmd = executor.resolve_command(
            IntentOrient(target_orientation=0.0),
            1,
            _snapshot(robot_orientation=math.pi / 2),
            True,
        )
        assert cmd is not None
        assert cmd.w < 0.0

    def test_intent_receive_produces_all_zero_velocity(self):
        executor = MotionExecutor()
        cmd = executor.resolve_command(IntentReceive(), 1, _snapshot(), True)
        assert cmd is not None
        assert cmd.vx == 0.0
        assert cmd.vy == 0.0
        assert cmd.w == 0.0


# ---------------------------------------------------------------------------
# Integration with dispatch_coordinator_output
# ---------------------------------------------------------------------------

class _FakeCoordinator:
    def __init__(self, intent):
        self.blackboards = {
            1: RobotBlackboard(
                robot_id=1,
                current_role=RoleType.ATTACKER,
                current_intent=intent,
            )
        }


class _FakeQueue:
    def __init__(self):
        self.items = []

    def full(self):
        return False

    def put(self, item):
        self.items.append(item)


class TestDispatchWithExecutor:
    def test_dispatch_routes_intent_move_through_pd_executor(self):
        executor = MotionExecutor()
        snapshot = _snapshot(robot_pos=(0.0, 0.0), robot_orientation=0.0)
        coordinator = _FakeCoordinator(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=None)
        )
        queue = _FakeQueue()

        dispatch_coordinator_output(
            coordinator, [1], snapshot, True, queue, executor=executor
        )

        assert len(queue.items) == 1
        cmd = queue.items[0][0]
        assert cmd.vx > 0.0

    def test_dispatch_without_executor_uses_legacy_path(self):
        # Passing no executor should still work (backward compat).
        snapshot = _snapshot(robot_pos=(0.0, 0.0), robot_orientation=0.0)
        coordinator = _FakeCoordinator(
            IntentMove(target_pos=(1.0, 0.0), target_orientation=None)
        )
        queue = _FakeQueue()

        dispatch_coordinator_output(
            coordinator, [1], snapshot, True, queue
        )

        assert len(queue.items) == 1
