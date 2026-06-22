"""Tests for BT adapter unit normalization."""
from __future__ import annotations

from TeamControl.SSL.game_controller.common import GameState
from TeamControl.bt.adapter import (
    DribbleLimitTracker,
    build_snapshot_from_world_model,
    dispatch_coordinator_output,
)
from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentMove
from TeamControl.bt.contracts.snapshot import GamePhase, RefereeState, RobotState, Snapshot


class _Ball:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _Robot:
    def __init__(self, robot_id: int, x: float, y: float, o: float) -> None:
        self.id = robot_id
        self.x = x
        self.y = y
        self.o = o


class _Frame:
    def __init__(self) -> None:
        self.ball = _Ball(1500.0, -2500.0)
        self.robots_yellow = [_Robot(0, 1000.0, 2000.0, 1.25)]
        self.robots_blue = [_Robot(1, -3000.0, 500.0, -0.5)]


class _WorldModel:
    def get_latest_frame(self):
        return _Frame()

    def us_yellow(self):
        return True

    def get_game_state(self):
        return GameState.RUNNING


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def full(self) -> bool:
        return False

    def put(self, item) -> None:
        self.items.append(item)


class _Coordinator:
    def __init__(self, intent) -> None:
        self.blackboards = {
            1: RobotBlackboard(
                robot_id=1,
                current_role=RoleType.ATTACKER,
                current_intent=intent,
            )
        }


def _bt_snapshot() -> Snapshot:
    return Snapshot(
        ball_position=(0.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=(RobotState(robot_id=1, position=(0.0, 0.0), orientation=0.0),),
        opponent_robots=(),
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_build_snapshot_from_world_model_converts_mm_to_m():
    snapshot = build_snapshot_from_world_model(_WorldModel())

    assert snapshot is not None
    assert snapshot.ball_position == (1.5, -2.5)
    assert snapshot.own_robots[0].position == (1.0, 2.0)
    assert snapshot.opponent_robots[0].position == (-3.0, 0.5)
    assert snapshot.own_robots[0].orientation == 1.25
    assert snapshot.referee_state.game_phase == GamePhase.RUNNING


def test_dribble_tracker_allows_up_to_three_seconds():
    tracker = DribbleLimitTracker(max_dribble_seconds=3.0)

    assert tracker.should_enable_dribbler(1, True, now=10.0) is True
    assert tracker.should_enable_dribbler(1, True, now=13.0) is True
    assert tracker.should_enable_dribbler(1, True, now=13.01) is False


def test_dispatch_turns_dribbler_off_after_limit_and_resets():
    snapshot = _bt_snapshot()
    tracker = DribbleLimitTracker(max_dribble_seconds=3.0)
    coordinator = _Coordinator(IntentDribble(target_pos=(1.0, 0.0)))
    queue = _Queue()

    dispatch_coordinator_output(
        coordinator,
        [1],
        snapshot,
        True,
        queue,
        dribble_tracker=tracker,
        now=20.0,
    )
    assert queue.items[-1][0].dribble == 1

    dispatch_coordinator_output(
        coordinator,
        [1],
        snapshot,
        True,
        queue,
        dribble_tracker=tracker,
        now=23.01,
    )
    assert queue.items[-1][0].dribble == 0

    coordinator.blackboards[1].current_intent = IntentMove(
        target_pos=(0.0, 0.0),
        target_orientation=None,
    )
    dispatch_coordinator_output(
        coordinator,
        [1],
        snapshot,
        True,
        queue,
        dribble_tracker=tracker,
        now=23.02,
    )

    coordinator.blackboards[1].current_intent = IntentDribble(target_pos=(1.0, 0.0))
    dispatch_coordinator_output(
        coordinator,
        [1],
        snapshot,
        True,
        queue,
        dribble_tracker=tracker,
        now=23.03,
    )
    assert queue.items[-1][0].dribble == 1
