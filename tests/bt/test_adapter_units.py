"""Tests for BT adapter unit normalization."""
from __future__ import annotations

from TeamControl.SSL.game_controller.common import GameState
from TeamControl.bt.adapter import (
    DribbleLimitTracker,
    build_snapshot_from_world_model,
    dispatch_coordinator_output,
    intent_to_motion_target,
    intent_to_robot_command,
)
from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentKick, IntentMove
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


def _bt_snapshot(
    robot_pos: tuple[float, float] = (0.0, 0.0),
    robot_orientation: float = 0.0,
    ball_pos: tuple[float, float] = (0.0, 0.0),
) -> Snapshot:
    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=(0.0, 0.0),
        own_robots=(
            RobotState(
                robot_id=1,
                position=robot_pos,
                orientation=robot_orientation,
            ),
        ),
        enemy_robots=(),
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_build_snapshot_from_world_model_converts_mm_to_m():
    snapshot = build_snapshot_from_world_model(_WorldModel())

    assert snapshot is not None
    assert snapshot.ball_position == (1.5, -2.5)
    assert snapshot.own_robots[0].position == (1.0, 2.0)
    assert snapshot.enemy_robots[0].position == (-3.0, 0.5)
    assert snapshot.own_robots[0].orientation == 1.25
    assert snapshot.referee_state.game_phase == GamePhase.RUNNING


def test_build_snapshot_filters_inactive_same_colour_robots_as_obstacles():
    class FrameWithInactiveOwn:
        ball = _Ball(0.0, 0.0)
        robots_yellow = [
            _Robot(0, 0.0, 0.0, 0.0),
            _Robot(1, 1000.0, 0.0, 0.0),
            _Robot(2, 2000.0, 0.0, 0.0),
            _Robot(3, 3000.0, 0.0, 0.0),
        ]
        robots_blue = [_Robot(0, -1000.0, 0.0, 0.0)]

    class WorldModelWithInactiveOwn(_WorldModel):
        def get_latest_frame(self):
            return FrameWithInactiveOwn()

    snapshot = build_snapshot_from_world_model(
        WorldModelWithInactiveOwn(),
        is_yellow=True,
        active_robot_ids=[0, 1, 2],
    )

    assert snapshot is not None
    assert [robot.robot_id for robot in snapshot.own_robots] == [0, 1, 2]
    assert {robot.position for robot in snapshot.enemy_robots} == {
        (-1.0, 0.0),
        (3.0, 0.0),
    }


def test_dribble_tracker_allows_up_to_one_metre():
    tracker = DribbleLimitTracker(max_dribble_distance_m=1.0)

    # First call records start position (0, 0)
    assert tracker.should_enable_dribbler(1, True, ball_pos=(0.0, 0.0)) is True
    # Still within 1 m
    assert tracker.should_enable_dribbler(1, True, ball_pos=(0.5, 0.0)) is True
    assert tracker.should_enable_dribbler(1, True, ball_pos=(0.999, 0.0)) is True
    # At/beyond 1 m
    assert tracker.should_enable_dribbler(1, True, ball_pos=(1.001, 0.0)) is False


def test_dispatch_forces_kick_after_limit_and_resets():
    snapshot_start = _bt_snapshot(ball_pos=(0.0, 0.0))
    tracker = DribbleLimitTracker(max_dribble_distance_m=1.0)
    coordinator = _Coordinator(IntentDribble(target_pos=(4.5, 0.0)))
    queue = _Queue()

    # First dispatch — records start position, dribble allowed
    dispatch_coordinator_output(
        coordinator, [1], snapshot_start, True, queue, dribble_tracker=tracker
    )
    assert queue.items[-1][0].dribble == 1

    # Second dispatch after exceeding 1 m — limit fires, dribbler disabled
    snapshot_over = _bt_snapshot(ball_pos=(1.01, 0.0))
    dispatch_coordinator_output(
        coordinator, [1], snapshot_over, True, queue, dribble_tracker=tracker
    )
    assert queue.items[-1][0].dribble == 0

    # Non-dribble intent resets the tracker
    coordinator.blackboards[1].current_intent = IntentMove(
        target_pos=(0.0, 0.0),
        target_orientation=None,
    )
    dispatch_coordinator_output(
        coordinator, [1], snapshot_over, True, queue, dribble_tracker=tracker
    )

    # New dribble from a fresh position — allowed again
    coordinator.blackboards[1].current_intent = IntentDribble(target_pos=(4.5, 0.0))
    snapshot_new = _bt_snapshot(ball_pos=(0.5, 0.0))
    dispatch_coordinator_output(
        coordinator, [1], snapshot_new, True, queue, dribble_tracker=tracker
    )
    assert queue.items[-1][0].dribble == 1


def test_move_to_ball_stops_and_faces_ball_when_misaligned():
    snapshot = _bt_snapshot(
        robot_pos=(0.0, 0.0),
        robot_orientation=0.0,
        ball_pos=(0.0, 0.2),
    )

    target = intent_to_motion_target(
        IntentMove(target_pos=snapshot.ball_position, target_orientation=None),
        1,
        snapshot,
    )

    assert target is not None
    assert target.target_velocity == (0.0, 0.0)
    assert target.target_orientation > 1.5


def test_dribble_faces_and_collects_ball_before_carrying_to_target():
    snapshot = _bt_snapshot(
        robot_pos=(0.0, 0.0),
        robot_orientation=0.0,
        ball_pos=(0.0, 0.2),
    )

    target = intent_to_motion_target(IntentDribble(target_pos=(2.0, 0.0)), 1, snapshot)

    assert target is not None
    assert target.target_velocity == (0.0, 0.0)
    assert target.target_orientation > 1.5


def test_kick_does_not_fire_until_robot_is_aligned_with_ball():
    snapshot = _bt_snapshot(
        robot_pos=(0.0, 0.0),
        robot_orientation=0.0,
        ball_pos=(0.0, 0.15),
    )

    cmd = intent_to_robot_command(IntentKick(target_pos=(4.5, 0.0)), 1, snapshot, True)

    assert cmd is not None
    assert cmd.kick == 0


def test_kick_repositions_behind_ball_when_close_but_not_behind():
    snapshot = _bt_snapshot(
        robot_pos=(0.0, 0.0),
        robot_orientation=0.0,
        ball_pos=(0.0, 0.15),
    )

    target = intent_to_motion_target(IntentKick(target_pos=(4.5, 0.0)), 1, snapshot)

    assert target is not None
    assert target.target_velocity[0] < -0.01
    assert target.target_orientation < 0.0


def test_kick_alignment_uses_stable_target_angle_near_approach_point():
    snapshot = _bt_snapshot(
        robot_pos=(-0.22, 0.06),
        robot_orientation=0.8,
        ball_pos=(0.0, 0.0),
    )

    target = intent_to_motion_target(IntentKick(target_pos=(4.5, 0.0)), 1, snapshot)

    assert target is not None
    assert target.target_velocity == (0.0, 0.0)
    assert abs(target.target_orientation) < 1e-6


def test_kick_moves_into_ball_once_behind_and_aligned():
    snapshot = _bt_snapshot(
        robot_pos=(-0.22, 0.0),
        robot_orientation=0.0,
        ball_pos=(0.0, 0.0),
    )

    target = intent_to_motion_target(IntentKick(target_pos=(4.5, 0.0)), 1, snapshot)

    assert target is not None
    assert target.target_velocity[0] > 0.0
    assert abs(target.target_velocity[1]) < 1e-6
    assert abs(target.target_orientation) < 1e-6


def test_kick_fires_when_robot_is_close_and_aligned():
    snapshot = _bt_snapshot(
        robot_pos=(-0.12, 0.0),
        robot_orientation=0.0,
        ball_pos=(0.0, 0.0),
    )

    cmd = intent_to_robot_command(IntentKick(target_pos=(4.5, 0.0)), 1, snapshot, True)

    assert cmd is not None
    assert cmd.kick == 1
