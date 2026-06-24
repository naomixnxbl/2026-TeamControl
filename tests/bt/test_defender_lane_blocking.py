from __future__ import annotations

import math

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.tactics.heuristic_role_swap import assign_roles_heuristically
from TeamControl.bt.trees.defender import (
    DEFENDER_TEAMMATE_MIN_GAP,
    PASS_BLOCK_FRACTION_FROM_CARRIER,
    SHOT_BLOCK_FRACTION_FROM_GOAL,
    DefenderTree,
)


OWN_GOAL = (-4.5, 0.0)
ATTACK_GOAL = (4.5, 0.0)


def _snapshot(
    *,
    ball: tuple[float, float],
    own: list[RobotState],
    opponents: list[RobotState],
) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        opponent_robots=opponents,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _tick_defender(snapshot: Snapshot, robot_id: int):
    tree = DefenderTree(us_positive=False)
    bb = RobotBlackboard(robot_id=robot_id, current_role=RoleType.DEFENDER)
    tree.set_snapshot(snapshot)
    tree.tick(bb)
    return bb.current_intent, bb.intent_source


def _interpolate(
    start: tuple[float, float],
    end: tuple[float, float],
    fraction: float,
) -> tuple[float, float]:
    return (
        start[0] + ((end[0] - start[0]) * fraction),
        start[1] + ((end[1] - start[1]) * fraction),
    )


def _assert_point_close(
    actual: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    assert math.isclose(actual[0], expected[0], abs_tol=1e-6)
    assert math.isclose(actual[1], expected[1], abs_tol=1e-6)


def test_primary_defender_blocks_shot_lane_to_own_goal() -> None:
    carrier = RobotState(robot_id=8, position=(-1.0, 1.0), orientation=0.0)
    snap = _snapshot(
        ball=(-1.05, 1.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(-3.7, 0.4), orientation=0.0),
            RobotState(robot_id=2, position=(-1.5, -2.0), orientation=0.0),
        ],
        opponents=[
            carrier,
            RobotState(robot_id=9, position=(-0.5, -1.5), orientation=0.0),
        ],
    )

    intent, source = _tick_defender(snap, robot_id=1)

    assert isinstance(intent, IntentMove)
    assert source == "BlockShotLane"
    _assert_point_close(
        intent.target_pos,
        _interpolate(OWN_GOAL, carrier.position, SHOT_BLOCK_FRACTION_FROM_GOAL),
    )


def test_secondary_defender_blocks_dangerous_pass_lane() -> None:
    carrier = RobotState(robot_id=8, position=(-1.0, 1.0), orientation=0.0)
    receiver = RobotState(robot_id=9, position=(-0.6, -1.8), orientation=0.0)
    snap = _snapshot(
        ball=(-1.05, 1.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(-4.0, 0.0), orientation=0.0),
            RobotState(robot_id=2, position=(-2.2, -1.0), orientation=0.0),
        ],
        opponents=[carrier, receiver],
    )

    intent, source = _tick_defender(snap, robot_id=2)

    assert isinstance(intent, IntentMove)
    assert source == "BlockPassLane"
    _assert_point_close(
        intent.target_pos,
        _interpolate(
            carrier.position,
            receiver.position,
            PASS_BLOCK_FRACTION_FROM_CARRIER,
        ),
    )


def test_defender_target_keeps_small_gap_from_teammate() -> None:
    carrier = RobotState(robot_id=8, position=(-1.0, 1.0), orientation=0.0)
    raw_block_target = _interpolate(
        OWN_GOAL,
        carrier.position,
        SHOT_BLOCK_FRACTION_FROM_GOAL,
    )
    teammate = RobotState(robot_id=2, position=raw_block_target, orientation=0.0)
    snap = _snapshot(
        ball=(-1.05, 1.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(-3.9, 0.6), orientation=0.0),
            teammate,
        ],
        opponents=[carrier],
    )

    intent, source = _tick_defender(snap, robot_id=1)

    assert isinstance(intent, IntentMove)
    assert source == "BlockShotLane"
    assert intent.target_pos != raw_block_target
    assert (
        math.dist(intent.target_pos, teammate.position)
        >= DEFENDER_TEAMMATE_MIN_GAP - 1e-6
    )


def test_defender_falls_back_to_shape_without_opponent_carrier() -> None:
    robot = RobotState(robot_id=1, position=(-2.5, 0.5), orientation=0.0)
    snap = _snapshot(
        ball=(1.0, 0.0),
        own=[robot],
        opponents=[],
    )

    intent, source = _tick_defender(snap, robot_id=1)

    assert isinstance(intent, IntentMove)
    assert source == "HoldDefendZone"
    assert intent.target_pos == robot.position


def test_close_ball_still_clears_before_lane_defending() -> None:
    snap = _snapshot(
        ball=(-2.55, 0.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(-2.6, 0.0), orientation=0.0),
        ],
        opponents=[
            RobotState(robot_id=8, position=(-2.5, 0.0), orientation=0.0),
        ],
    )

    intent, source = _tick_defender(snap, robot_id=1)

    assert isinstance(intent, IntentKick)
    assert source == "ClearBall"
    assert intent.target_pos == ATTACK_GOAL


def test_role_heuristic_adds_second_defender_when_opponent_controls_ball() -> None:
    current_roles = {
        0: RoleType.GOALIE,
        1: RoleType.DEFENDER,
        2: RoleType.SUPPORTER,
        3: RoleType.SUPPORTER,
        4: RoleType.SUPPORTER,
        5: RoleType.ATTACKER,
    }
    snap = _snapshot(
        ball=(-0.5, 0.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(-3.6, -0.7), orientation=0.0),
            RobotState(robot_id=2, position=(-2.7, 0.8), orientation=0.0),
            RobotState(robot_id=3, position=(-0.8, -1.6), orientation=0.0),
            RobotState(robot_id=4, position=(0.5, 1.2), orientation=0.0),
            RobotState(robot_id=5, position=(1.0, 0.0), orientation=0.0),
        ],
        opponents=[
            RobotState(robot_id=8, position=(-0.45, 0.0), orientation=0.0),
            RobotState(robot_id=9, position=(0.2, 1.8), orientation=0.0),
        ],
    )

    result = assign_roles_heuristically(
        snap,
        [0, 1, 2, 3, 4, 5],
        current_roles,
        base_roles=current_roles,
        own_goal=OWN_GOAL,
        attack_goal=ATTACK_GOAL,
    )

    defenders = [
        robot_id
        for robot_id, role in result.roles.items()
        if role == RoleType.DEFENDER
    ]
    assert len(defenders) == 2
