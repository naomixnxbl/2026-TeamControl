"""Focused attacker possession balance tests."""
from __future__ import annotations

import math

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentKick, IntentPass
from TeamControl.bt.contracts.snapshot import GamePhase, RefereeState, RobotState, Snapshot
from TeamControl.bt.trees.attacker import AttackerTree, SHOT_SETTLE_TICKS

_ATTACKER_ID = 5
_SUPPORTER_ID = 3


def _tick_possession_case(
    own_robots: list[RobotState],
    enemy_robots: list[RobotState],
    ball_position: tuple[float, float] = (3.08, 0.0),
):
    snap = Snapshot(
        ball_position=ball_position,
        ball_velocity=(0.0, 0.0),
        own_robots=own_robots,
        enemy_robots=enemy_robots,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )
    tree = AttackerTree(us_positive=False)
    bb = RobotBlackboard(robot_id=_ATTACKER_ID, current_role=RoleType.ATTACKER)
    tree.set_snapshot(snap)
    tree.tick(bb)
    return bb.current_intent


def test_clear_close_shot_still_shoots_before_passing():
    own_robots = [
        RobotState(robot_id=_ATTACKER_ID, position=(3.0, 0.0), orientation=0.0),
        RobotState(robot_id=_SUPPORTER_ID, position=(3.4, 1.2), orientation=0.0),
    ]
    snap = Snapshot(
        ball_position=(3.08, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=own_robots,
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )
    tree = AttackerTree(us_positive=False)
    bb = RobotBlackboard(robot_id=_ATTACKER_ID, current_role=RoleType.ATTACKER)

    tree.set_snapshot(snap)
    tree.tick(bb)
    assert isinstance(bb.current_intent, IntentDribble)

    for _ in range(SHOT_SETTLE_TICKS - 1):
        tree.set_snapshot(snap)
        tree.tick(bb)

    assert isinstance(bb.current_intent, IntentKick)


def test_blocked_shot_with_open_teammate_dribbles_to_face_pass_first():
    intent = _tick_possession_case(
        own_robots=[
            RobotState(robot_id=_ATTACKER_ID, position=(3.0, 0.0), orientation=0.0),
            RobotState(robot_id=_SUPPORTER_ID, position=(3.4, 1.2), orientation=0.0),
        ],
        enemy_robots=[
            RobotState(robot_id=10, position=(3.8, 0.0), orientation=0.0),
        ],
    )

    assert isinstance(intent, IntentDribble)
    assert intent.target_pos == (3.4, 1.2)


def test_blocked_shot_with_open_teammate_passes_once_aligned():
    target_angle = math.atan2(1.2, 0.4)
    ball_position = (
        3.0 + 0.08 * math.cos(target_angle),
        0.0 + 0.08 * math.sin(target_angle),
    )
    intent = _tick_possession_case(
        own_robots=[
            RobotState(
                robot_id=_ATTACKER_ID,
                position=(3.0, 0.0),
                orientation=target_angle,
            ),
            RobotState(robot_id=_SUPPORTER_ID, position=(3.4, 1.2), orientation=0.0),
        ],
        enemy_robots=[
            RobotState(robot_id=10, position=(3.8, 0.0), orientation=0.0),
        ],
        ball_position=ball_position,
    )

    assert isinstance(intent, IntentPass)
    assert intent.target_robot_id == _SUPPORTER_ID


def test_blocked_shot_without_open_teammate_keeps_possession():
    intent = _tick_possession_case(
        own_robots=[
            RobotState(robot_id=_ATTACKER_ID, position=(3.0, 0.0), orientation=0.0),
        ],
        enemy_robots=[
            RobotState(robot_id=10, position=(3.8, 0.0), orientation=0.0),
        ],
    )

    assert isinstance(intent, IntentDribble)
