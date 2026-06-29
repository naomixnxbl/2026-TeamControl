"""Regression tests for the current supporter-overhaul attacker tree.

These tests intentionally describe the BT we are keeping as the baseline:
possession first, wait near goal when the ball is in our half, otherwise
chase the ball while facing it.
"""
from __future__ import annotations

import math
import py_trees

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.trees.attacker import AttackerTree, SHOT_SETTLE_TICKS


ATTACKER_ID = 5
SUPPORTER_ID = 3


def _snapshot(
    *,
    ball: tuple[float, float],
    own: list[RobotState],
    enemies: list[RobotState] | None = None,
) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=[] if enemies is None else enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _tick(snapshot: Snapshot, *, us_positive: bool = False):
    tree = AttackerTree(us_positive=us_positive)
    bb = RobotBlackboard(robot_id=ATTACKER_ID, current_role=RoleType.ATTACKER)
    tree.set_snapshot(snapshot)
    tree.tick(bb)
    return bb


def test_attacker_topology_matches_overhaul_tree() -> None:
    tree = AttackerTree(us_positive=False)

    assert isinstance(tree.root, py_trees.composites.Selector)
    assert tree.root.name == "AttackingSelector"
    assert [child.name for child in tree.root.children] == [
        "PossessionSequence",
        "WaitSequence",
        "ChaseBall",
    ]


def test_chases_enemy_half_ball_and_faces_it() -> None:
    snap = _snapshot(
        ball=(2.0, 1.0),
        own=[
            RobotState(robot_id=ATTACKER_ID, position=(0.0, 0.0), orientation=0.0),
        ],
    )

    bb = _tick(snap, us_positive=False)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "ChaseBall"
    assert bb.current_intent.target_pos == snap.ball_position
    assert bb.current_intent.target_orientation == math.atan2(1.0, 2.0)


def test_non_closest_attacker_chases_slowly() -> None:
    snap = _snapshot(
        ball=(2.0, 0.0),
        own=[
            RobotState(robot_id=ATTACKER_ID, position=(0.0, 0.0), orientation=0.0),
            RobotState(robot_id=SUPPORTER_ID, position=(1.9, 0.0), orientation=0.0),
        ],
    )

    bb = _tick(snap, us_positive=False)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "ChaseBall"
    assert bb.current_intent.max_speed is not None
    assert bb.current_intent.max_speed < 1.0


def test_waits_near_goal_when_ball_is_in_own_half() -> None:
    snap = _snapshot(
        ball=(-1.0, 1.25),
        own=[
            RobotState(robot_id=ATTACKER_ID, position=(1.5, 0.0), orientation=0.0),
        ],
    )

    bb = _tick(snap, us_positive=False)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "WaitNearGoal"
    assert bb.current_intent.target_pos == (3.5, 1.25)
    assert bb.current_intent.target_orientation == math.atan2(1.25, -2.5)


def test_possession_holds_until_shot_settles_then_shoots() -> None:
    snap = _snapshot(
        ball=(3.08, 0.0),
        own=[
            RobotState(robot_id=ATTACKER_ID, position=(3.0, 0.0), orientation=0.0),
        ],
    )
    tree = AttackerTree(us_positive=False)
    bb = RobotBlackboard(robot_id=ATTACKER_ID, current_role=RoleType.ATTACKER)

    tree.set_snapshot(snap)
    tree.tick(bb)
    assert isinstance(bb.current_intent, IntentDribble)
    assert bb.intent_source == "HoldPossession"

    for _ in range(SHOT_SETTLE_TICKS - 1):
        tree.set_snapshot(snap)
        tree.tick(bb)

    assert isinstance(bb.current_intent, IntentKick)
    assert bb.intent_source == "ShootAtGoal"
    assert bb.current_intent.target_pos == (4.5, 0.0)


def test_tree_outputs_intents_not_robot_commands() -> None:
    snap = _snapshot(
        ball=(2.0, 0.0),
        own=[
            RobotState(robot_id=ATTACKER_ID, position=(0.0, 0.0), orientation=0.0),
        ],
    )

    bb = _tick(snap, us_positive=False)

    assert bb.current_intent is not None
    assert not type(bb.current_intent).__name__.endswith("Command")
