"""Regression tests for the current supporter-overhaul defender tree."""
from __future__ import annotations

import math
import py_trees
import pytest

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.trees.defender import DefenderTree


DEFENDER_ID = 1


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
    tree = DefenderTree(us_positive=us_positive)
    bb = RobotBlackboard(robot_id=DEFENDER_ID, current_role=RoleType.DEFENDER)
    tree.set_snapshot(snapshot)
    tree.tick(bb)
    return bb


def test_defender_topology_matches_overhaul_tree() -> None:
    tree = DefenderTree(us_positive=False)

    assert isinstance(tree.root, py_trees.composites.Sequence)
    assert tree.root.name == "DefendingSequenceNode"
    assert [child.name for child in tree.root.children] == [
        "LookAtBall",
        "DefendZoneFallback",
        "ChallengeSequence",
    ]
    assert [child.name for child in tree.root.children[1].children] == [
        "InDefendingZone",
        "GoToDefendZone",
    ]


def test_holds_current_defensive_shape_when_no_carrier() -> None:
    robot = RobotState(robot_id=DEFENDER_ID, position=(-2.5, 0.75), orientation=0.0)
    snap = _snapshot(ball=(1.0, 0.0), own=[robot])

    bb = _tick(snap, us_positive=False)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "HoldDefendZone"
    assert bb.current_intent.target_pos == robot.position
    assert bb.current_intent.target_orientation == math.atan2(-0.75, 3.5)


def test_out_of_half_returns_to_defend_zone() -> None:
    snap = _snapshot(
        ball=(2.5, 0.0),
        own=[
            RobotState(robot_id=DEFENDER_ID, position=(0.4, 0.2), orientation=0.0),
        ],
    )

    bb = _tick(snap, us_positive=False)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "GoToDefendZone"
    assert bb.current_intent.target_pos == (-3.0, 0.0)


def test_close_ball_clear_overrides_positioning() -> None:
    snap = _snapshot(
        ball=(-2.55, 0.0),
        own=[
            RobotState(robot_id=DEFENDER_ID, position=(-2.6, 0.0), orientation=0.0),
        ],
    )

    bb = _tick(snap, us_positive=False)

    assert isinstance(bb.current_intent, IntentKick)
    assert bb.intent_source == "ClearBall"
    assert bb.current_intent.target_pos == (4.5, 0.0)


def test_primary_defender_blocks_goal_lane_when_enemy_has_ball() -> None:
    carrier = RobotState(robot_id=8, position=(-1.0, 1.0), orientation=0.0)
    snap = _snapshot(
        ball=(-1.05, 1.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=DEFENDER_ID, position=(-3.7, 0.4), orientation=0.0),
            RobotState(robot_id=2, position=(-1.5, -2.0), orientation=0.0),
        ],
        enemies=[carrier],
    )

    bb = _tick(snap, us_positive=False)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "BlockShotLane"
    assert bb.current_intent.target_pos == pytest.approx((-2.575, 0.55))


def test_tree_outputs_intents_not_robot_commands() -> None:
    snap = _snapshot(
        ball=(1.0, 0.0),
        own=[
            RobotState(robot_id=DEFENDER_ID, position=(-2.0, 0.0), orientation=0.0),
        ],
    )

    bb = _tick(snap, us_positive=False)

    assert bb.current_intent is not None
    assert not type(bb.current_intent).__name__.endswith("Command")
