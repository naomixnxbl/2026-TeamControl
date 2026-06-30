"""Tests for GegenPressing containment in the attacker tree (press_enabled)."""
from __future__ import annotations

import math

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.trees.attacker import AttackerBehaviorConfig, AttackerTree

ATTACKER_ID = 1


def _snapshot(
    *,
    ball: tuple[float, float],
    own: list[RobotState],
    enemies: list[RobotState],
) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _press_tree() -> AttackerTree:
    return AttackerTree(
        us_positive=False,
        behavior_config=AttackerBehaviorConfig(press_enabled=True),
    )


def _tick(tree: AttackerTree, snap: Snapshot) -> RobotBlackboard:
    bb = RobotBlackboard(robot_id=ATTACKER_ID, current_role=RoleType.ATTACKER)
    tree.set_snapshot(snap)
    tree.tick(bb)
    return bb


def test_press_branch_absent_by_default() -> None:
    tree = AttackerTree(us_positive=False)  # press_enabled defaults False
    names = [c.name for c in tree.root.children]
    assert "PressContainment" not in names
    assert names == ["PossessionSequence", "ChaseBall"]


def test_press_branch_present_when_enabled() -> None:
    tree = _press_tree()
    names = [c.name for c in tree.root.children]
    assert names == ["PossessionSequence", "PressContainment", "ChaseBall"]


def test_contains_carrier_from_goal_side() -> None:
    attacker = RobotState(robot_id=ATTACKER_ID, position=(-2.0, 0.0), orientation=0.0)
    carrier = RobotState(robot_id=10, position=(0.1, 0.0), orientation=0.0)
    snap = _snapshot(ball=(0.0, 0.0), own=[attacker], enemies=[carrier])

    bb = _tick(_press_tree(), snap)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.intent_source == "PressContainment"
    # us_positive=False → own goal at (-4.5,0). Standoff 0.5 goal-side of carrier.
    tx, ty = bb.current_intent.target_pos
    assert math.isclose(tx, -0.4, abs_tol=1e-6)
    assert math.isclose(ty, 0.0, abs_tol=1e-6)


def test_chases_when_ball_is_loose() -> None:
    attacker = RobotState(robot_id=ATTACKER_ID, position=(-2.0, 0.0), orientation=0.0)
    far_opp = RobotState(robot_id=10, position=(4.0, 2.0), orientation=0.0)
    snap = _snapshot(ball=(0.0, 0.0), own=[attacker], enemies=[far_opp])

    bb = _tick(_press_tree(), snap)

    # No opponent controls the loose ball → fall through to ChaseBall.
    assert bb.intent_source == "ChaseBall"


def test_chases_when_we_are_closer_than_opponent() -> None:
    attacker = RobotState(robot_id=ATTACKER_ID, position=(0.1, 0.0), orientation=0.0)
    opp = RobotState(robot_id=10, position=(0.4, 0.0), orientation=0.0)
    snap = _snapshot(ball=(0.0, 0.0), own=[attacker], enemies=[opp])

    bb = _tick(_press_tree(), snap)

    # We are nearer the ball than the opponent → go win it, don't contain.
    assert bb.intent_source == "ChaseBall"
