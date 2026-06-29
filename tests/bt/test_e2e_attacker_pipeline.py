"""End-to-end attacker pipeline tests for the current supporter-overhaul tree.

These tests intentionally describe the working tree shape we are keeping:

    Snapshot -> AttackerTree -> RobotBlackboard.current_intent

The attacker now chases only useful loose balls, waits high when the ball is
in our own half, dribbles while settling possession, shoots only after a short
settle period, and pass-dribbles before releasing a pass.
"""
from __future__ import annotations

import inspect
import math

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import (
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentPass,
)
from TeamControl.bt.contracts.snapshot import GamePhase, RefereeState, RobotState, Snapshot
from TeamControl.bt.trees.attacker import AttackerTree, SHOT_SETTLE_TICKS


_ATTACKER_ID = 1
_TEAMMATE_ID = 2
_ATTACK_GOAL = (4.5, 0.0)


def _make_snapshot(
    *,
    ball_pos: tuple[float, float],
    attacker_pos: tuple[float, float],
    attacker_orientation: float = 0.0,
    teammates: list[tuple[int, tuple[float, float]]] | None = None,
    enemies: list[tuple[int, tuple[float, float]]] | None = None,
) -> Snapshot:
    own_robots = [
        RobotState(
            robot_id=_ATTACKER_ID,
            position=attacker_pos,
            orientation=attacker_orientation,
        )
    ]
    for robot_id, position in teammates or []:
        own_robots.append(
            RobotState(robot_id=robot_id, position=position, orientation=0.0)
        )

    enemy_robots = [
        RobotState(robot_id=robot_id, position=position, orientation=0.0)
        for robot_id, position in enemies or []
    ]

    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=(0.0, 0.0),
        own_robots=own_robots,
        enemy_robots=enemy_robots,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _make_blackboard() -> RobotBlackboard:
    return RobotBlackboard(robot_id=_ATTACKER_ID, current_role=RoleType.ATTACKER)


def _tick(tree: AttackerTree, snapshot: Snapshot) -> RobotBlackboard:
    bb = _make_blackboard()
    tree.set_snapshot(snapshot)
    tree.tick(bb)
    return bb


def test_enemy_half_loose_ball_is_chased() -> None:
    tree = AttackerTree(us_positive=False)
    snapshot = _make_snapshot(
        ball_pos=(2.0, 0.0),
        attacker_pos=(0.0, -1.0),
    )

    bb = _tick(tree, snapshot)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.current_intent.target_pos == snapshot.ball_position
    assert bb.current_intent.target_orientation == math.atan2(1.0, 2.0)
    assert bb.intent_source == "ChaseBall"


def test_own_half_loose_ball_makes_attacker_wait_high() -> None:
    tree = AttackerTree(us_positive=False)
    snapshot = _make_snapshot(
        ball_pos=(-1.0, 0.5),
        attacker_pos=(2.0, 0.0),
    )

    bb = _tick(tree, snapshot)

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.current_intent.target_pos == (tree.behavior_config.wait_x, 0.5)
    assert bb.intent_source == "WaitNearGoal"


def test_fresh_possession_dribbles_toward_goal_before_shooting() -> None:
    tree = AttackerTree(us_positive=False)
    snapshot = _make_snapshot(
        ball_pos=(3.08, 0.0),
        attacker_pos=(3.0, 0.0),
        attacker_orientation=0.0,
    )

    bb = _tick(tree, snapshot)

    assert isinstance(bb.current_intent, IntentDribble)
    assert bb.current_intent.target_pos == _ATTACK_GOAL
    assert bb.intent_source == "HoldPossession"


def test_settled_clear_possession_shoots_at_goal() -> None:
    tree = AttackerTree(us_positive=False)
    snapshot = _make_snapshot(
        ball_pos=(3.68, 0.0),
        attacker_pos=(3.6, 0.0),
        attacker_orientation=0.0,
    )

    bb = None
    for _ in range(SHOT_SETTLE_TICKS):
        bb = _tick(tree, snapshot)

    assert bb is not None
    assert isinstance(bb.current_intent, IntentKick)
    assert bb.current_intent.target_pos == _ATTACK_GOAL
    assert bb.intent_source == "ShootAtGoal"


def test_blocked_or_pressured_attacker_dribbles_before_passing() -> None:
    tree = AttackerTree(us_positive=False)
    teammate_pos = (3.7, 1.0)
    snapshot = _make_snapshot(
        ball_pos=(3.08, 0.0),
        attacker_pos=(3.0, 0.0),
        attacker_orientation=0.0,
        teammates=[(_TEAMMATE_ID, teammate_pos)],
        enemies=[(8, (3.0, -0.45))],
    )

    bb = _tick(tree, snapshot)

    assert isinstance(bb.current_intent, IntentDribble)
    assert bb.current_intent.target_pos == teammate_pos
    assert bb.intent_source == "DribbleTowardPassTarget"


def test_aligned_pressured_attacker_passes_to_open_teammate() -> None:
    tree = AttackerTree(us_positive=False)
    attacker_pos = (3.0, 0.0)
    teammate_pos = (3.7, 1.0)
    target_angle = math.atan2(
        teammate_pos[1] - attacker_pos[1],
        teammate_pos[0] - attacker_pos[0],
    )
    ball_pos = (
        attacker_pos[0] + math.cos(target_angle) * 0.08,
        attacker_pos[1] + math.sin(target_angle) * 0.08,
    )
    snapshot = _make_snapshot(
        ball_pos=ball_pos,
        attacker_pos=attacker_pos,
        attacker_orientation=target_angle,
        teammates=[(_TEAMMATE_ID, teammate_pos)],
        enemies=[(8, (3.0, -0.45))],
    )

    bb = _tick(tree, snapshot)

    assert isinstance(bb.current_intent, IntentPass)
    assert bb.current_intent.target_robot_id == _TEAMMATE_ID
    assert bb.current_intent.target_pos == teammate_pos
    assert bb.intent_source == "PassToOpenTeammate"


def test_tree_is_reusable_across_snapshots() -> None:
    tree = AttackerTree(us_positive=False)

    first = _tick(
        tree,
        _make_snapshot(ball_pos=(2.0, 0.0), attacker_pos=(0.0, 0.0)),
    )
    second = _tick(
        tree,
        _make_snapshot(ball_pos=(-1.0, 0.0), attacker_pos=(2.0, 0.0)),
    )

    assert isinstance(first.current_intent, IntentMove)
    assert first.intent_source == "ChaseBall"
    assert isinstance(second.current_intent, IntentMove)
    assert second.intent_source == "WaitNearGoal"


def test_snapshot_is_not_mutated_by_tick() -> None:
    tree = AttackerTree(us_positive=False)
    snapshot = _make_snapshot(ball_pos=(2.0, 0.0), attacker_pos=(0.0, 0.0))
    original_ball = snapshot.ball_position
    original_robots = tuple(snapshot.own_robots)

    _tick(tree, snapshot)

    assert snapshot.ball_position == original_ball
    assert tuple(snapshot.own_robots) == original_robots


def test_attacker_source_does_not_emit_robot_commands() -> None:
    import TeamControl.bt.trees.attacker as mod

    source = inspect.getsource(mod)
    assert "RobotCommand" not in source


def test_intents_do_not_contain_robot_command_fields() -> None:
    tree = AttackerTree(us_positive=False)
    snapshot = _make_snapshot(ball_pos=(2.0, 0.0), attacker_pos=(0.0, 0.0))

    bb = _tick(tree, snapshot)

    for field in ("vx", "vy", "vtheta", "kick", "dribbler"):
        assert not hasattr(bb.current_intent, field)
