"""Coordinator integration tests for the current supporter-overhaul BT wiring.

The branch we are keeping uses the static layout below unless heuristic role
swapping is explicitly enabled:

    0 goalie, 1 attacker, 2-5 supporters

These tests verify coordinator dispatch and blackboard state without forcing
older main-branch assumptions back into the working trees.
"""
from __future__ import annotations

import inspect

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.intent import (
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentOrient,
    IntentPass,
    IntentReceive,
)
from TeamControl.bt.contracts.snapshot import GamePhase, RefereeState, RobotState, Snapshot
from TeamControl.bt.coordinator import Coordinator, ROLE_ASSIGNMENT
from TeamControl.bt.trees.attacker import AttackerTree
from TeamControl.bt.trees.defender import DefenderTree
from TeamControl.bt.trees.goalie import GoalieTree
from TeamControl.bt.trees.supporter import SupporterTree


_GOALIE_ID = 0
_ATTACKER_ID = 1
_SUPPORTER_IDS = (2, 3, 4, 5)
_ALL_ROBOT_IDS = [_GOALIE_ID, _ATTACKER_ID, *_SUPPORTER_IDS]
_INTENT_TYPES = (
    IntentMove,
    IntentKick,
    IntentPass,
    IntentDribble,
    IntentReceive,
    IntentOrient,
)


def _make_full_snapshot(
    *,
    ball_pos: tuple[float, float] = (2.0, 0.0),
    attacker_pos: tuple[float, float] = (0.0, 0.0),
    attacker_orientation: float = 0.0,
    enemies: list[tuple[int, tuple[float, float]]] | None = None,
) -> Snapshot:
    own_robots = [
        RobotState(robot_id=_GOALIE_ID, position=(-4.0, 0.0), orientation=0.0),
        RobotState(
            robot_id=_ATTACKER_ID,
            position=attacker_pos,
            orientation=attacker_orientation,
        ),
        RobotState(robot_id=2, position=(0.5, 1.4), orientation=0.0),
        RobotState(robot_id=3, position=(0.5, -1.4), orientation=0.0),
        RobotState(robot_id=4, position=(-1.0, 2.0), orientation=0.0),
        RobotState(robot_id=5, position=(-1.0, -2.0), orientation=0.0),
    ]
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


def _make_coordinator() -> Coordinator:
    return Coordinator(
        trees={
            RoleType.ATTACKER: AttackerTree(us_positive=False),
            RoleType.DEFENDER: DefenderTree(us_positive=False),
            RoleType.SUPPORTER: SupporterTree(us_positive=False),
            RoleType.GOALIE: GoalieTree(us_positive=False),
        },
        us_positive=False,
    )


def test_static_role_assignment_matches_supporter_overhaul_layout() -> None:
    assert ROLE_ASSIGNMENT == {
        0: RoleType.GOALIE,
        1: RoleType.ATTACKER,
        2: RoleType.SUPPORTER,
        3: RoleType.SUPPORTER,
        4: RoleType.SUPPORTER,
        5: RoleType.SUPPORTER,
    }


def test_returns_one_intent_for_each_present_robot() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot()

    intents = coord.tick(snapshot, _ALL_ROBOT_IDS)

    assert len(intents) == len(_ALL_ROBOT_IDS)
    assert all(isinstance(intent, _INTENT_TYPES) for intent in intents)


def test_no_robot_command_fields_in_any_returned_intent() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot()

    intents = coord.tick(snapshot, _ALL_ROBOT_IDS)

    for intent in intents:
        for field in ("vx", "vy", "vtheta", "kick", "dribbler"):
            assert not hasattr(intent, field)


def test_blackboards_are_created_with_static_roles() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot()

    coord.tick(snapshot, _ALL_ROBOT_IDS)

    assert coord.blackboards[_GOALIE_ID].current_role == RoleType.GOALIE
    assert coord.blackboards[_ATTACKER_ID].current_role == RoleType.ATTACKER
    for robot_id in _SUPPORTER_IDS:
        assert coord.blackboards[robot_id].current_role == RoleType.SUPPORTER


def test_attacker_chases_enemy_half_ball_through_coordinator() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot(
        ball_pos=(2.0, 0.0),
        attacker_pos=(0.0, 0.0),
    )

    coord.tick(snapshot, _ALL_ROBOT_IDS)
    bb = coord.blackboards[_ATTACKER_ID]

    assert isinstance(bb.current_intent, IntentMove)
    assert bb.current_intent.target_pos == (2.0, 0.0)
    assert bb.intent_source == "ChaseBall"


def test_attacker_holds_fresh_possession_through_coordinator() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot(
        ball_pos=(2.08, 0.0),
        attacker_pos=(2.0, 0.0),
        attacker_orientation=0.0,
    )

    coord.tick(snapshot, _ALL_ROBOT_IDS)
    bb = coord.blackboards[_ATTACKER_ID]

    assert isinstance(bb.current_intent, IntentDribble)
    assert bb.intent_source == "HoldPossession"


def test_supporters_still_produce_intents() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot()

    coord.tick(snapshot, _ALL_ROBOT_IDS)

    for robot_id in _SUPPORTER_IDS:
        assert coord.blackboards[robot_id].current_intent is not None


def test_missing_robot_ids_are_skipped() -> None:
    coord = _make_coordinator()
    snapshot = Snapshot(
        ball_position=(2.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=_ATTACKER_ID, position=(0.0, 0.0), orientation=0.0),
        ],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )

    intents = coord.tick(snapshot, _ALL_ROBOT_IDS)

    assert len(intents) == 1
    assert _ATTACKER_ID in coord.blackboards
    assert _GOALIE_ID not in coord.blackboards


def test_empty_robot_id_list_returns_no_intents() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot()

    assert coord.tick(snapshot, []) == []


def test_last_intent_is_shifted_on_second_tick() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot()

    coord.tick(snapshot, [_ATTACKER_ID])
    first_intent = coord.blackboards[_ATTACKER_ID].current_intent
    coord.tick(snapshot, [_ATTACKER_ID])

    assert coord.blackboards[_ATTACKER_ID].last_intent == first_intent


def test_separate_blackboard_per_robot() -> None:
    coord = _make_coordinator()
    snapshot = _make_full_snapshot()

    coord.tick(snapshot, [_ATTACKER_ID, _GOALIE_ID])

    assert coord.blackboards[_ATTACKER_ID] is not coord.blackboards[_GOALIE_ID]


def test_tree_and_coordinator_sources_do_not_emit_robot_commands() -> None:
    import TeamControl.bt.coordinator as coordinator_mod
    import TeamControl.bt.trees.attacker as attacker_mod
    import TeamControl.bt.trees.defender as defender_mod
    import TeamControl.bt.trees.goalie as goalie_mod
    import TeamControl.bt.trees.supporter as supporter_mod

    for module in (
        coordinator_mod,
        attacker_mod,
        defender_mod,
        goalie_mod,
        supporter_mod,
    ):
        assert "RobotCommand" not in inspect.getsource(module)
