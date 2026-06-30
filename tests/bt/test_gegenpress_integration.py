"""End-to-end Coordinator test for the GegenPressing setup.

Mirrors the `gegenpress` UI mode: 1 goalie, 1 pressing attacker, 4 markers,
with the attacker's press_enabled flag on. Drives a full RUNNING tick through
the real trees and checks markers mark and the attacker contains the carrier.
"""
from __future__ import annotations

import dataclasses

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.coordinator import Coordinator
from TeamControl.bt.trees.attacker import AttackerBehaviorConfig, AttackerTree
from TeamControl.bt.trees.defender import DefenderTree
from TeamControl.bt.trees.goalie import GoalieTree
from TeamControl.bt.trees.marker import MarkerTree
from TeamControl.bt.trees.supporter import SupporterTree

GEGENPRESS_ROLES = {
    0: RoleType.GOALIE,
    1: RoleType.ATTACKER,
    2: RoleType.MARKER,
    3: RoleType.MARKER,
    4: RoleType.MARKER,
    5: RoleType.MARKER,
}


def _gegenpress_coordinator(us_positive: bool = False) -> Coordinator:
    attacker = AttackerTree(
        us_positive=us_positive,
        behavior_config=AttackerBehaviorConfig(press_enabled=True),
    )
    return Coordinator(
        trees={
            RoleType.GOALIE: GoalieTree(us_positive=us_positive),
            RoleType.DEFENDER: DefenderTree(us_positive=us_positive),
            RoleType.SUPPORTER: SupporterTree(us_positive=us_positive),
            RoleType.ATTACKER: attacker,
            RoleType.MARKER: MarkerTree(us_positive=us_positive),
        },
        us_positive=us_positive,
        role_assignment=GEGENPRESS_ROLES,
        heuristic_role_swap=False,
        movement_safety=None,
    )


def _running_snapshot() -> Snapshot:
    # us_positive=False → we defend -x, attack +x. Opponent holds the ball at
    # centre; three markable outlets sit in midfield / our half.
    own = [
        RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),  # goalie
        RobotState(robot_id=1, position=(-1.0, 0.0), orientation=0.0),  # presser
        RobotState(robot_id=2, position=(-2.0, 1.0), orientation=0.0),  # marker
        RobotState(robot_id=3, position=(-2.0, -1.0), orientation=0.0),  # marker
        RobotState(robot_id=4, position=(-2.5, 0.5), orientation=0.0),  # marker
        RobotState(robot_id=5, position=(-2.5, -0.5), orientation=0.0),  # marker
    ]
    enemies = [
        RobotState(robot_id=10, position=(0.0, 0.0), orientation=0.0),   # carrier
        RobotState(robot_id=11, position=(-1.0, 1.5), orientation=0.0),  # outlet
        RobotState(robot_id=12, position=(-1.0, -1.5), orientation=0.0),  # outlet
        RobotState(robot_id=13, position=(-0.5, 0.0), orientation=0.0),  # outlet
    ]
    return Snapshot(
        ball_position=(0.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_full_tick_presses_and_marks() -> None:
    coord = _gegenpress_coordinator()
    ids = [0, 1, 2, 3, 4, 5]
    intents = coord.tick(_running_snapshot(), ids)

    # Every robot produced an intent (no crash, no missing role tree).
    assert len(intents) == len(ids)

    # The attacker contains the carrier (opponent controls the ball).
    assert coord.blackboards[1].intent_source == "PressContainment"

    # Markers are doing marking work, and at least one has a real man assigned.
    marker_sources = {
        rid: coord.blackboards[rid].intent_source for rid in (2, 3, 4, 5)
    }
    assert all(src in ("ShadowOpponent", "ZoneCover") for src in marker_sources.values())
    assert any(
        coord.blackboards[rid].mark_target_id is not None for rid in (2, 3, 4, 5)
    )
    # The carrier (10) is never man-marked — the presser handles him.
    assigned = {coord.blackboards[rid].mark_target_id for rid in (2, 3, 4, 5)}
    assert 10 not in assigned


def test_marker_tree_is_dispatched_for_marker_role() -> None:
    coord = _gegenpress_coordinator()
    coord.tick(_running_snapshot(), [0, 1, 2, 3, 4, 5])
    # Robots assigned MARKER keep that role under static assignment.
    for rid in (2, 3, 4, 5):
        assert coord.blackboards[rid].current_role == RoleType.MARKER
