"""Tests for the reactive GegenPressing trigger in the Coordinator.

The team plays its base roles (here: btv2-style goalie + attacker + supporters)
and only switches to presser + markers once the opponent has *held* the ball
for `enter_ticks`, reverting after `exit_ticks` of regained possession.
"""
from __future__ import annotations

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

BASE_ROLES = {
    0: RoleType.GOALIE,
    1: RoleType.ATTACKER,
    2: RoleType.SUPPORTER,
    3: RoleType.SUPPORTER,
    4: RoleType.SUPPORTER,
    5: RoleType.SUPPORTER,
}
IDS = [0, 1, 2, 3, 4, 5]


def _coord(gegenpress: dict) -> Coordinator:
    attacker = AttackerTree(
        us_positive=False,
        behavior_config=AttackerBehaviorConfig(press_enabled=True),
    )
    return Coordinator(
        trees={
            RoleType.GOALIE: GoalieTree(us_positive=False),
            RoleType.DEFENDER: DefenderTree(us_positive=False),
            RoleType.SUPPORTER: SupporterTree(us_positive=False),
            RoleType.ATTACKER: attacker,
            RoleType.MARKER: MarkerTree(us_positive=False),
        },
        us_positive=False,
        role_assignment=BASE_ROLES,
        heuristic_role_swap=False,
        movement_safety=None,
        gegenpress=gegenpress,
    )


def _own_robots() -> list[RobotState]:
    return [
        RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),   # goalie
        RobotState(robot_id=1, position=(-0.5, 0.0), orientation=0.0),   # nearest ball
        RobotState(robot_id=2, position=(-2.0, 1.0), orientation=0.0),
        RobotState(robot_id=3, position=(-2.0, -1.0), orientation=0.0),
        RobotState(robot_id=4, position=(-2.5, 0.5), orientation=0.0),
        RobotState(robot_id=5, position=(-2.5, -0.5), orientation=0.0),
    ]


def _opp_control_snapshot() -> Snapshot:
    enemies = [
        RobotState(robot_id=10, position=(0.5, 0.0), orientation=0.0),   # carrier on ball
        RobotState(robot_id=11, position=(-1.0, 1.2), orientation=0.0),
        RobotState(robot_id=12, position=(-1.0, -1.2), orientation=0.0),
    ]
    return Snapshot(
        ball_position=(0.5, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=_own_robots(),
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _own_control_snapshot() -> Snapshot:
    own = _own_robots()
    own[1] = RobotState(robot_id=1, position=(0.0, 0.0), orientation=0.0)  # on the ball
    enemies = [RobotState(robot_id=10, position=(4.0, 2.0), orientation=0.0)]  # far
    return Snapshot(
        ball_position=(0.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _stopped_snapshot() -> Snapshot:
    return Snapshot(
        ball_position=(0.5, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=_own_robots(),
        enemy_robots=[RobotState(robot_id=10, position=(0.5, 0.0), orientation=0.0)],
        referee_state=RefereeState(game_phase=GamePhase.STOPPED, score=(0, 0)),
    )


def _roles(coord: Coordinator) -> dict[int, RoleType]:
    return {rid: coord.blackboards[rid].current_role for rid in IDS}


def test_does_not_engage_before_debounce() -> None:
    coord = _coord({"enabled": True, "enter_ticks": 3, "exit_ticks": 3})
    snap = _opp_control_snapshot()
    for _ in range(2):  # fewer than enter_ticks
        coord.tick(snap, IDS)
    assert coord._gegenpress_active is False
    # Still in base shape — supporters have NOT become markers.
    assert _roles(coord)[2] == RoleType.SUPPORTER


def test_engages_after_sustained_opponent_possession() -> None:
    coord = _coord({"enabled": True, "enter_ticks": 3, "exit_ticks": 3})
    snap = _opp_control_snapshot()
    for _ in range(3):
        coord.tick(snap, IDS)
    assert coord._gegenpress_active is True

    roles = _roles(coord)
    assert roles[0] == RoleType.GOALIE
    # Exactly one presser; the rest of the field man-marks.
    field = [roles[rid] for rid in (1, 2, 3, 4, 5)]
    assert field.count(RoleType.ATTACKER) == 1
    assert field.count(RoleType.MARKER) == 4
    # The presser is the robot nearest the ball (robot 1 at (-0.5, 0)).
    assert roles[1] == RoleType.ATTACKER


def test_disengages_after_regaining_possession() -> None:
    coord = _coord({"enabled": True, "enter_ticks": 3, "exit_ticks": 3})
    for _ in range(3):
        coord.tick(_opp_control_snapshot(), IDS)
    assert coord._gegenpress_active is True

    for _ in range(3):
        coord.tick(_own_control_snapshot(), IDS)
    assert coord._gegenpress_active is False
    # Back to base roles.
    assert _roles(coord)[2] == RoleType.SUPPORTER


def test_phase_change_resets_the_press() -> None:
    coord = _coord({"enabled": True, "enter_ticks": 3, "exit_ticks": 3})
    for _ in range(3):
        coord.tick(_opp_control_snapshot(), IDS)
    assert coord._gegenpress_active is True

    # A stoppage resets the trigger; it must re-accumulate afterwards.
    coord.tick(_stopped_snapshot(), IDS)
    assert coord._gegenpress_active is False

    coord.tick(_opp_control_snapshot(), IDS)  # only one RUNNING tick
    assert coord._gegenpress_active is False


def test_secure_possession_breaks_press_instantly_into_counter_attack() -> None:
    # exit_ticks is huge to prove the break is INSTANT on secure possession,
    # not waiting out the normal release debounce.
    coord = _coord({"enabled": True, "enter_ticks": 3, "exit_ticks": 999})
    for _ in range(3):
        coord.tick(_opp_control_snapshot(), IDS)
    assert coord._gegenpress_active is True

    coord.tick(_own_control_snapshot(), IDS)  # a single tick with the ball secured
    assert coord._gegenpress_active is False

    roles = _roles(coord)
    field = [roles[rid] for rid in (1, 2, 3, 4, 5)]
    # Carrier attacks; everyone else supports (breaks forward) — no marking,
    # no defenders clustering in our half.
    assert field.count(RoleType.ATTACKER) == 1
    assert RoleType.MARKER not in field
    assert RoleType.DEFENDER not in field
    assert all(r in (RoleType.ATTACKER, RoleType.SUPPORTER) for r in field)


def test_disabled_trigger_keeps_base_roles() -> None:
    coord = _coord({"enabled": False})
    for _ in range(10):
        coord.tick(_opp_control_snapshot(), IDS)
    assert coord._gegenpress_active is False
    assert _roles(coord)[2] == RoleType.SUPPORTER
