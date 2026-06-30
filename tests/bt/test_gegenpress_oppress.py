"""Oppress structure: shot-zone blocker priority + no shadowing of deep robots.

When the opponent brings the ball into our half, the reactive press should
dedicate one robot to the shot zone (DEFENDER, prioritised) and only man-mark
threats on our side — not opponents sitting back in their own field.
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


def _coord() -> Coordinator:
    return Coordinator(
        trees={
            RoleType.GOALIE: GoalieTree(us_positive=False),
            RoleType.DEFENDER: DefenderTree(us_positive=False),
            RoleType.SUPPORTER: SupporterTree(us_positive=False),
            RoleType.ATTACKER: AttackerTree(
                us_positive=False,
                behavior_config=AttackerBehaviorConfig(press_enabled=True),
            ),
            RoleType.MARKER: MarkerTree(us_positive=False),
        },
        us_positive=False,
        role_assignment=BASE_ROLES,
        heuristic_role_swap=False,
        movement_safety=None,
        gegenpress={"enabled": True, "enter_ticks": 2, "exit_ticks": 5},
    )


def _ball_in_our_half_snapshot() -> Snapshot:
    # us_positive=False → our half is x < 0. Opponent carrier deep in our half,
    # plus a dangerous receiver on our side and a deep build-up robot sitting in
    # their own half (should NOT be shadowed).
    own = [
        RobotState(robot_id=0, position=(-4.2, 0.0), orientation=0.0),   # goalie
        RobotState(robot_id=1, position=(-1.6, 0.2), orientation=0.0),   # near ball
        RobotState(robot_id=2, position=(-3.0, 0.0), orientation=0.0),   # deep → blocker
        RobotState(robot_id=3, position=(-2.0, 1.2), orientation=0.0),
        RobotState(robot_id=4, position=(-2.0, -1.2), orientation=0.0),
        RobotState(robot_id=5, position=(-2.5, 0.5), orientation=0.0),
    ]
    enemies = [
        RobotState(robot_id=10, position=(-1.5, 0.0), orientation=0.0),   # carrier (our half)
        RobotState(robot_id=11, position=(-1.0, 1.5), orientation=0.0),   # threat (our half)
        RobotState(robot_id=12, position=(3.5, 0.0), orientation=0.0),    # deep in THEIR half
    ]
    return Snapshot(
        ball_position=(-1.5, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _roles(coord: Coordinator) -> dict[int, RoleType]:
    return {rid: coord.blackboards[rid].current_role for rid in IDS}


def test_dedicates_shot_zone_blockers_in_our_half() -> None:
    coord = _coord()
    snap = _ball_in_our_half_snapshot()
    for _ in range(3):  # past enter_ticks
        coord.tick(snap, IDS)
    assert coord._gegenpress_active is True

    roles = _roles(coord)
    # One presser, a primary + extra DEFENDER (shot zone), the rest man-mark.
    field = [roles[rid] for rid in (1, 2, 3, 4, 5)]
    assert field.count(RoleType.ATTACKER) == 1
    assert field.count(RoleType.DEFENDER) == 2
    assert field.count(RoleType.MARKER) == 2
    # The deepest field robot (closest to our goal) is the primary blocker, and
    # the next-deepest becomes the orbiting extra defender.
    assert roles[2] == RoleType.DEFENDER
    assert coord._extra_defender_id is not None
    assert roles[coord._extra_defender_id] == RoleType.DEFENDER


def test_extra_defender_orbits_then_tackles_a_frozen_ball() -> None:
    coord = _coord()
    snap = _ball_in_our_half_snapshot()
    # A few ticks in: ball not yet "frozen" → the extra defender shuffles.
    for _ in range(3):
        coord.tick(snap, IDS)
    extra = coord._extra_defender_id
    assert extra is not None
    assert coord.blackboards[extra].intent_source == "OrbitDefenderShuffle"

    # Hold the (stationary) ball long enough and the extra defender pounces.
    from TeamControl.bt.coordinator import FREEZE_TACKLE_TICKS
    for _ in range(FREEZE_TACKLE_TICKS + 2):
        coord.tick(snap, IDS)
    assert coord.blackboards[coord._extra_defender_id].intent_source == "OrbitDefenderTackle"


def test_does_not_shadow_opponents_in_their_own_half() -> None:
    coord = _coord()
    snap = _ball_in_our_half_snapshot()
    for _ in range(3):
        coord.tick(snap, IDS)

    # No marker is assigned to the deep build-up opponent (id 12) sitting back
    # in their own half.
    assigned = {
        coord.blackboards[rid].mark_target_id
        for rid in IDS
        if coord.blackboards[rid].current_role == RoleType.MARKER
    }
    assert 12 not in assigned
