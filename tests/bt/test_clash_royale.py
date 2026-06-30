"""clash_royale: free a ball wedged between clashing robots."""
from __future__ import annotations

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.coordinator import CLASH_STALL_TICKS, Coordinator
from TeamControl.bt.trees.attacker import AttackerTree
from TeamControl.bt.trees.defender import DefenderTree
from TeamControl.bt.trees.goalie import GoalieTree
from TeamControl.bt.trees.marker import MarkerTree
from TeamControl.bt.trees.supporter import SupporterTree

BASE_ROLES = {
    0: RoleType.GOALIE,
    1: RoleType.ATTACKER,
    2: RoleType.SUPPORTER,
}
IDS = [0, 1, 2]


def _coord(clash_royale: bool = True) -> Coordinator:
    return Coordinator(
        trees={
            RoleType.GOALIE: GoalieTree(us_positive=False),
            RoleType.DEFENDER: DefenderTree(us_positive=False),
            RoleType.SUPPORTER: SupporterTree(us_positive=False),
            RoleType.ATTACKER: AttackerTree(us_positive=False),
            RoleType.MARKER: MarkerTree(us_positive=False),
        },
        us_positive=False,
        role_assignment=BASE_ROLES,
        heuristic_role_swap=False,
        movement_safety=None,
        clash_royale=clash_royale,
    )


def _clash_snapshot() -> Snapshot:
    # Ball wedged: our robot 1 and opponent 10 both touching it, ball stationary.
    return Snapshot(
        ball_position=(0.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(0.1, 0.0), orientation=3.14159),  # on ball
            RobotState(robot_id=2, position=(-2.0, 1.0), orientation=0.0),
        ],
        enemy_robots=[RobotState(robot_id=10, position=(-0.1, 0.0), orientation=0.0)],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_jiggles_contesting_robot_after_stall() -> None:
    coord = _coord(clash_royale=True)
    snap = _clash_snapshot()
    # tick 1 seeds the ball history; stall accrues from tick 2 onward.
    for _ in range(CLASH_STALL_TICKS + 2):
        coord.tick(snap, IDS)

    bb = coord.blackboards[1]
    assert bb.intent_source == "ClashRoyale"
    assert isinstance(bb.current_intent, IntentMove)


def test_no_jiggle_without_opponent_contact() -> None:
    coord = _coord(clash_royale=True)
    # Same stuck ball but the opponent is far away — not a clash, just a parked
    # ball the normal chase should handle.
    snap = Snapshot(
        ball_position=(0.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(0.1, 0.0), orientation=3.14159),
            RobotState(robot_id=2, position=(-2.0, 1.0), orientation=0.0),
        ],
        enemy_robots=[RobotState(robot_id=10, position=(3.0, 2.0), orientation=0.0)],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )
    for _ in range(CLASH_STALL_TICKS + 2):
        coord.tick(snap, IDS)
    assert coord.blackboards[1].intent_source != "ClashRoyale"


def test_disabled_never_jiggles() -> None:
    coord = _coord(clash_royale=False)
    snap = _clash_snapshot()
    for _ in range(CLASH_STALL_TICKS + 5):
        coord.tick(snap, IDS)
    assert coord.blackboards[1].intent_source != "ClashRoyale"
