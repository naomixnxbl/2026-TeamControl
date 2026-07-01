"""Interception: when the opponent hogs the ball, the presser wins it back."""
from __future__ import annotations

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.coordinator import (
    CLASH_BALL_CONTACT,
    INTERCEPT_AFTER_TICKS,
    Coordinator,
)
from TeamControl.bt.trees.attacker import AttackerBehaviorConfig, AttackerTree
from TeamControl.bt.trees.defender import DefenderTree
from TeamControl.bt.trees.goalie import GoalieTree
from TeamControl.bt.trees.marker import MarkerTree
from TeamControl.bt.trees.supporter import SupporterTree

ROLES = {0: RoleType.GOALIE, 1: RoleType.ATTACKER, 2: RoleType.SUPPORTER}
IDS = [0, 1, 2]


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
        role_assignment=ROLES,
        heuristic_role_swap=False,
        movement_safety=None,
        gegenpress={"enabled": True, "enter_ticks": 1, "exit_ticks": 99},
    )


def _opp_holds_snapshot(our_attacker_pos=(-0.6, 0.0)) -> Snapshot:
    # Opponent 10 sits on the ball; our robot 1 is nearby but not on it.
    return Snapshot(
        ball_position=(0.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=our_attacker_pos, orientation=0.0),
            RobotState(robot_id=2, position=(-2.0, 1.0), orientation=0.0),
        ],
        enemy_robots=[RobotState(robot_id=10, position=(0.05, 0.0), orientation=0.0)],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_charges_in_after_opponent_holds_too_long() -> None:
    coord = _coord()
    snap = _opp_holds_snapshot(our_attacker_pos=(-0.8, 0.0))  # not in contact yet
    for _ in range(INTERCEPT_AFTER_TICKS + 2):
        coord.tick(snap, IDS)

    bb = coord.blackboards[1]
    assert bb.intent_source == "InterceptCharge"
    assert isinstance(bb.current_intent, IntentMove)
    assert bb.current_intent.target_pos == (0.0, 0.0)  # charging the ball


def test_jiggles_to_knock_loose_when_in_contact() -> None:
    coord = _coord()
    # Our robot is right on the contested ball (within contact distance).
    snap = _opp_holds_snapshot(our_attacker_pos=(CLASH_BALL_CONTACT * 0.5, 0.0))
    for _ in range(INTERCEPT_AFTER_TICKS + 2):
        coord.tick(snap, IDS)

    assert coord.blackboards[1].intent_source == "InterceptJiggle"


def test_no_interception_before_threshold() -> None:
    coord = _coord()
    snap = _opp_holds_snapshot()
    for _ in range(INTERCEPT_AFTER_TICKS // 2):  # not long enough yet
        coord.tick(snap, IDS)
    assert coord.blackboards[1].intent_source not in ("InterceptCharge", "InterceptJiggle")


def test_opp_hold_clock_resets_when_we_get_close() -> None:
    coord = _coord()
    # Build up the opponent-hold clock...
    for _ in range(INTERCEPT_AFTER_TICKS // 2):
        coord.tick(_opp_holds_snapshot(), IDS)
    assert coord._opp_hold_ticks > 0
    # ...then we become nearest to the ball → clock resets.
    ours_closest = Snapshot(
        ball_position=(0.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(0.05, 0.0), orientation=0.0),  # on the ball
            RobotState(robot_id=2, position=(-2.0, 1.0), orientation=0.0),
        ],
        enemy_robots=[RobotState(robot_id=10, position=(1.0, 0.0), orientation=0.0)],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )
    coord.tick(ours_closest, IDS)
    assert coord._opp_hold_ticks == 0
