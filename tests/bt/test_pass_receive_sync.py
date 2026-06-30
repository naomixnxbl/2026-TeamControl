"""Coordinator pass⇄receive sync: the receiver commits to the in-flight ball."""
from __future__ import annotations

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.intent import IntentMove, IntentPass
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.coordinator import Coordinator

ROLES = {0: RoleType.GOALIE, 1: RoleType.ATTACKER, 2: RoleType.SUPPORTER}
IDS = [1, 2]


def _coord() -> Coordinator:
    # No trees needed: _apply_pass_receive_sync reads/writes blackboards only.
    return Coordinator(trees={}, us_positive=False, role_assignment=ROLES)


def _snap(ball, r1, r2) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=1, position=r1, orientation=0.0),
            RobotState(robot_id=2, position=r2, orientation=0.0),
        ],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _latch_pass(coord, snap, target_pos=(2.0, 0.0)):
    coord._ensure_blackboards(snap, IDS)
    coord.blackboards[1].current_intent = IntentPass(target_robot_id=2, target_pos=target_pos)
    coord._apply_pass_receive_sync(snap, IDS)


def test_receiver_holds_reception_point_facing_ball() -> None:
    coord = _coord()
    snap = _snap(ball=(0.0, 0.0), r1=(0.0, 0.0), r2=(2.0, 0.0))  # ball far from receiver
    _latch_pass(coord, snap)

    bb2 = coord.blackboards[2]
    assert bb2.intent_source == "ReceivePass"
    assert isinstance(bb2.current_intent, IntentMove)
    assert bb2.current_intent.target_pos == (2.0, 0.0)  # holds the reception point
    assert coord._incoming_pass is not None


def test_receiver_steps_onto_a_near_ball() -> None:
    coord = _coord()
    _latch_pass(coord, _snap(ball=(0.0, 0.0), r1=(0.0, 0.0), r2=(2.0, 0.0)))
    # Ball now in flight and close to the receiver — it should meet it.
    near = _snap(ball=(1.6, 0.0), r1=(0.3, 0.0), r2=(2.0, 0.0))
    coord.blackboards[1].current_intent = None  # passer already released
    coord._apply_pass_receive_sync(near, IDS)

    bb2 = coord.blackboards[2]
    assert bb2.intent_source == "ReceivePass"
    assert bb2.current_intent.target_pos == (1.6, 0.0)  # steps onto the ball


def test_pass_completes_when_receiver_reaches_ball() -> None:
    coord = _coord()
    _latch_pass(coord, _snap(ball=(0.0, 0.0), r1=(0.0, 0.0), r2=(2.0, 0.0)))
    # Ball arrives at the receiver.
    arrived = _snap(ball=(2.0, 0.0), r1=(0.5, 0.0), r2=(2.0, 0.0))
    coord.blackboards[1].current_intent = None
    coord._apply_pass_receive_sync(arrived, IDS)

    assert coord._incoming_pass is None  # latch released — tree takes over again


def test_no_pass_no_override() -> None:
    coord = _coord()
    snap = _snap(ball=(0.0, 0.0), r1=(0.0, 0.0), r2=(2.0, 0.0))
    coord._ensure_blackboards(snap, IDS)
    coord._apply_pass_receive_sync(snap, IDS)  # nobody is passing
    assert coord._incoming_pass is None
    assert coord.blackboards[2].current_intent is None
