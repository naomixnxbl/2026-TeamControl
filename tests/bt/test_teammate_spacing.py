"""Team-wide spacing guard: our robots stop clustering on each other."""
from __future__ import annotations

import math

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.coordinator import SPACING_MIN_GAP, Coordinator

ROLES = {0: RoleType.GOALIE, 1: RoleType.ATTACKER, 2: RoleType.SUPPORTER, 3: RoleType.SUPPORTER}
IDS = [0, 1, 2, 3]


def _coord() -> Coordinator:
    return Coordinator(trees={}, us_positive=False, role_assignment=ROLES)


def _snap() -> Snapshot:
    # Ball far away (top corner) so no one is the "ball chaser" near the cluster.
    return Snapshot(
        ball_position=(4.0, 2.5),
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(-1.0, 0.0), orientation=0.0),
            RobotState(robot_id=2, position=(-1.0, 0.1), orientation=0.0),
            RobotState(robot_id=3, position=(-1.0, -0.1), orientation=0.0),
        ],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def test_clustered_move_targets_get_separated() -> None:
    coord = _coord()
    snap = _snap()
    coord._ensure_blackboards(snap, IDS)
    # Robots 1, 2, 3 all want the SAME spot — a cluster.
    for rid in (1, 2, 3):
        coord.blackboards[rid].current_intent = IntentMove(
            target_pos=(-1.0, 0.0), target_orientation=0.0
        )

    coord._apply_teammate_spacing(snap, IDS)

    targets = {rid: coord.blackboards[rid].current_intent.target_pos for rid in (1, 2, 3)}
    # At least one pair must now be pulled apart (no longer all coincident).
    pairs = [
        math.hypot(targets[a][0] - targets[b][0], targets[a][1] - targets[b][1])
        for a, b in ((1, 2), (1, 3), (2, 3))
    ]
    assert max(pairs) > 0.2  # the cluster was spread


def test_ball_chaser_and_goalie_are_exempt() -> None:
    coord = _coord()
    # Robot 1 sits on the ball → it's the exempt chaser; 2 and 3 cluster.
    snap = Snapshot(
        ball_position=(4.0, 2.5),
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=0, position=(-4.0, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(4.0, 2.4), orientation=0.0),  # on the ball
            RobotState(robot_id=2, position=(-1.0, 0.1), orientation=0.0),
            RobotState(robot_id=3, position=(-1.0, -0.1), orientation=0.0),
        ],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )
    coord._ensure_blackboards(snap, IDS)
    chaser_target = (4.0, 2.5)
    coord.blackboards[1].current_intent = IntentMove(target_pos=chaser_target, target_orientation=0.0)
    coord.blackboards[2].current_intent = IntentMove(target_pos=(-1.0, 0.0), target_orientation=0.0)
    coord.blackboards[3].current_intent = IntentMove(target_pos=(-1.0, 0.0), target_orientation=0.0)

    coord._apply_teammate_spacing(snap, IDS)

    # The ball chaser's target is untouched.
    assert coord.blackboards[1].current_intent.target_pos == chaser_target


def test_active_ball_plays_are_not_nudged() -> None:
    coord = _coord()
    snap = _snap()
    coord._ensure_blackboards(snap, IDS)
    coord.blackboards[1].current_intent = IntentDribble(target_pos=(-1.0, 0.0))
    coord.blackboards[2].current_intent = IntentMove(target_pos=(-1.0, 0.0), target_orientation=0.0)
    coord.blackboards[3].current_intent = IntentMove(target_pos=(-1.0, 0.0), target_orientation=0.0)

    coord._apply_teammate_spacing(snap, IDS)

    # The dribble intent is left as-is (not a positional move).
    assert isinstance(coord.blackboards[1].current_intent, IntentDribble)
    assert coord.blackboards[1].current_intent.target_pos == (-1.0, 0.0)
