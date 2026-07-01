"""MARKER as a first-class heuristic role (role_swap.marker / role_targets.markers)."""
from __future__ import annotations

from collections import Counter

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.tactics.heuristic_role_swap import (
    RoleHeuristicWeights,
    RoleTargetCounts,
    assign_roles_heuristically,
)

BASE = {
    0: RoleType.GOALIE,
    1: RoleType.ATTACKER,
    2: RoleType.SUPPORTER,
    3: RoleType.SUPPORTER,
    4: RoleType.SUPPORTER,
    5: RoleType.SUPPORTER,
}
IDS = [0, 1, 2, 3, 4, 5]


def _own():
    return [
        RobotState(0, (-4.0, 0.0), 0.0),
        RobotState(1, (-1.5, 0.0), 0.0),
        RobotState(2, (-1.2, 0.8), 0.0),
        RobotState(3, (-1.2, -0.8), 0.0),
        RobotState(4, (-2.0, 0.5), 0.0),
        RobotState(5, (-2.0, -0.5), 0.0),
    ]


def _defending_snapshot() -> Snapshot:
    # Opponent holds the ball in our half (we attack +x, own goal at -4.5).
    enemies = [
        RobotState(10, (-1.4, 0.0), 0.0),   # carrier in our half
        RobotState(11, (-1.1, 0.9), 0.0),   # outlet near robot 2
        RobotState(12, (-1.1, -0.9), 0.0),  # outlet near robot 3
    ]
    return Snapshot(
        ball_position=(-1.4, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=_own(),
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _attacking_snapshot() -> Snapshot:
    # We hold the ball deep in their half — no marking should happen.
    enemies = [RobotState(10, (4.0, 1.5), 0.0)]
    own = _own()
    own[1] = RobotState(1, (3.0, 0.0), 0.0)  # our attacker on the ball
    return Snapshot(
        ball_position=(3.0, 0.0),
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _assign(snap, weights):
    return assign_roles_heuristically(
        snap, IDS, BASE, base_roles=BASE,
        attack_goal=(4.5, 0.0), own_goal=(-4.5, 0.0),
        heuristic_weights=weights,
    ).roles


def test_marker_off_by_default() -> None:
    roles = _assign(_defending_snapshot(), RoleHeuristicWeights())  # markers=0
    assert RoleType.MARKER not in roles.values()


def test_marker_assigned_when_defending_and_enabled() -> None:
    weights = RoleHeuristicWeights(role_targets=RoleTargetCounts(markers=2))
    counts = Counter(r.name for r in _assign(_defending_snapshot(), weights).values())
    assert counts["MARKER"] >= 1
    # Cap respected: at least one supporter is preserved.
    assert counts["SUPPORTER"] >= 1


def test_no_markers_when_attacking_even_if_enabled() -> None:
    # Markers are a defending role — when WE have the ball in their half, none.
    weights = RoleHeuristicWeights(role_targets=RoleTargetCounts(markers=2))
    roles = _assign(_attacking_snapshot(), weights)
    assert RoleType.MARKER not in roles.values()
