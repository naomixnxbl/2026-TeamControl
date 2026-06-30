"""Tests for Coordinator marker assignment (zone-flex man-marking)."""
from __future__ import annotations

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.coordinator import Coordinator


def _snapshot(
    *,
    ball: tuple[float, float],
    own: list[RobotState],
    enemies: list[RobotState],
) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=enemies,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _coord(role_assignment: dict[int, RoleType]) -> Coordinator:
    # No trees needed: _apply_marker_assignment reads/writes blackboards only.
    return Coordinator(
        trees={},
        us_positive=False,
        role_assignment=role_assignment,
    )


GEGENPRESS_ROLES = {
    0: RoleType.GOALIE,
    1: RoleType.ATTACKER,
    2: RoleType.MARKER,
    3: RoleType.MARKER,
}


def _assign(coord: Coordinator, snap: Snapshot, ids: list[int]) -> dict[int, int | None]:
    coord._ensure_blackboards(snap, ids)
    coord._apply_marker_assignment(snap, ids)
    return {rid: coord.blackboards[rid].mark_target_id for rid in ids if rid in coord.blackboards}


def test_ball_carrier_is_left_unmarked() -> None:
    # Opponent 10 holds the ball; opponent 11 is a markable outlet.
    own = [
        RobotState(robot_id=2, position=(-2.0, 0.5), orientation=0.0),
        RobotState(robot_id=3, position=(-2.0, -0.5), orientation=0.0),
    ]
    enemies = [
        RobotState(robot_id=10, position=(0.2, 0.0), orientation=0.0),   # carrier
        RobotState(robot_id=11, position=(-1.0, 1.0), orientation=0.0),  # outlet
    ]
    snap = _snapshot(ball=(0.0, 0.0), own=own, enemies=enemies)

    coord = _coord(GEGENPRESS_ROLES)
    targets = _assign(coord, snap, [2, 3])

    assigned = set(t for t in targets.values() if t is not None)
    assert 10 not in assigned                     # carrier never marked
    assert 11 in assigned                          # outlet is marked
    # Nearest free marker to opp 11 is robot 2; robot 3 drops to zone (None).
    assert targets[2] == 11
    assert targets[3] is None


def test_clustered_opponents_get_one_marker_not_two() -> None:
    # Two outlets bunched within a body length of each other: only one is
    # man-marked; the spare marker drops to zone cover instead of converging.
    own = [
        RobotState(robot_id=2, position=(-2.0, 0.0), orientation=0.0),
        RobotState(robot_id=3, position=(-2.0, 0.5), orientation=0.0),
    ]
    enemies = [
        RobotState(robot_id=10, position=(1.2, 0.0), orientation=0.0),   # carrier (far)
        RobotState(robot_id=11, position=(-1.0, 0.0), orientation=0.0),  # cluster
        RobotState(robot_id=12, position=(-1.2, 0.2), orientation=0.0),  # cluster (bunched)
    ]
    snap = _snapshot(ball=(1.0, 0.0), own=own, enemies=enemies)

    coord = _coord(GEGENPRESS_ROLES)
    targets = _assign(coord, snap, [2, 3])

    assigned = [t for t in targets.values() if t is not None]
    # Exactly one of the bunched pair is covered; the other marker zone-covers.
    assert len(assigned) == 1
    assert assigned[0] in (11, 12)
    assert None in targets.values()


def test_assignment_is_sticky_across_ticks() -> None:
    own = [
        RobotState(robot_id=2, position=(-2.0, 1.0), orientation=0.0),
        RobotState(robot_id=3, position=(-2.0, -1.0), orientation=0.0),
    ]
    enemies = [
        RobotState(robot_id=10, position=(4.0, 2.0), orientation=0.0),   # carrier (near ball)
        RobotState(robot_id=11, position=(-1.0, 1.0), orientation=0.0),
        RobotState(robot_id=12, position=(-1.0, -1.0), orientation=0.0),
    ]
    snap = _snapshot(ball=(4.0, 2.0), own=own, enemies=enemies)

    coord = _coord(GEGENPRESS_ROLES)
    first = _assign(coord, snap, [2, 3])
    assert first[2] == 11 and first[3] == 12

    # Swap marker positions so a greedy re-match would flip the pairing.
    own_swapped = [
        RobotState(robot_id=2, position=(-2.0, -1.0), orientation=0.0),
        RobotState(robot_id=3, position=(-2.0, 1.0), orientation=0.0),
    ]
    snap2 = _snapshot(ball=(4.0, 2.0), own=own_swapped, enemies=enemies)
    second = _assign(coord, snap2, [2, 3])

    # Stickiness keeps each marker on its original man (no flicker).
    assert second[2] == 11
    assert second[3] == 12


def test_marker_drops_man_who_leaves_danger_area() -> None:
    own = [RobotState(robot_id=2, position=(-2.0, 0.0), orientation=0.0)]
    enemies = [
        RobotState(robot_id=10, position=(4.0, 2.0), orientation=0.0),   # carrier
        RobotState(robot_id=11, position=(-1.0, 0.0), orientation=0.0),  # markable
    ]
    snap = _snapshot(ball=(4.0, 2.0), own=own, enemies=enemies)

    coord = _coord({0: RoleType.GOALIE, 1: RoleType.ATTACKER, 2: RoleType.MARKER})
    first = _assign(coord, snap, [2])
    assert first[2] == 11

    # Opponent 11 retreats deep toward its own goal (past the danger line).
    enemies_retreat = [
        RobotState(robot_id=10, position=(4.0, 2.0), orientation=0.0),
        RobotState(robot_id=11, position=(4.0, 0.0), orientation=0.0),
    ]
    snap2 = _snapshot(ball=(4.0, 2.0), own=own, enemies=enemies_retreat)
    second = _assign(coord, snap2, [2])

    # Man left the danger area → marker drops to zone cover (None).
    assert second[2] is None


def test_scarce_markers_cover_most_dangerous_first() -> None:
    # One marker, two markable opponents at different depths.
    own = [RobotState(robot_id=2, position=(-2.0, 0.0), orientation=0.0)]
    enemies = [
        RobotState(robot_id=10, position=(4.0, 2.0), orientation=0.0),   # carrier
        RobotState(robot_id=11, position=(-3.0, 0.0), orientation=0.0),  # deep, most dangerous
        RobotState(robot_id=12, position=(2.0, 0.0), orientation=0.0),   # high, less dangerous
    ]
    snap = _snapshot(ball=(4.0, 2.0), own=own, enemies=enemies)

    coord = _coord({0: RoleType.GOALIE, 1: RoleType.ATTACKER, 2: RoleType.MARKER})
    targets = _assign(coord, snap, [2])

    # The opponent nearest OUR goal (11) is the bigger threat and gets the marker.
    assert targets[2] == 11


def test_no_markers_is_a_noop() -> None:
    own = [RobotState(robot_id=1, position=(-2.0, 0.0), orientation=0.0)]
    enemies = [RobotState(robot_id=10, position=(0.0, 0.0), orientation=0.0)]
    snap = _snapshot(ball=(0.0, 0.0), own=own, enemies=enemies)

    coord = _coord({0: RoleType.GOALIE, 1: RoleType.ATTACKER})
    coord._ensure_blackboards(snap, [1])
    # Should not raise and should not invent a marker target.
    coord._apply_marker_assignment(snap, [1])
    assert coord.blackboards[1].mark_target_id is None
