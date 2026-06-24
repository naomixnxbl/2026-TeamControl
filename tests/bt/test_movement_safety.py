from __future__ import annotations

import pytest

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import GamePhase, RefereeState, RobotState, Snapshot
from TeamControl.bt.coordinator import Coordinator
from TeamControl.bt.tactics.rule_following import MovementSafetyConfig


class FixedTree:
    def __init__(self, intent) -> None:
        self.intent = intent
        self.snapshot = None

    def set_snapshot(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot

    def tick(self, blackboard: RobotBlackboard) -> None:
        blackboard.current_intent = self.intent


def _snapshot(
    robot_id: int,
    position: tuple[float, float] = (0.0, 0.0),
    ball: tuple[float, float] = (0.0, 0.0),
    phase: GamePhase = GamePhase.RUNNING,
) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=[
            RobotState(robot_id=robot_id, position=position, orientation=0.0)
        ],
        opponent_robots=[],
        referee_state=RefereeState(game_phase=phase, score=(0, 0)),
    )


def _trees(intent) -> dict[RoleType, FixedTree]:
    return {role: FixedTree(intent) for role in RoleType}


def test_non_goalie_move_target_is_clamped_inside_field() -> None:
    coord = Coordinator(
        trees=_trees(IntentMove(target_pos=(99.0, -99.0), target_orientation=None)),
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=True,
            keep_goalie_in_goal_box=False,
            field_margin=0.10,
        ),
    )

    result = coord.tick(_snapshot(1), [1])

    assert result[0].target_pos == (4.4, -2.9)
    assert coord.blackboards[1].current_intent.target_pos == (4.4, -2.9)


def test_field_clamp_can_be_disabled() -> None:
    coord = Coordinator(
        trees=_trees(IntentMove(target_pos=(99.0, -99.0), target_orientation=None)),
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
        ),
    )

    result = coord.tick(_snapshot(1), [1])

    assert result[0].target_pos == (99.0, -99.0)


def test_goalie_move_target_is_clamped_inside_own_goalie_box() -> None:
    coord = Coordinator(
        trees=_trees(IntentMove(target_pos=(0.0, 2.5), target_orientation=None)),
        us_positive=False,
        role_assignment={0: RoleType.GOALIE},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=True,
            field_margin=0.05,
            goalie_box_depth=1.0,
            goalie_box_width=2.0,
            goalie_box_margin=0.05,
        ),
    )

    result = coord.tick(_snapshot(0, position=(-4.0, 0.0)), [0])

    assert result[0].target_pos == (-3.55, 0.95)


def test_kick_aim_target_is_not_clamped() -> None:
    coord = Coordinator(
        trees=_trees(IntentKick(target_pos=(99.0, 99.0))),
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=True,
            keep_goalie_in_goal_box=True,
        ),
    )

    result = coord.tick(_snapshot(1), [1])

    assert result[0].target_pos == (99.0, 99.0)


def test_non_goalie_target_inside_own_goalie_box_is_moved_out() -> None:
    coord = Coordinator(
        trees=_trees(IntentMove(target_pos=(-4.0, 0.0), target_orientation=None)),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=True,
            goalie_box_avoid_margin=0.15,
        ),
    )

    result = coord.tick(_snapshot(1, position=(-2.0, 0.0)), [1])

    assert result[0].target_pos == (-3.38, 0.0)


def test_non_goalie_route_crossing_goalie_box_is_rerouted_to_corner() -> None:
    coord = Coordinator(
        trees=_trees(IntentMove(target_pos=(-4.4, 1.4), target_orientation=None)),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=True,
            goalie_box_avoid_margin=0.15,
        ),
    )

    result = coord.tick(_snapshot(1, position=(-2.0, 0.0)), [1])

    assert result[0].target_pos == pytest.approx((-3.38, 1.12))


def test_non_goalie_goalie_box_avoidance_can_be_disabled() -> None:
    coord = Coordinator(
        trees=_trees(IntentMove(target_pos=(-4.0, 0.0), target_orientation=None)),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=False,
        ),
    )

    result = coord.tick(_snapshot(1, position=(-2.0, 0.0)), [1])

    assert result[0].target_pos == (-4.0, 0.0)


def test_move_to_ball_inside_opponent_defense_area_stops_at_edge() -> None:
    coord = Coordinator(
        trees=_trees(IntentMove(target_pos=(4.0, 0.0), target_orientation=None)),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=False,
            avoid_ball_touch_in_opponent_defense_area=True,
            goalie_box_avoid_margin=0.15,
        ),
    )

    result = coord.tick(_snapshot(1, position=(2.0, 0.0), ball=(4.0, 0.0)), [1])

    assert isinstance(result[0], IntentMove)
    assert result[0].target_pos == (3.38, 0.0)


def test_kick_is_blocked_when_ball_is_in_opponent_defense_area() -> None:
    coord = Coordinator(
        trees=_trees(IntentKick(target_pos=(4.5, 0.0))),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=False,
            avoid_ball_touch_in_opponent_defense_area=True,
            goalie_box_avoid_margin=0.15,
        ),
    )

    result = coord.tick(_snapshot(1, position=(3.0, 0.0), ball=(4.0, 0.0)), [1])

    assert isinstance(result[0], IntentMove)
    assert result[0].target_pos == (3.38, 0.0)


def test_dribble_toward_opponent_defense_area_is_rerouted_to_edge() -> None:
    coord = Coordinator(
        trees=_trees(IntentDribble(target_pos=(4.5, 0.0))),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=False,
            avoid_ball_touch_in_opponent_defense_area=True,
            goalie_box_avoid_margin=0.15,
        ),
    )

    result = coord.tick(_snapshot(1, position=(2.0, 0.0), ball=(2.0, 0.0)), [1])

    assert isinstance(result[0], IntentDribble)
    assert result[0].target_pos == (3.38, 0.0)


def test_kick_from_inside_opponent_defense_area_moves_out_first() -> None:
    coord = Coordinator(
        trees=_trees(IntentKick(target_pos=(4.5, 0.0))),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=False,
            avoid_ball_touch_in_opponent_defense_area=True,
            goalie_box_avoid_margin=0.15,
        ),
    )

    result = coord.tick(_snapshot(1, position=(4.0, 0.0), ball=(3.0, 0.0)), [1])

    assert isinstance(result[0], IntentMove)
    assert result[0].target_pos == (3.38, 0.0)


def test_opponent_defense_area_touch_guard_can_be_disabled() -> None:
    coord = Coordinator(
        trees=_trees(IntentKick(target_pos=(4.5, 0.0))),
        us_positive=False,
        role_assignment={1: RoleType.ATTACKER},
        movement_safety=MovementSafetyConfig(
            keep_robots_in_bounds=False,
            keep_goalie_in_goal_box=False,
            keep_non_goalies_out_of_goalie_box=False,
            avoid_ball_touch_in_opponent_defense_area=False,
        ),
    )

    result = coord.tick(_snapshot(1, position=(3.0, 0.0), ball=(4.0, 0.0)), [1])

    assert isinstance(result[0], IntentKick)
    assert result[0].target_pos == (4.5, 0.0)
