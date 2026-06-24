from __future__ import annotations

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.tactics.heuristic_role_swap import (
    AttackerScoreWeights,
    ContextScaleWeights,
    DefenderScoreWeights,
    RoleHeuristicWeights,
    RoleStabilityWeights,
    RoleTargetCounts,
    SupporterScoreWeights,
    assign_roles_heuristically,
)


OWN_GOAL = (-4.5, 0.0)
ATTACK_GOAL = (4.5, 0.0)


def _snapshot(
    *,
    ball: tuple[float, float],
    own: list[RobotState],
    opponents: list[RobotState],
) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        opponent_robots=opponents,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _press_only_weights(*, loose_ball: bool = False) -> RoleHeuristicWeights:
    return RoleHeuristicWeights(
        attacker=AttackerScoreWeights(
            ball_close=0.0,
            approach_quality=0.0,
            angle_score=0.0,
            opponent_goal_close=0.0,
            goal_sight=0.0,
            pressure_escape=0.0,
            own_has_ball=0.0,
            opponent_has_ball_pressure=0.0 if loose_ball else 1.0,
            loose_ball_pressure=1.0 if loose_ball else 0.0,
        ),
        defender=DefenderScoreWeights(
            own_goal_close=0.0,
            ball_close=0.0,
            own_lane=0.0,
            ball_danger=0.0,
            pressure_escape=0.0,
            opponent_has_ball=0.0,
        ),
        supporter=SupporterScoreWeights(
            spacing=0.0,
            opponent_goal_close=0.0,
            pressure_escape=0.0,
            goal_sight=0.0,
            not_crowding_ball=0.0,
            forward_lane=0.0,
            own_has_ball=0.0,
        ),
        stability=RoleStabilityWeights(
            current_role_bias=0.0,
            cooldown_bias=0.0,
            minimum_swap_interval=0.0,
        ),
        context=ContextScaleWeights(
            goal_sight_clearance_field_scale=0.02,
            lane_width_field_scale=0.04,
            pressure_radius_field_scale=0.12,
            possession_radius_field_scale=0.06,
        ),
        role_targets=RoleTargetCounts(
            attackers=1,
            min_defenders=0,
            max_defenders=0,
            min_supporters=0,
        ),
    )


def test_closest_robot_presses_when_opponent_has_ball() -> None:
    roles = {
        0: RoleType.GOALIE,
        1: RoleType.ATTACKER,
        2: RoleType.SUPPORTER,
        3: RoleType.SUPPORTER,
    }
    snap = _snapshot(
        ball=(0.0, 0.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(3.0, 0.0), orientation=0.0),
            RobotState(robot_id=2, position=(0.35, 0.0), orientation=0.0),
            RobotState(robot_id=3, position=(1.5, 1.0), orientation=0.0),
        ],
        opponents=[
            RobotState(robot_id=8, position=(0.05, 0.0), orientation=0.0),
        ],
    )

    result = assign_roles_heuristically(
        snap,
        [0, 1, 2, 3],
        roles,
        base_roles=roles,
        own_goal=OWN_GOAL,
        attack_goal=ATTACK_GOAL,
        heuristic_weights=_press_only_weights(),
    )

    assert result.contexts[2].current_ball_holder == "opponent"
    assert result.roles[2] == RoleType.ATTACKER


def test_closest_robot_claims_loose_ball() -> None:
    roles = {
        0: RoleType.GOALIE,
        1: RoleType.ATTACKER,
        2: RoleType.SUPPORTER,
        3: RoleType.SUPPORTER,
    }
    snap = _snapshot(
        ball=(0.0, 0.0),
        own=[
            RobotState(robot_id=0, position=(-4.3, 0.0), orientation=0.0),
            RobotState(robot_id=1, position=(3.0, 0.0), orientation=0.0),
            RobotState(robot_id=2, position=(0.8, 0.0), orientation=0.0),
            RobotState(robot_id=3, position=(1.4, 1.0), orientation=0.0),
        ],
        opponents=[
            RobotState(robot_id=8, position=(1.2, 0.0), orientation=0.0),
        ],
    )

    result = assign_roles_heuristically(
        snap,
        [0, 1, 2, 3],
        roles,
        base_roles=roles,
        own_goal=OWN_GOAL,
        attack_goal=ATTACK_GOAL,
        heuristic_weights=_press_only_weights(loose_ball=True),
    )

    assert result.contexts[2].current_ball_holder == "none"
    assert result.roles[2] == RoleType.ATTACKER
