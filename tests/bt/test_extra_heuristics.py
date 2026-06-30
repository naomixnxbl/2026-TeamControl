"""Tests for the extra per-role heuristics added to the BT configs.

Each new knob must (a) default to the current behaviour and (b) actually take
effect when changed. These tests prove both for the attacker possession
hysteresis + pass-quality gate, the defender challenge distance + midfield
push, the supporter repositioning knobs, and the role-swap last-man / width /
approach-quality additions.
"""
from __future__ import annotations

import pytest

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.tactics.heuristic_role_swap import (
    ApproachQualityWeights,
    DefenderScoreWeights,
    RoleHeuristicWeights,
    RoleTargetCounts,
    SupporterScoreWeights,
    _approach_quality,
    assign_roles_heuristically,
    build_role_swap_context,
    load_role_heuristic_weights,
)
from TeamControl.bt.trees.attacker import (
    AttackerBehaviorConfig,
    AttackerTree,
    _find_best_pass_target,
    load_attacker_behavior_config,
)
from TeamControl.bt.trees.defender import (
    DefenderPositioningConfig,
    DefenderTree,
    _clamp_defensive_target,
    load_defender_positioning_config,
)
from TeamControl.bt.trees.supporter import (
    SupporterBehaviorConfig,
    SupporterTree,
    load_supporter_behavior_config,
)


def _snapshot(ball, own, enemy=()):
    own_robots = tuple(
        RobotState(robot_id=rid, position=pos, orientation=ori)
        for rid, pos, ori in own
    )
    enemy_robots = tuple(
        RobotState(robot_id=rid, position=pos, orientation=0.0)
        for rid, pos in enemy
    )
    return Snapshot(
        ball, (0.0, 0.0), own_robots, enemy_robots,
        RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


# ---------------------------------------------------------------------------
# Config defaults still match the checked-in tuning (new fields included)
# ---------------------------------------------------------------------------


def test_new_config_defaults_are_neutral() -> None:
    atk = load_attacker_behavior_config()
    assert atk.possession_release_dist == atk.possession_dist  # no hysteresis
    assert atk.pass_min_score == 0.0

    sup = load_supporter_behavior_config()
    assert sup.reposition_min_ball_distance == 0.0
    assert sup.reposition_goal_weight == 0.0

    dfn = load_defender_positioning_config()
    assert dfn.challenge_distance == 0.6
    assert dfn.field_margin == 0.2
    assert dfn.allow_cross_midfield_m == 0.0

    weights = load_role_heuristic_weights()
    assert weights.defender.last_man == 0.0
    assert weights.supporter.width == 0.0
    assert weights.approach == ApproachQualityWeights()


# ---------------------------------------------------------------------------
# Attacker — possession hysteresis
# ---------------------------------------------------------------------------


def _tick_possession_distances(release_dist: float) -> int:
    """Tick the attacker with the ball at 0.10 m then 0.15 m; return the
    possession tick count after the second tick (0 = possession dropped)."""
    tree = AttackerTree(
        us_positive=False,
        behavior_config=AttackerBehaviorConfig(
            possession_dist=0.11, possession_release_dist=release_dist
        ),
    )
    bb = RobotBlackboard(robot_id=1, current_role=RoleType.ATTACKER)
    # Robot at origin facing +x (toward the ball) so the heading check passes.
    tree.set_snapshot(_snapshot((0.10, 0.0), own=[(1, (0.0, 0.0), 0.0)]))
    tree.tick(bb)
    tree.set_snapshot(_snapshot((0.15, 0.0), own=[(1, (0.0, 0.0), 0.0)]))
    tree.tick(bb)
    return tree._possession_ticks_by_robot.get(1, 0)


def test_possession_without_hysteresis_drops_at_acquire_distance() -> None:
    # release == acquire (default) → ball at 0.15 > 0.11 drops possession.
    assert _tick_possession_distances(release_dist=0.11) == 0


def test_possession_hysteresis_retains_between_acquire_and_release() -> None:
    # release 0.20 → ball at 0.15 stays controlled (second consecutive tick).
    assert _tick_possession_distances(release_dist=0.20) == 2


# ---------------------------------------------------------------------------
# Attacker — pass quality gate
# ---------------------------------------------------------------------------


def test_pass_min_score_rejects_weak_pass() -> None:
    snap = _snapshot((0.0, 0.0), own=[(1, (0.0, 0.0), 0.0), (2, (1.0, 0.5), 0.0)])
    goal = (4.5, 0.0)

    # Default: an open forward teammate is accepted.
    open_target, _ = _find_best_pass_target(
        snap, 1, goal, AttackerBehaviorConfig(pass_min_score=0.0)
    )
    assert open_target == 2

    # A high threshold rejects even the best available pass.
    blocked_target, blocked_pos = _find_best_pass_target(
        snap, 1, goal, AttackerBehaviorConfig(pass_min_score=999.0)
    )
    assert blocked_target is None
    assert blocked_pos is None


# ---------------------------------------------------------------------------
# Defender — challenge distance + midfield push
# ---------------------------------------------------------------------------


def _defender_intent(challenge_distance: float):
    tree = DefenderTree(
        us_positive=False,
        positioning_config=DefenderPositioningConfig(
            challenge_distance=challenge_distance
        ),
    )
    bb = RobotBlackboard(robot_id=1, current_role=RoleType.DEFENDER)
    # Defender in its own half (x < 0), ball 0.4 m away, no opponents.
    tree.set_snapshot(_snapshot((-1.4, 0.0), own=[(1, (-1.0, 0.0), 0.0)]))
    tree.tick(bb)
    return bb.current_intent


def test_challenge_distance_controls_clearing() -> None:
    # 0.4 m away with a 0.6 m challenge radius → step out and clear.
    assert isinstance(_defender_intent(0.6), IntentKick)
    # Same ball with a tight 0.3 m radius → stay in position (no clear).
    assert isinstance(_defender_intent(0.3), IntentMove)


def test_allow_cross_midfield_relaxes_half_clamp() -> None:
    # us_positive=False → own half is -x, so targets are clamped to x <= cross.
    default_cfg = DefenderPositioningConfig()
    pushed_cfg = DefenderPositioningConfig(allow_cross_midfield_m=1.0)

    clamped = _clamp_defensive_target((0.5, 0.0), us_positive=False, config=default_cfg)
    assert clamped[0] == 0.0  # held at halfway

    pushed = _clamp_defensive_target((0.5, 0.0), us_positive=False, config=pushed_cfg)
    assert pushed[0] == pytest.approx(0.5)  # allowed to step over halfway

    # us_positive=True mirrors the sign.
    pushed_pos = _clamp_defensive_target((-0.5, 0.0), us_positive=True, config=pushed_cfg)
    assert pushed_pos[0] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Supporter — repositioning knobs
# ---------------------------------------------------------------------------


def _reposition_target(behavior_config: SupporterBehaviorConfig):
    tree = SupporterTree(us_positive=False, behavior_config=behavior_config)
    bb = RobotBlackboard(robot_id=2, current_role=RoleType.SUPPORTER)
    # id2 far from the ball and not closest (id1 sits on it) → falls through to
    # RepositionToSpace.
    snap = _snapshot(
        (3.0, 0.0),
        own=[(0, (-4.0, 0.0), 0.0), (1, (2.8, 0.0), 0.0), (2, (0.0, 0.0), 0.0)],
    )
    tree.set_snapshot(snap)
    tree.tick(bb)
    assert bb.intent_source == "RepositionToSpace"
    return bb.current_intent.target_pos


def test_reposition_min_ball_distance_excludes_all_cells_falls_back() -> None:
    # A radius larger than the whole grid skips every cell → fallback (centre).
    target = _reposition_target(
        SupporterBehaviorConfig(reposition_min_ball_distance=100.0)
    )
    assert target == (1.5, 0.0)  # midpoint of the reposition bounds


def test_reposition_goal_weight_pulls_toward_goal() -> None:
    # A huge goal weight makes the supporter pick the grid cell nearest goal.
    target = _reposition_target(SupporterBehaviorConfig(reposition_goal_weight=100.0))
    assert target[0] == pytest.approx(4.0)   # max grid x (closest to opp goal)
    assert target[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Role swap — approach weights, last man, width
# ---------------------------------------------------------------------------


def test_approach_quality_honours_weights() -> None:
    robot = RobotState(robot_id=1, position=(0.0, 0.0), orientation=0.0)
    ball = (0.0, 0.0)
    goal = (4.5, 0.0)

    # distance-only weighting at dist 0 → max score 1.0.
    distance_only = _approach_quality(
        robot, ball, goal, ApproachQualityWeights(behind_ball=0.0, facing=0.0, distance=1.0)
    )
    assert distance_only == pytest.approx(1.0)

    # all-zero weights → zero score, regardless of geometry.
    zeroed = _approach_quality(
        robot, ball, goal, ApproachQualityWeights(behind_ball=0.0, facing=0.0, distance=0.0)
    )
    assert zeroed == 0.0


def test_build_context_records_lateral_offset() -> None:
    snap = _snapshot((0.5, 0.5), own=[(1, (2.0, 1.5), 0.0)])
    ctx = build_role_swap_context(snap, 1, RoleType.SUPPORTER)
    assert ctx.lateral_offset_from_ball == pytest.approx(1.0)  # |1.5 - 0.5|


def test_last_man_weight_makes_deepest_robot_defend() -> None:
    own = [
        RobotState(robot_id=0, position=(-4.2, 0.0), orientation=0.0),  # goalie
        RobotState(robot_id=1, position=(-3.0, 0.0), orientation=0.0),  # deepest
        RobotState(robot_id=2, position=(0.0, 0.0), orientation=0.0),
        RobotState(robot_id=3, position=(2.0, 0.0), orientation=0.0),
        RobotState(robot_id=4, position=(3.0, 0.0), orientation=0.0),
    ]
    snap = _snapshot((0.0, 0.0), own=[(r.robot_id, r.position, r.orientation) for r in own])

    # Defender score driven purely by the last-man bonus; one defender slot.
    weights = RoleHeuristicWeights(
        defender=DefenderScoreWeights(
            own_goal_close=0.0, ball_close=0.0, own_lane=0.0,
            ball_danger=0.0, pressure_escape=0.0, opponent_has_ball=0.0,
            last_man=1.0,
        ),
        role_targets=RoleTargetCounts(
            attackers=0, min_defenders=1, max_defenders=1, min_supporters=1
        ),
    )
    base_roles = {0: RoleType.GOALIE, 1: RoleType.SUPPORTER, 2: RoleType.SUPPORTER,
                  3: RoleType.SUPPORTER, 4: RoleType.SUPPORTER}

    result = assign_roles_heuristically(
        snap,
        [0, 1, 2, 3, 4],
        current_roles=base_roles,
        base_roles=base_roles,
        own_goal=(-4.5, 0.0),
        attack_goal=(4.5, 0.0),
        heuristic_weights=weights,
    )

    # The deepest field robot (id 1, closest to our own goal) becomes defender.
    assert result.roles[1] == RoleType.DEFENDER
