"""Tests for the team-strategy layer (tactics/strategy.py).

The two invariants this layer must uphold:

1. The checked-in ``bt_tuning.yaml`` leaves strategy DISABLED and neutral, so
   the team behaves exactly as before (``load_strategy_config()`` equals the
   all-default ``StrategyConfig()``).
2. While the strategy is inactive, every ``apply_strategy_*`` transform is the
   identity — even if dials are set to non-neutral values.

The rest of the tests exercise the actual rewrites once the layer is enabled.
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
from TeamControl.bt.tactics.heuristic_role_swap import (
    RoleHeuristicWeights,
    load_role_heuristic_weights,
)
from TeamControl.bt.tactics.strategy import (
    AttackingStrategy,
    DefendingStrategy,
    FormationStrategy,
    GameContext,
    PostureDials,
    RoleBiasStrategy,
    RuleConditions,
    StabilityStrategy,
    StrategyConfig,
    StrategyRule,
    apply_strategy_to_attacker_config,
    apply_strategy_to_defender_positioning,
    apply_strategy_to_role_weights,
    apply_strategy_to_supporter_config,
    evaluate_game_context,
    is_strategy_active,
    load_strategy_config,
    resolve_effective_strategy,
)
from TeamControl.bt.trees.attacker import (
    AttackerBehaviorConfig,
    AttackerTree,
    load_attacker_behavior_config,
)
from TeamControl.bt.trees.defender import (
    DefenderPositioningConfig,
    DefenderTree,
    load_defender_positioning_config,
)
from TeamControl.bt.trees.goalie import GoalieTree
from TeamControl.bt.trees.supporter import (
    SupporterBehaviorConfig,
    SupporterTree,
    load_supporter_behavior_config,
)


# ---------------------------------------------------------------------------
# Invariant 1 — checked-in defaults are disabled + neutral
# ---------------------------------------------------------------------------


def test_checked_in_bt_tuning_disables_strategy() -> None:
    strategy = load_strategy_config()

    assert strategy == StrategyConfig()
    assert strategy.enabled is False
    assert is_strategy_active(strategy) is False


def test_default_strategy_config_is_neutral() -> None:
    strategy = StrategyConfig()

    assert strategy.role_bias == RoleBiasStrategy(1.0, 1.0, 1.0, 1.0)
    assert strategy.formation == FormationStrategy(0, 0, 0)
    assert strategy.stability == StabilityStrategy(1.0)
    assert strategy.attacking == AttackingStrategy(1.0, 1.0, 1.0, 1.0)
    assert strategy.defending == DefendingStrategy(1.0)


# ---------------------------------------------------------------------------
# Invariant 2 — inactive strategy is identity, even with non-neutral dials
# ---------------------------------------------------------------------------


def test_disabled_strategy_is_identity_even_with_dials_set() -> None:
    loud = StrategyConfig(
        enabled=False,
        role_bias=RoleBiasStrategy(attacker_weight_scale=5.0, press_scale=3.0),
        formation=FormationStrategy(extra_attackers=2, extra_defenders=2),
        stability=StabilityStrategy(role_stickiness_scale=4.0),
        attacking=AttackingStrategy(shoot_distance_scale=9.0, settle_time_scale=0.1),
        defending=DefendingStrategy(line_height_scale=1.7),
    )
    weights = RoleHeuristicWeights()
    attacker = AttackerBehaviorConfig()
    supporter = SupporterBehaviorConfig()
    defender = DefenderPositioningConfig()

    assert apply_strategy_to_role_weights(weights, loud) == weights
    assert apply_strategy_to_attacker_config(attacker, loud) == attacker
    assert apply_strategy_to_supporter_config(supporter, loud) == supporter
    assert apply_strategy_to_defender_positioning(defender, loud) == defender


def test_none_strategy_is_identity() -> None:
    weights = RoleHeuristicWeights()
    assert apply_strategy_to_role_weights(weights, None) == weights
    assert is_strategy_active(None) is False


def test_enabled_but_neutral_strategy_is_identity() -> None:
    neutral_on = StrategyConfig(enabled=True)
    weights = RoleHeuristicWeights()
    attacker = AttackerBehaviorConfig()

    assert is_strategy_active(neutral_on) is True
    assert apply_strategy_to_role_weights(weights, neutral_on) == weights
    assert apply_strategy_to_attacker_config(attacker, neutral_on) == attacker


# ---------------------------------------------------------------------------
# Active transforms — role heuristic weights
# ---------------------------------------------------------------------------


def test_attacker_weight_and_press_scale() -> None:
    strategy = StrategyConfig(
        enabled=True,
        role_bias=RoleBiasStrategy(attacker_weight_scale=2.0, press_scale=2.0),
    )
    base = RoleHeuristicWeights()

    out = apply_strategy_to_role_weights(base, strategy)

    # Non-press attacker term scaled once (×2).
    assert out.attacker.goal_sight == base.attacker.goal_sight * 2.0
    # Press term scaled by weight_scale ×2 then press_scale ×2 = ×4.
    assert out.attacker.ball_close == base.attacker.ball_close * 4.0
    assert (
        out.attacker.opponent_has_ball_pressure
        == base.attacker.opponent_has_ball_pressure * 4.0
    )
    # Other roles untouched.
    assert out.supporter == base.supporter


def test_formation_offsets_role_targets() -> None:
    strategy = StrategyConfig(
        enabled=True,
        formation=FormationStrategy(
            extra_attackers=1, extra_defenders=1, extra_supporters=2
        ),
    )
    base = RoleHeuristicWeights()

    out = apply_strategy_to_role_weights(base, strategy)

    assert out.role_targets.attackers == base.role_targets.attackers + 1
    assert out.role_targets.min_defenders == base.role_targets.min_defenders + 1
    assert out.role_targets.max_defenders == base.role_targets.max_defenders + 1
    assert out.role_targets.min_supporters == base.role_targets.min_supporters + 2


def test_role_stickiness_scales_stability_terms() -> None:
    strategy = StrategyConfig(
        enabled=True, stability=StabilityStrategy(role_stickiness_scale=2.0)
    )
    base = RoleHeuristicWeights()

    out = apply_strategy_to_role_weights(base, strategy)

    assert out.stability.current_role_bias == base.stability.current_role_bias * 2.0
    assert out.stability.minimum_swap_interval == base.stability.minimum_swap_interval * 2.0
    assert out.defender_stability.min_hold_seconds == base.defender_stability.min_hold_seconds * 2.0
    assert out.defender_stability.stay_bias == base.defender_stability.stay_bias * 2.0


# ---------------------------------------------------------------------------
# Active transforms — tree behavior configs
# ---------------------------------------------------------------------------


def test_attacking_dials_rewrite_attacker_config() -> None:
    strategy = StrategyConfig(
        enabled=True,
        attacking=AttackingStrategy(
            shoot_distance_scale=1.5,
            shot_alignment_tolerance_scale=2.0,
            settle_time_scale=0.5,
            pass_caution_scale=2.0,
        ),
    )
    base = AttackerBehaviorConfig()

    out = apply_strategy_to_attacker_config(base, strategy)

    assert out.shoot_dist_threshold == base.shoot_dist_threshold * 1.5
    assert out.shot_heading_tol == base.shot_heading_tol * 2.0
    assert out.shot_settle_ticks == max(1, round(base.shot_settle_ticks * 0.5))
    assert out.pass_marked_distance_frac == base.pass_marked_distance_frac * 2.0
    assert out.pass_lane_clearance_frac == base.pass_lane_clearance_frac * 2.0
    # Untouched fields stay put.
    assert out.possession_dist == base.possession_dist


def test_attacking_dials_rewrite_supporter_config() -> None:
    strategy = StrategyConfig(
        enabled=True,
        attacking=AttackingStrategy(shoot_distance_scale=0.5, pass_caution_scale=1.5),
    )
    base = SupporterBehaviorConfig()

    out = apply_strategy_to_supporter_config(base, strategy)

    assert out.shoot_dist_threshold == base.shoot_dist_threshold * 0.5
    assert out.marked_threshold == base.marked_threshold * 1.5
    assert out.grid_step == base.grid_step


def test_line_height_scale_clamps_defender_fractions() -> None:
    strategy = StrategyConfig(
        enabled=True, defending=DefendingStrategy(line_height_scale=10.0)
    )
    base = DefenderPositioningConfig()

    out = apply_strategy_to_defender_positioning(base, strategy)

    # Scaled but clamped to the 0.95 ceiling.
    assert out.shot_block_fraction_from_goal == 0.95
    assert out.pass_block_fraction_from_carrier == 0.95
    # Non-fraction fields untouched.
    assert out.teammate_min_gap == base.teammate_min_gap


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_from_mapping_parses_nested_sections() -> None:
    raw = {
        "enabled": True,
        "role_bias": {"attacker_weight_scale": 1.5},
        "formation": {"extra_defenders": 2},
        "attacking": {"shoot_distance_scale": 0.75},
    }

    strategy = StrategyConfig.from_mapping(raw)

    assert strategy.enabled is True
    assert strategy.role_bias.attacker_weight_scale == 1.5
    # Unspecified keys keep neutral defaults.
    assert strategy.role_bias.defender_weight_scale == 1.0
    assert strategy.formation.extra_defenders == 2
    assert strategy.attacking.shoot_distance_scale == 0.75
    assert strategy.defending.line_height_scale == 1.0


def test_strategy_section_override_via_file(tmp_path) -> None:
    config_path = tmp_path / "bt_tuning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "strategy:",
                "  enabled: true",
                "  role_bias:",
                "    press_scale: 1.25",
            ]
        )
    )

    strategy = load_strategy_config(config_path)

    assert strategy.enabled is True
    assert strategy.role_bias.press_scale == 1.25


# ---------------------------------------------------------------------------
# Coordinator integration
# ---------------------------------------------------------------------------


def _make_trees():
    return {
        RoleType.GOALIE: GoalieTree(us_positive=False),
        RoleType.DEFENDER: DefenderTree(us_positive=False),
        RoleType.SUPPORTER: SupporterTree(us_positive=False),
        RoleType.ATTACKER: AttackerTree(us_positive=False),
    }


def test_coordinator_default_strategy_leaves_everything_unchanged() -> None:
    # Compare against the freshly LOADED configs, not the dataclass defaults:
    # bt_tuning.yaml legitimately overrides some values (e.g. shot_settle_ticks).
    # The invariant is that a disabled strategy leaves the loaded configs as-is.
    coordinator = Coordinator(trees=_make_trees(), us_positive=False)

    assert coordinator.strategy == StrategyConfig()
    assert coordinator.heuristic_weights == load_role_heuristic_weights()
    assert (
        coordinator.trees[RoleType.ATTACKER].behavior_config
        == load_attacker_behavior_config()
    )
    assert (
        coordinator.trees[RoleType.SUPPORTER].behavior_config
        == load_supporter_behavior_config()
    )
    assert (
        coordinator.trees[RoleType.DEFENDER].positioning_config
        == load_defender_positioning_config()
    )


def test_coordinator_active_strategy_rewrites_weights_and_trees() -> None:
    strategy = StrategyConfig(
        enabled=True,
        role_bias=RoleBiasStrategy(attacker_weight_scale=2.0),
        attacking=AttackingStrategy(shoot_distance_scale=2.0),
        defending=DefendingStrategy(line_height_scale=0.5),
    )
    base_attacker = AttackerBehaviorConfig()
    base_defender = DefenderPositioningConfig()

    coordinator = Coordinator(
        trees=_make_trees(), us_positive=False, strategy=strategy
    )

    assert coordinator.heuristic_weights.attacker.goal_sight == (
        RoleHeuristicWeights().attacker.goal_sight * 2.0
    )
    assert coordinator.trees[RoleType.ATTACKER].behavior_config.shoot_dist_threshold == (
        base_attacker.shoot_dist_threshold * 2.0
    )
    assert coordinator.trees[RoleType.DEFENDER].positioning_config.shot_block_fraction_from_goal == (
        base_defender.shot_block_fraction_from_goal * 0.5
    )


# ---------------------------------------------------------------------------
# Dynamic layer — game context evaluation
# ---------------------------------------------------------------------------


def _snapshot(ball=(0.0, 0.0), score=(0, 0), own=((0.0, 0.0),), enemy=((4.0, 0.0),)):
    own_robots = tuple(
        RobotState(robot_id=i, position=p, orientation=0.0) for i, p in enumerate(own)
    )
    enemy_robots = tuple(
        RobotState(robot_id=i, position=p, orientation=0.0) for i, p in enumerate(enemy)
    )
    referee = RefereeState(game_phase=GamePhase.RUNNING, score=score)
    return Snapshot(ball, (0.0, 0.0), own_robots, enemy_robots, referee)


def test_evaluate_context_ball_zone_uses_attack_sign() -> None:
    # us_positive=False → attack_sign = +1 (attack toward +x).
    assert evaluate_game_context(_snapshot(ball=(3.0, 0.0)), 1.0).ball_zone == "attacking"
    assert evaluate_game_context(_snapshot(ball=(-3.0, 0.0)), 1.0).ball_zone == "defensive"
    assert evaluate_game_context(_snapshot(ball=(0.0, 0.0)), 1.0).ball_zone == "middle"
    # Flipped attack direction flips the zones.
    assert evaluate_game_context(_snapshot(ball=(3.0, 0.0)), -1.0).ball_zone == "defensive"


def test_evaluate_context_possession_and_scoreline() -> None:
    # Own robot sits on the ball, enemy is far → own possession.
    own_ctx = evaluate_game_context(
        _snapshot(ball=(0.0, 0.0), own=((0.0, 0.0),), enemy=((4.0, 0.0),)), 1.0
    )
    assert own_ctx.possession == "own"

    # Nobody within the possession radius → loose.
    loose_ctx = evaluate_game_context(
        _snapshot(ball=(0.0, 0.0), own=((3.0, 0.0),), enemy=((-3.0, 0.0),)), 1.0
    )
    assert loose_ctx.possession == "loose"

    assert evaluate_game_context(_snapshot(score=(2, 1)), 1.0).scoreline == "leading"
    assert evaluate_game_context(_snapshot(score=(1, 1)), 1.0).scoreline == "level"
    assert evaluate_game_context(_snapshot(score=(0, 2)), 1.0).scoreline == "trailing"
    assert evaluate_game_context(_snapshot(score=(3, 1)), 1.0).goal_margin == 2


# ---------------------------------------------------------------------------
# Dynamic layer — rule matching and composition
# ---------------------------------------------------------------------------


def _ctx(**kw) -> GameContext:
    base = dict(
        ball_zone="middle",
        possession="loose",
        scoreline="level",
        ball_in_our_half=False,
        goal_margin=0,
    )
    base.update(kw)
    return GameContext(**base)


def test_rule_conditions_match_all_specified() -> None:
    cond = RuleConditions(ball_zone="attacking", scoreline="leading")
    assert cond.matches(_ctx(ball_zone="attacking", scoreline="leading")) is True
    assert cond.matches(_ctx(ball_zone="attacking", scoreline="level")) is False
    assert cond.matches(_ctx(ball_zone="middle", scoreline="leading")) is False
    # Empty conditions match everything.
    assert RuleConditions().matches(_ctx()) is True


def test_goal_margin_at_least_uses_absolute_value() -> None:
    cond = RuleConditions(goal_margin_at_least=2)
    assert cond.matches(_ctx(goal_margin=2)) is True
    assert cond.matches(_ctx(goal_margin=-3)) is True
    assert cond.matches(_ctx(goal_margin=1)) is False


def test_resolve_inactive_strategy_is_passthrough() -> None:
    disabled = StrategyConfig(enabled=False, role_bias=RoleBiasStrategy(press_scale=2.0))
    assert resolve_effective_strategy(disabled, _ctx()) is disabled
    assert resolve_effective_strategy(None, _ctx()) is None


def test_resolve_composes_base_and_matching_rules() -> None:
    strategy = StrategyConfig(
        enabled=True,
        role_bias=RoleBiasStrategy(press_scale=1.5),  # base
        rules=(
            StrategyRule(
                name="press_high",
                when=RuleConditions(ball_zone="attacking"),
                apply=PostureDials(role_bias=RoleBiasStrategy(press_scale=2.0)),
            ),
            StrategyRule(
                name="never_fires_here",
                when=RuleConditions(ball_zone="defensive"),
                apply=PostureDials(role_bias=RoleBiasStrategy(press_scale=10.0)),
            ),
        ),
    )

    effective = resolve_effective_strategy(strategy, _ctx(ball_zone="attacking"))

    # base 1.5 × matching rule 2.0 = 3.0; non-matching rule ignored.
    assert effective.role_bias.press_scale == 3.0
    assert effective.rules == ()  # rules consumed into the dials

    # In a different zone only the base survives.
    base_only = resolve_effective_strategy(strategy, _ctx(ball_zone="middle"))
    assert base_only.role_bias.press_scale == 1.5


def test_resolve_offsets_add_across_rules() -> None:
    strategy = StrategyConfig(
        enabled=True,
        rules=(
            StrategyRule(
                when=RuleConditions(scoreline="leading"),
                apply=PostureDials(formation=FormationStrategy(extra_defenders=1)),
            ),
            StrategyRule(
                when=RuleConditions(ball_in_our_half=True),
                apply=PostureDials(formation=FormationStrategy(extra_defenders=1)),
            ),
        ),
    )

    both = resolve_effective_strategy(
        strategy, _ctx(scoreline="leading", ball_in_our_half=True)
    )
    assert both.formation.extra_defenders == 2  # 0 + 1 + 1


def test_strategy_rules_parse_from_file(tmp_path) -> None:
    config_path = tmp_path / "bt_tuning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "strategy:",
                "  enabled: true",
                "  rules:",
                "    - name: high_press",
                "      when:",
                "        ball_zone: attacking",
                "      apply:",
                "        role_bias: {press_scale: 1.4}",
                "        formation: {extra_attackers: 1}",
            ]
        )
    )

    strategy = load_strategy_config(config_path)

    assert len(strategy.rules) == 1
    rule = strategy.rules[0]
    assert rule.name == "high_press"
    assert rule.when.ball_zone == "attacking"
    assert rule.apply.role_bias.press_scale == 1.4
    assert rule.apply.formation.extra_attackers == 1


# ---------------------------------------------------------------------------
# Dynamic layer — Coordinator per-tick behaviour
# ---------------------------------------------------------------------------


def _dynamic_coordinator() -> Coordinator:
    strategy = StrategyConfig(
        enabled=True,
        rules=(
            StrategyRule(
                name="press_high",
                when=RuleConditions(ball_zone="attacking"),
                apply=PostureDials(
                    role_bias=RoleBiasStrategy(press_scale=2.0),
                    formation=FormationStrategy(extra_attackers=1),
                ),
            ),
        ),
    )
    return Coordinator(trees=_make_trees(), us_positive=False, strategy=strategy)


def test_coordinator_refresh_reacts_to_ball_zone() -> None:
    coordinator = _dynamic_coordinator()
    base = RoleHeuristicWeights()

    # Ball in the attacking third → rule fires.
    coordinator._refresh_dynamic_strategy(_snapshot(ball=(3.0, 0.0)))
    assert coordinator.heuristic_weights.attacker.ball_close == base.attacker.ball_close * 2.0
    assert coordinator.heuristic_weights.role_targets.attackers == base.role_targets.attackers + 1


def test_coordinator_refresh_reverts_and_does_not_compound() -> None:
    coordinator = _dynamic_coordinator()
    base = RoleHeuristicWeights()

    # Fire the rule three times in a row...
    for _ in range(3):
        coordinator._refresh_dynamic_strategy(_snapshot(ball=(3.0, 0.0)))
    # ...press scale must NOT compound (still ×2, not ×8).
    assert coordinator.heuristic_weights.attacker.ball_close == base.attacker.ball_close * 2.0

    # Ball moves back to our defensive third → rule stops firing, revert to base.
    coordinator._refresh_dynamic_strategy(_snapshot(ball=(-3.0, 0.0)))
    assert coordinator.heuristic_weights.attacker.ball_close == base.attacker.ball_close
    assert coordinator.heuristic_weights.role_targets.attackers == base.role_targets.attackers
