from __future__ import annotations

from TeamControl.bt.tactics.heuristic_role_swap import load_role_heuristic_weights


def test_default_heuristic_weight_file_loads_current_values() -> None:
    weights = load_role_heuristic_weights()

    assert weights.attacker.ball_close == 0.40
    assert weights.attacker.approach_quality == 0.20
    assert weights.attacker.opponent_has_ball_pressure == 0.12
    assert weights.attacker.loose_ball_pressure == 0.10
    assert weights.defender.own_goal_close == 0.28
    assert weights.supporter.spacing == 0.30
    assert weights.stability.current_role_bias == 0.08
    assert weights.stability.cooldown_bias == 0.16
    assert weights.context.goal_sight_clearance_field_scale == 0.02
    assert weights.context.possession_radius_field_scale == 0.06
    assert weights.role_targets.max_defenders == 2


def test_partial_heuristic_weight_file_keeps_defaults(tmp_path) -> None:
    config_path = tmp_path / "heuristic_weight.yaml"
    config_path.write_text(
        "\n".join(
            [
                "attacker:",
                "  ball_close: 0.75",
                "stability:",
                "  minimum_swap_interval: 2.5",
                "defender_count:",
                "  add_second_when_opponent_has_ball: false",
                "role_targets:",
                "  max_defenders: 3",
            ]
        ),
        encoding="utf-8",
    )

    weights = load_role_heuristic_weights(config_path)

    assert weights.attacker.ball_close == 0.75
    assert weights.attacker.approach_quality == 0.20
    assert weights.attacker.opponent_has_ball_pressure == 0.12
    assert weights.stability.minimum_swap_interval == 2.5
    assert weights.stability.current_role_bias == 0.08
    assert weights.defender_count.add_second_when_opponent_has_ball is False
    assert weights.defender_count.add_second_when_ball_in_our_half is True
    assert weights.role_targets.max_defenders == 3
    assert weights.role_targets.min_defenders == 1
