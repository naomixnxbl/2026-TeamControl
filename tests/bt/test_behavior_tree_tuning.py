from __future__ import annotations

from TeamControl.bt.trees.attacker import (
    AttackerBehaviorConfig,
    load_attacker_behavior_config,
)
from TeamControl.bt.trees.supporter import (
    SupporterBehaviorConfig,
    load_supporter_behavior_config,
)


def test_attacker_behavior_config_loads_current_bt_tuning_defaults() -> None:
    config = load_attacker_behavior_config()

    assert config == AttackerBehaviorConfig()


def test_supporter_behavior_config_loads_current_bt_tuning_defaults() -> None:
    config = load_supporter_behavior_config()

    assert config == SupporterBehaviorConfig()


def test_attacker_behavior_config_can_override_values(tmp_path) -> None:
    config_path = tmp_path / "bt_tuning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "behavior_tree:",
                "  attacker:",
                "    shot_settle_ticks: 12",
                "    supporter_role_ids: [2, 5]",
                "    pass_openness_weight: 0.7",
            ]
        )
    )

    config = load_attacker_behavior_config(config_path)

    assert config.shot_settle_ticks == 12
    assert config.supporter_role_ids == (2, 5)
    assert config.pass_openness_weight == 0.7
    assert config.possession_dist == AttackerBehaviorConfig().possession_dist


def test_supporter_behavior_config_can_override_values(tmp_path) -> None:
    config_path = tmp_path / "bt_tuning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "behavior_tree:",
                "  supporter:",
                "    pass_signal_timeout_ticks: 42",
                "    grid_step: 0.25",
                "    attacker_pass_bonus: 2.0",
            ]
        )
    )

    config = load_supporter_behavior_config(config_path)

    assert config.pass_signal_timeout_ticks == 42
    assert config.grid_step == 0.25
    assert config.attacker_pass_bonus == 2.0
    assert config.possession_dist == SupporterBehaviorConfig().possession_dist
