"""Reusable team tactics helpers for BT-level decision making.

These modules are intentionally passive: importing them does not change any
runtime behaviour. Coordinator integration should be explicit.
"""

from TeamControl.bt.tactics.rule_following import (
    MovementSafetyConfig,
    apply_rule_following,
    has_rule_following_enabled,
)
from TeamControl.bt.tactics.strategy import (
    GameContext,
    PostureDials,
    RuleConditions,
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

__all__ = [
    "MovementSafetyConfig",
    "apply_rule_following",
    "has_rule_following_enabled",
    "GameContext",
    "PostureDials",
    "RuleConditions",
    "StrategyConfig",
    "StrategyRule",
    "apply_strategy_to_attacker_config",
    "apply_strategy_to_defender_positioning",
    "apply_strategy_to_role_weights",
    "apply_strategy_to_supporter_config",
    "evaluate_game_context",
    "is_strategy_active",
    "load_strategy_config",
    "resolve_effective_strategy",
]
