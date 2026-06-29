"""Reusable team tactics helpers for BT-level decision making.

These modules are intentionally passive: importing them does not change any
runtime behaviour. Coordinator integration should be explicit.
"""

from TeamControl.bt.tactics.rule_following import (
    MovementSafetyConfig,
    apply_rule_following,
    has_rule_following_enabled,
)

__all__ = [
    "MovementSafetyConfig",
    "apply_rule_following",
    "has_rule_following_enabled",
]
