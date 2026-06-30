"""The blue Division B opponent: a distinct, defensively-solid + direct side.

Built natively from our trees + the strategy layer (no external/licensed code),
used as blue in the `gegenpress` UI mode so it's a real foil, not a mirror.
"""
from __future__ import annotations

from TeamControl.bt.run_bt_v2_process import _build_coordinator
from TeamControl.utils.sim_config import SimBlueDivBConfig


def test_config_loads_distinct_div_b_posture() -> None:
    cfg = SimBlueDivBConfig()
    # Nearest-robot-attacks + direct counter, distinct from yellow's press.
    assert cfg.heuristic_role_swap is True
    assert cfg.counter_attack is True
    assert cfg.gegenpress is None  # blue does NOT gegenpress
    assert cfg.strategy is not None and cfg.strategy.get("enabled") is True


def test_strategy_posture_reaches_the_coordinator() -> None:
    cfg = SimBlueDivBConfig()
    coord = _build_coordinator(
        us_positive=False,
        role_assignment=cfg.roles,
        heuristic_role_swap=cfg.heuristic_role_swap,
        counter_attack=cfg.counter_attack,
        strategy=cfg.strategy,
    )
    # The defensive/direct posture is live on the coordinator.
    assert coord.strategy.enabled is True
    assert coord.strategy.formation.extra_defenders >= 1          # 2-man back line
    assert coord.strategy.role_bias.defender_weight_scale > 1.0   # pulls back
    assert coord.strategy.attacking.pass_caution_scale > 1.0      # direct, sparing passes
    assert len(coord.strategy.rules) >= 1                         # context rules present


def test_blue_does_not_use_press_machinery() -> None:
    cfg = SimBlueDivBConfig()
    coord = _build_coordinator(
        us_positive=False,
        role_assignment=cfg.roles,
        heuristic_role_swap=cfg.heuristic_role_swap,
        counter_attack=cfg.counter_attack,
        strategy=cfg.strategy,
    )
    assert coord.gegenpress.enabled is False
