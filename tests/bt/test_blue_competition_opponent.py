"""The default blue opponent: a competition-realistic, adaptive team."""
from __future__ import annotations

from TeamControl.bt.run_bt_v2_process import _build_coordinator
from TeamControl.utils.sim_config import SimBlueCompetitionConfig


def test_loads_competition_posture() -> None:
    cfg = SimBlueCompetitionConfig()
    assert cfg.heuristic_role_swap is True          # dynamic, real shape
    assert cfg.counter_attack is True               # direct forward play
    assert cfg.gegenpress is None                   # not our reactive press
    assert cfg.strategy is not None and cfg.strategy.get("enabled") is True
    # An adaptive match posture: presses, commits forward, defends, hunts loose.
    assert len(cfg.strategy.get("rules", [])) >= 3


def test_posture_reaches_coordinator_but_no_press() -> None:
    cfg = SimBlueCompetitionConfig()
    coord = _build_coordinator(
        us_positive=False,
        role_assignment=cfg.roles,
        heuristic_role_swap=cfg.heuristic_role_swap,
        counter_attack=cfg.counter_attack,
        strategy=cfg.strategy,
    )
    assert coord.strategy.enabled is True
    assert len(coord.strategy.rules) >= 3
    # Beatable: the reactive GegenPressing trigger is OFF for the opponent.
    assert coord.gegenpress.enabled is False
