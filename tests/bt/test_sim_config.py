from __future__ import annotations

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.utils.sim_config import BTSimConfig, Sim3v3Config, Sim6v6Config


def test_sim_6v6_config_still_loads_default_file() -> None:
    config = Sim6v6Config()

    assert config.yellow_ids == [0, 1, 2, 3, 4, 5]
    assert config.blue_ids == [0, 1, 2, 3, 4, 5]
    assert config.roles[0] == RoleType.GOALIE
    assert config.tick_period == 0.01


def test_sim_3v3_config_loads_goalie_attacker_supporter_roles() -> None:
    config = Sim3v3Config()

    assert config.yellow_ids == [0, 1, 2]
    assert config.blue_ids == [0, 1, 2]
    assert config.roles == {
        0: RoleType.GOALIE,
        1: RoleType.ATTACKER,
        2: RoleType.SUPPORTER,
    }
    assert config.heuristic_role_swap is False
    assert config.movement_safety["keep_goalie_in_goal_box"] is True
    assert config.movement_safety["defense_area_dribble_kick_margin"] == 0.30


def test_generic_bt_sim_config_can_load_3v3_file() -> None:
    config = BTSimConfig("sim_3v3.yaml")

    assert config.yellow_ids == [0, 1, 2]
    assert config.roles[0] == RoleType.GOALIE
