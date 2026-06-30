"""Loader for BT-vs-BT simulation configs.

Kept separate from ``yaml_config.Config`` (which loads ``ipconfig.yaml``)
so the two concerns don't bleed into each other: network/team settings vs
sim scenario settings.
"""
from __future__ import annotations

from pathlib import Path

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from TeamControl.bt.contracts.blackboard import RoleType


_ROLE_LOOKUP = {r.name.lower(): r for r in RoleType}
_TEAM_LOOKUP = {"yellow": True, "blue": False}
_DEFAULT_MOVEMENT_SAFETY = {
    "keep_robots_in_bounds": True,
    "keep_goalie_in_goal_box": True,
    "keep_non_goalies_out_of_goalie_box": True,
    "avoid_ball_touch_in_opponent_defense_area": True,
    "field_length": 9.0,
    "field_width": 6.0,
    "field_margin": 0.05,
    "goalie_box_depth": 1.0,
    "goalie_box_width": 2.0,
    "goalie_box_margin": 0.05,
    "goalie_box_avoid_margin": 0.15,
    "goalie_box_exit_margin": 0.10,
    "defense_area_ball_touch_margin": 0.18,
    "defense_area_dribble_kick_margin": 0.30,
}


def _parse_roles(raw: dict, config_filename: str) -> dict[int, RoleType]:
    out: dict[int, RoleType] = {}
    for key, value in raw.items():
        try:
            role = _ROLE_LOOKUP[str(value).lower()]
        except KeyError as e:
            raise ValueError(
                f"Unknown role {value!r} for robot {key} in {config_filename} "
                f"(valid: {list(_ROLE_LOOKUP)})"
            ) from e
        out[int(key)] = role
    return out


def _parse_movement_safety(raw: dict | None) -> dict[str, bool | float]:
    if raw is None:
        return dict(_DEFAULT_MOVEMENT_SAFETY)

    out = dict(_DEFAULT_MOVEMENT_SAFETY)
    for key in (
        "keep_robots_in_bounds",
        "keep_goalie_in_goal_box",
        "keep_non_goalies_out_of_goalie_box",
        "avoid_ball_touch_in_opponent_defense_area",
    ):
        if key in raw:
            out[key] = bool(raw[key])
    for key in (
        "field_length",
        "field_width",
        "field_margin",
        "goalie_box_depth",
        "goalie_box_width",
        "goalie_box_margin",
        "goalie_box_avoid_margin",
        "goalie_box_exit_margin",
        "defense_area_ball_touch_margin",
        "defense_area_dribble_kick_margin",
    ):
        if key in raw:
            out[key] = float(raw[key])
    return out


def _parse_controlled_team(raw: object, config_filename: str) -> str:
    team = str(raw).strip().lower()
    if team not in _TEAM_LOOKUP:
        raise ValueError(
            f"controlled_team must be 'yellow' or 'blue' in {config_filename}; "
            f"got {raw!r}"
        )
    return team


class BTSimConfig:
    """In-memory view of a BT-vs-BT simulation yaml config."""

    def __init__(self, config_filename: str = "sim_6v6.yaml") -> None:
        self.config_filename = config_filename
        path = Path(__file__).resolve().parent / config_filename
        with open(path, "r") as f:
            raw = yaml.load(f, Loader)
        self._raw_config = raw

        self.yellow_ids: list[int] = [int(x) for x in raw["yellow"]["robot_ids"]]
        self.blue_ids: list[int] = [int(x) for x in raw["blue"]["robot_ids"]]
        self.roles: dict[int, RoleType] = _parse_roles(raw["roles"], config_filename)
        self.heuristic_role_swap: bool = bool(raw.get("heuristic_role_swap", False))
        # GegenPressing containment: when true the lone attacker contains the
        # ball carrier goal-side instead of diving at the ball. Default False so
        # every existing sim config is unchanged.
        self.attacker_press: bool = bool(raw.get("attacker_press", False))
        # Reactive GegenPressing trigger (see coordinator.GegenpressConfig).
        # Raw mapping passed straight through to the Coordinator; None when the
        # config omits it (every existing sim config), keeping the press off.
        self.gegenpress: dict | None = raw.get("gegenpress")
        # Counter-attack release: when true the attacker prefers a direct forward
        # pass into the opponent half over carrying the ball up. Default False so
        # existing sim configs are unchanged.
        self.counter_attack: bool = bool(raw.get("counter_attack", False))
        self.movement_safety: dict[str, bool | float] = _parse_movement_safety(
            raw.get("movement_safety")
        )
        self.tick_period: float = float(raw.get("tick_period", 0.01))

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return (
            f"{type(self).__name__}(config_filename={self.config_filename!r}, "
            f"yellow_ids={self.yellow_ids}, "
            f"blue_ids={self.blue_ids}, "
            f"roles={ {k: v.name for k, v in self.roles.items()} }, "
            f"heuristic_role_swap={self.heuristic_role_swap}, "
            f"movement_safety={self.movement_safety}, "
            f"tick_period={self.tick_period})"
        )


class Sim6v6Config(BTSimConfig):
    """In-memory view of ``sim_6v6.yaml``."""

    def __init__(self, config_filename: str = "sim_6v6.yaml") -> None:
        super().__init__(config_filename)


class Sim3v3Config(BTSimConfig):
    """In-memory view of ``sim_3v3.yaml``."""

    def __init__(self, config_filename: str = "sim_3v3.yaml") -> None:
        super().__init__(config_filename)


class SimGegenpressConfig(BTSimConfig):
    """In-memory view of ``sim_gegenpress.yaml`` — the GegenPressing strategy.

    Same shape as the 6v6 sim but with MARKER roles for the outfield robots and
    ``attacker_press: true`` so the lone attacker contains the ball carrier.
    """

    def __init__(self, config_filename: str = "sim_gegenpress.yaml") -> None:
        super().__init__(config_filename)


class Btv2Config(BTSimConfig):
    """Single-team GUI ``btv2`` config."""

    def __init__(self, config_filename: str = "btv2.yaml") -> None:
        super().__init__(config_filename)
        self.controlled_team: str = _parse_controlled_team(
            self._raw_config.get("controlled_team", "yellow"),
            config_filename,
        )
        self.controlled_is_yellow: bool = _TEAM_LOOKUP[self.controlled_team]
        self.controlled_robot_ids: list[int] = (
            self.yellow_ids if self.controlled_is_yellow else self.blue_ids
        )


if __name__ == "__main__":
    print(Sim6v6Config())
