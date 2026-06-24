"""Loader for ``sim_6v6.yaml`` — 6v6 BT-vs-BT simulation config.

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
}


def _parse_roles(raw: dict) -> dict[int, RoleType]:
    out: dict[int, RoleType] = {}
    for key, value in raw.items():
        try:
            role = _ROLE_LOOKUP[str(value).lower()]
        except KeyError as e:
            raise ValueError(
                f"Unknown role {value!r} for robot {key} in sim_6v6.yaml "
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
    ):
        if key in raw:
            out[key] = float(raw[key])
    return out


class Sim6v6Config:
    """In-memory view of ``sim_6v6.yaml``."""

    def __init__(self, config_filename: str = "sim_6v6.yaml") -> None:
        path = Path(__file__).resolve().parent / config_filename
        with open(path, "r") as f:
            raw = yaml.load(f, Loader)

        self.yellow_ids: list[int] = [int(x) for x in raw["yellow"]["robot_ids"]]
        self.blue_ids: list[int] = [int(x) for x in raw["blue"]["robot_ids"]]
        self.roles: dict[int, RoleType] = _parse_roles(raw["roles"])
        self.heuristic_role_swap: bool = bool(raw.get("heuristic_role_swap", False))
        self.movement_safety: dict[str, bool | float] = _parse_movement_safety(
            raw.get("movement_safety")
        )
        self.tick_period: float = float(raw.get("tick_period", 0.01))

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return (
            f"Sim6v6Config(yellow_ids={self.yellow_ids}, "
            f"blue_ids={self.blue_ids}, "
            f"roles={ {k: v.name for k, v in self.roles.items()} }, "
            f"heuristic_role_swap={self.heuristic_role_swap}, "
            f"movement_safety={self.movement_safety}, "
            f"tick_period={self.tick_period})"
        )


if __name__ == "__main__":
    print(Sim6v6Config())
