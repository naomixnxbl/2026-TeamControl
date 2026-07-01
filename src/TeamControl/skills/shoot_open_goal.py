"""shoot_open_goal skill — strike at the most open part of the goal mouth.

Keeper-aware finish: aim at the point across the goal mouth farthest from any
opponent (the keeper). Emits an ``IntentKick``; the PD-backed motion layer drives
behind the ball, aligns, and fires the kicker on contact.
"""
from __future__ import annotations

from TeamControl.bt.contracts.intent import IntentKick
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import best_goal_target


def shoot_open_goal(snap: Snapshot, robot: RobotState | None, target) -> IntentKick | None:
    """Strike at the most open part of the goal (keeper- & side-aware)."""
    if robot is None:
        return None
    return IntentKick(target_pos=best_goal_target(snap))
