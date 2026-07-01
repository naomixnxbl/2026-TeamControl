from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentKick
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import best_goal_target


def kick_at_goal(snap: Snapshot, robot: RobotState | None, target) -> IntentKick | None:
    """Strike at the most open part of the opponent goal (keeper- & side-aware).

    Emits an ``IntentKick`` at the open aim point; the PD-backed motion layer
    drives behind the ball, aligns, and fires the kicker on contact.
    """
    if robot is None:
        return None
    return IntentKick(target_pos=best_goal_target(snap))
