from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentKick
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot


def kick_at_point(snap: Snapshot, robot: RobotState | None,
                  target: tuple[float, float] | None) -> IntentKick | None:
    """Pass/kick toward a chosen point.

    Emits an ``IntentKick``; the PD-backed motion layer drives behind the ball,
    aligns toward the target, and fires the kicker on contact.
    """
    if robot is None or target is None:
        return None
    return IntentKick(target_pos=target)
