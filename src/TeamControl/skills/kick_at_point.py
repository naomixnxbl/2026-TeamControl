from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentKick
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import kick_sequence


def kick_at_point(snap: Snapshot, robot: RobotState | None,
                  target: tuple[float, float] | None) -> IntentKick | None:
    """Pass: get behind ball → align heading → strike toward a chosen point."""
    if robot is None or target is None:
        return None
    prep = kick_sequence(snap, robot, target)
    return prep if prep is not None else IntentKick(target_pos=target)
