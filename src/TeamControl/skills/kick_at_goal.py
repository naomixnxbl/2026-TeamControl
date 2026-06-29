from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentKick
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import HALF_LEN_M, kick_sequence


def kick_at_goal(snap: Snapshot, robot: RobotState | None, target) -> IntentKick | None:
    """Get behind ball → align heading → strike toward the opponent goal."""
    if robot is None:
        return None
    prep = kick_sequence(snap, robot, (HALF_LEN_M, 0.0))
    return prep if prep is not None else IntentKick(target_pos=(HALF_LEN_M, 0.0))
