from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentDribble
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot


def dribble_to_point(snap: Snapshot, robot: RobotState | None,
                     target: tuple[float, float] | None) -> IntentDribble | None:
    """Carry (dribble) ball to a chosen point — used for ball placement (§9)."""
    if target is None:
        return None
    return IntentDribble(target_pos=target)
