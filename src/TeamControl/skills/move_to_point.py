from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot


def move_to_point(snap: Snapshot, robot: RobotState | None,
                  target: tuple[float, float] | None) -> IntentMove | None:
    """Drive to a chosen field position."""
    if target is None:
        return None
    return IntentMove(target_pos=target, target_orientation=None)
