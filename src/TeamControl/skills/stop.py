from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot


def stop(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Hold current position and heading."""
    if robot is None:
        return None
    return IntentMove(target_pos=robot.position, target_orientation=robot.orientation)
