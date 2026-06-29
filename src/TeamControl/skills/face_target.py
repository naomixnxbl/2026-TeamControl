from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentOrient
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import angle_to


def face_target(snap: Snapshot, robot: RobotState | None,
                target: tuple[float, float] | None) -> IntentOrient | None:
    """Rotate in place to face a chosen field point."""
    if robot is None or target is None:
        return None
    return IntentOrient(target_orientation=angle_to(robot.position, target))
