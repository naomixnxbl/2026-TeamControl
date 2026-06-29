from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentOrient
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import angle_to


def face_ball(snap: Snapshot, robot: RobotState | None, target) -> IntentOrient | None:
    """Rotate in place to face the ball."""
    if robot is None:
        return None
    return IntentOrient(target_orientation=angle_to(robot.position, snap.ball_position))
