from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import CENTER_CIRCLE_R_M, angle_to


def kickoff_stance(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Move to centre-circle edge, own half, facing ball — §5.3.2 PREPARE_KICKOFF.

    All non-kicker robots must be in their own half and outside the centre circle
    (radius 0.5 m) until the ball is touched.
    """
    if robot is None:
        return None
    stance = (-CENTER_CIRCLE_R_M, 0.0)
    return IntentMove(target_pos=stance, target_orientation=angle_to(stance, snap.ball_position))
