from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import HALF_LEN_M, PENALTY_SPOT_X_M, angle_to


def penalty_attacker_stance(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Move to the penalty spot facing the opponent goal — §8.2.3 PENALTY_SHOOT.

    Penalty spot is 1 m from the goal line.
    """
    if robot is None:
        return None
    stance = (PENALTY_SPOT_X_M, 0.0)
    return IntentMove(target_pos=stance, target_orientation=angle_to(stance, (HALF_LEN_M, 0.0)))
