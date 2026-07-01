from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import GOAL_HW_M, own_goal, angle_to


def hold_goal_line(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Goalkeeper holds on our goal line tracking ball y — §8.2 PENALTY_DEFEND.

    Clamps to goal half-width so the keeper stays inside the posts. Side-aware
    via the global attack direction.
    """
    if robot is None:
        return None
    stance_y = max(-GOAL_HW_M, min(GOAL_HW_M, snap.ball_position[1]))
    stance   = (own_goal()[0], stance_y)
    return IntentMove(target_pos=stance, target_orientation=angle_to(stance, snap.ball_position))
