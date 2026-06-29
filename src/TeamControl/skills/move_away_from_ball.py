from __future__ import annotations
import math
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import STOP_CLEARANCE_M, angle_to, nudge_away
from TeamControl.skills.stop import stop


def move_away_from_ball(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Move to exactly 0.6 m from the ball — satisfies §5.4 STOPPED clearance (≥ 0.5 m).

    The 0.6 m buffer gives a 10 cm safety margin over the 0.5 m rule threshold.
    """
    if robot is None:
        return None
    dist = math.hypot(robot.position[0] - snap.ball_position[0],
                      robot.position[1] - snap.ball_position[1])
    if dist >= STOP_CLEARANCE_M:
        return stop(snap, robot, target)
    stance = nudge_away(robot.position, snap.ball_position, STOP_CLEARANCE_M)
    return IntentMove(target_pos=stance, target_orientation=angle_to(stance, snap.ball_position))


def compliance(snap: Snapshot, robot: RobotState | None) -> list[str]:
    if robot is None:
        return []
    dist = math.hypot(robot.position[0] - snap.ball_position[0],
                      robot.position[1] - snap.ball_position[1])
    status = "OK" if dist >= 0.5 else "TOO CLOSE (< 0.5 m)"
    return [f"ball distance: {dist:.2f} m — rule limit 0.5 m — {status}"]
