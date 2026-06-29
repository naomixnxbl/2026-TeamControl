"""Stop — full SSL STOPPED state compliance (§5.4).

Two conditions must be met simultaneously:
  1. Ball clearance : stay ≥ 0.5 m from ball   (code nudges to 0.55 m buffer)
  2. Speed cap      : never exceed 1.4 m/s       (rule threshold is < 1.5 m/s)
"""
from __future__ import annotations
import math
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import angle_to, nudge_away

_CLEARANCE_M  = 0.55   # buffer above the 0.5 m rule minimum (matches coordinator)
_MAX_SPEED    = 1.4    # m/s — SSL §5.4 requires < 1.5 m/s


def stop(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Hold position and satisfy both SSL STOPPED conditions (§5.4):
      - Nudge away to 0.55 m clearance if closer than that to the ball.
      - Cap all movement at 1.4 m/s regardless.
    """
    if robot is None:
        return None
    dist = math.hypot(robot.position[0] - snap.ball_position[0],
                      robot.position[1] - snap.ball_position[1])
    if dist < _CLEARANCE_M:
        stance = nudge_away(robot.position, snap.ball_position, _CLEARANCE_M)
        return IntentMove(target_pos=stance,
                          target_orientation=angle_to(stance, snap.ball_position),
                          max_speed=_MAX_SPEED)
    return IntentMove(target_pos=robot.position,
                      target_orientation=robot.orientation,
                      max_speed=_MAX_SPEED)


def compliance(snap: Snapshot, robot: RobotState | None) -> list[str]:
    if robot is None:
        return []
    dist = math.hypot(robot.position[0] - snap.ball_position[0],
                      robot.position[1] - snap.ball_position[1])
    dist_ok  = dist >= 0.5
    return [
        f"ball clearance: {dist:.2f} m  (rule ≥ 0.5 m) — {'OK' if dist_ok else 'VIOLATION'}",
        f"speed cap: 1.4 m/s enforced  (rule < 1.5 m/s)",
    ]
