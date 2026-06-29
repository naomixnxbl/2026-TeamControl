from __future__ import annotations
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import HALF_LEN_M, seq_move_then_kick


def move_then_attack(snap: Snapshot, robot: RobotState | None, _target):
    """Move to ball then decide target by field position:
      own half  (ball x < 0) → pass/clear toward centre
      enemy half (ball x ≥ 0) → shoot at goal
    """
    if robot is None:
        return None
    kick_target = (0.0, 0.0) if snap.ball_position[0] < 0 else (HALF_LEN_M, 0.0)
    return seq_move_then_kick(snap, robot, kick_target)
