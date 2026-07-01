from __future__ import annotations
from TeamControl.bt.contracts.intent import IntentKick
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import (
    best_goal_target, forward_outlet, in_own_half, opp_goal,
)


def move_then_attack(snap: Snapshot, robot: RobotState | None, _target) -> IntentKick | None:
    """Counter-attack: release the ball the most direct way (side-aware).

    Mirrors the gegenpress attacker's priority:
      * ball in OUR half   → pass to the most-advanced open teammate (forward
        outlet); if none is open, drive it forward toward the opponent goal.
      * ball in ENEMY half → shoot at the most open part of the goal.

    Emits an ``IntentKick``; the PD-backed motion layer drives to the ball,
    aligns, and fires the kicker on contact.
    """
    if robot is None:
        return None
    if in_own_half(snap.ball_position[0]):
        outlet = forward_outlet(snap, robot)
        kick_target = outlet if outlet is not None else opp_goal()
    else:
        kick_target = best_goal_target(snap)
    return IntentKick(target_pos=kick_target)
