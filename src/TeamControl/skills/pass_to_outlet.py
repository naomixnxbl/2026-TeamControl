"""pass_to_outlet skill — auto-pick the most advanced open teammate and pass.

The counter-attack release primitive: find the teammate furthest forward (toward
the opponent goal we attack) that is open and on a clear lane, then kick to them.
When no clean forward outlet exists, the robot faces the ball ("looking for a
pass") rather than forcing a bad one.

Emits an ``IntentKick`` at the outlet; the PD-backed motion layer drives behind
the ball, aligns, and fires the kicker on contact. Side-aware via the global
attack direction (``_shared.forward_outlet``).
"""
from __future__ import annotations

from TeamControl.bt.contracts.intent import IntentKick, IntentOrient
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import angle_to, forward_outlet


def pass_to_outlet(
    snap: Snapshot, robot: RobotState | None, target
) -> IntentKick | IntentOrient | None:
    """Kick to the most-advanced open teammate; face the ball if none is open."""
    if robot is None:
        return None
    outlet = forward_outlet(snap, robot)
    if outlet is None:
        return IntentOrient(target_orientation=angle_to(robot.position, snap.ball_position))
    return IntentKick(target_pos=outlet)
