"""Blackboard — per-robot mutable decision state for the behaviour tree pipeline.

R003: RobotBlackboard holds decision state only. No world state belongs here;
world state lives in Snapshot. The blackboard is updated after each tick:
current_intent is written and last_intent is shifted.
"""
from __future__ import annotations

import dataclasses
from enum import Enum

from TeamControl.bt.contracts.intent import Intent


class RoleType(str, Enum):
    """Role assigned to a robot for the current tick."""

    ATTACKER = "ATTACKER"
    DEFENDER = "DEFENDER"
    SUPPORTER = "SUPPORTER"
    GOALIE = "GOALIE"
    # Man-marking role used by the GegenPressing strategy: shadow a specific
    # opponent goal-side (one-on-one), dropping into zone cover when that
    # opponent leaves the danger area. The opponent it shadows is decided
    # team-level by the Coordinator and handed to the tree via
    # ``RobotBlackboard.mark_target_id``.
    MARKER = "MARKER"


@dataclasses.dataclass
class RobotBlackboard:
    """Per-robot mutable decision state.

    This dataclass holds only decision state — never world state.
    World state (ball position, opponent data, etc.) is passed in via Snapshot.

    Fields
    ------
    robot_id : int
        Unique identifier for the robot this blackboard belongs to.
    current_role : RoleType
        The role the robot is currently assigned.
    current_intent : Intent | None
        The intent produced by the behaviour tree on the most recent tick.
        Written by the tree after each tick.
    last_intent : Intent | None
        The intent produced on the previous tick.
        Shifted from current_intent at the start of each tick.
    """

    robot_id: int
    current_role: RoleType
    current_intent: Intent | None = dataclasses.field(default=None)
    last_intent: Intent | None = dataclasses.field(default=None)
    # Name of the BT node (or phase handler) that wrote ``current_intent``
    # this tick. Used for debug logging — lets traces show which branch of
    # the tree fired without parsing the intent payload.
    intent_source: str | None = dataclasses.field(default=None)
    # For a MARKER robot: the id of the opponent it has been assigned to shadow
    # this tick, or ``None`` when it has no man (drop into zone cover). Written
    # by the Coordinator's marker-assignment pass before the marker tree ticks;
    # ignored by every other role.
    mark_target_id: int | None = dataclasses.field(default=None)
