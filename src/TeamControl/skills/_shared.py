"""Shared constants, caches, helpers, and the Behaviour contract used by every skill."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from TeamControl.bt.contracts.intent import Intent
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.world.field_config import FIELD_LENGTH_MM, GOAL_HALF_WIDTH_MM

# ── Field geometry ────────────────────────────────────────────────────────────
HALF_LEN_M: float = FIELD_LENGTH_MM / 2.0 / 1000.0
GOAL_HW_M:  float = GOAL_HALF_WIDTH_MM / 1000.0

# ── Movement tuning ───────────────────────────────────────────────────────────
BALL_APPROACH_OFFSET_M:  float = 0.15
FACE_BALL_TOLERANCE_RAD: float = 0.10
MAX_SPEED_M_S:           float = 2.0
KP: float = 2.0
KD: float = 2.5

# ── Kick sequence tuning ──────────────────────────────────────────────────────
KICK_BEHIND_DIST_M:    float = 0.25
KICK_APPROACH_TOL_M:   float = 0.07
KICK_ALIGN_TOL_RAD:    float = 0.08
KICK_STRIKE_SPEED_M_S: float = 0.6
KICK_READY_DIST_M:     float = 0.05

# ── Rule clearances ───────────────────────────────────────────────────────────
STOP_CLEARANCE_M:    float = 0.60   # §5.4 STOPPED — legal ≥ 0.5 m; 0.6 m buffer
CENTER_CIRCLE_R_M:   float = 0.50   # §2.1.3 centre-circle radius
PENALTY_SPOT_X_M:    float = HALF_LEN_M - 1.0   # §8.2.3

# ── Goalie tuning ─────────────────────────────────────────────────────────────
GK_BASE_SPEED:         float = 0.5
GK_SPEED_SCALE:        float = 0.7
GK_MAX_SPEED:          float = 2.0
GK_PREDICT_HORIZON_S:  float = 3.0
GK_STANCE_RATIO:       float = 0.25

# ── Defender tuning ───────────────────────────────────────────────────────────
DEF_BLOCK_ADVANCE_M: float = 0.35

# ── Intercept lookahead ───────────────────────────────────────────────────────
INTERCEPT_LOOKAHEAD_S: float = 0.8

# ── Per-robot caches (module-level so all skill files share the same state) ───
approach_cache: dict[int, tuple[tuple[float, float], float]] = {}
kick_phase:     dict[int, str] = {}   # "position" | "align" | "strike"
seq_phase:      dict[int, str] = {}   # "approach" | "kick"


def reset_robot_state(robot_id: int) -> None:
    """Clear all per-robot caches. Call whenever a skill is started or restarted."""
    approach_cache.pop(robot_id, None)
    kick_phase.pop(robot_id, None)
    seq_phase.pop(robot_id, None)


# ── Behaviour contract ────────────────────────────────────────────────────────
SkillFn      = Callable[[Snapshot, "RobotState | None", "tuple[float,float] | None"], "Intent | None"]
ComplianceFn = Callable[[Snapshot, "RobotState | None"], "list[str]"]


@dataclass(frozen=True)
class Behaviour:
    id:            str
    label:         str
    description:   str
    needs_target:  bool
    intent_fn:     SkillFn
    compliance_fn: ComplianceFn | None = None


# ── Shared math helpers ───────────────────────────────────────────────────────

def angle_to(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def pd_speed(robot_id: int, pos: tuple[float, float],
             dx: float, dy: float, dist: float) -> float:
    now = time.monotonic()
    closing_vel = 0.0
    if robot_id in approach_cache:
        prev_pos, prev_t = approach_cache[robot_id]
        dt = now - prev_t
        if dt > 1e-6 and dist > 1e-6:
            vx = (pos[0] - prev_pos[0]) / dt
            vy = (pos[1] - prev_pos[1]) / dt
            closing_vel = vx * (dx / dist) + vy * (dy / dist)
    approach_cache[robot_id] = (pos, now)
    return max(0.0, min(KP * dist - KD * closing_vel, MAX_SPEED_M_S))


def nudge_away(robot_pos: tuple[float, float],
               ball_pos:  tuple[float, float],
               clearance: float) -> tuple[float, float]:
    dx = robot_pos[0] - ball_pos[0]
    dy = robot_pos[1] - ball_pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (robot_pos[0] + clearance, robot_pos[1])
    return (ball_pos[0] + dx / dist * clearance,
            ball_pos[1] + dy / dist * clearance)


# ── Kick helpers (used by kick_at_goal, kick_at_point, move_then_attack) ──────

def kick_sequence(snap: Snapshot, robot: RobotState,
                  kick_target: tuple[float, float]):
    """3-phase: position behind ball → align heading → strike. Returns Intent | None."""
    from TeamControl.bt.contracts.intent import IntentKick, IntentMove, IntentOrient
    ball = snap.ball_position
    rid  = robot.robot_id

    tdx, tdy  = kick_target[0] - ball[0], kick_target[1] - ball[1]
    tdist = math.hypot(tdx, tdy)
    if tdist < 1e-6:
        return None
    kux, kuy      = tdx / tdist, tdy / tdist
    kick_heading  = math.atan2(kuy, kux)
    behind_pt     = (ball[0] - kux * KICK_BEHIND_DIST_M,
                     ball[1] - kuy * KICK_BEHIND_DIST_M)

    dist_to_ball   = math.hypot(ball[0] - robot.position[0], ball[1] - robot.position[1])
    dist_to_behind = math.hypot(behind_pt[0] - robot.position[0], behind_pt[1] - robot.position[1])
    heading_err    = abs(math.remainder(robot.orientation - kick_heading, 2 * math.pi))

    if dist_to_ball <= KICK_READY_DIST_M and heading_err <= KICK_ALIGN_TOL_RAD:
        kick_phase.pop(rid, None)
        approach_cache.pop(rid, None)
        return None

    phase = kick_phase.get(rid, "position")
    if phase == "position" and dist_to_behind <= KICK_APPROACH_TOL_M:
        kick_phase[rid] = phase = "align"
    if phase == "align" and heading_err <= KICK_ALIGN_TOL_RAD:
        kick_phase[rid] = phase = "strike"

    if phase == "position":
        dx, dy = behind_pt[0] - robot.position[0], behind_pt[1] - robot.position[1]
        return IntentMove(target_pos=behind_pt, target_orientation=kick_heading,
                          max_speed=pd_speed(rid, robot.position, dx, dy, dist_to_behind))
    if phase == "align":
        approach_cache.pop(rid, None)
        return IntentOrient(target_orientation=kick_heading)
    approach_cache.pop(rid, None)
    return IntentMove(target_pos=ball, target_orientation=kick_heading,
                      max_speed=KICK_STRIKE_SPEED_M_S)


def seq_move_then_kick(snap: Snapshot, robot: RobotState,
                       kick_target: tuple[float, float]):
    """Approach ball, then run kick_sequence. Returns Intent | None (None = fire kick)."""
    from TeamControl.bt.contracts.intent import IntentKick
    from TeamControl.skills.move_to_ball import move_to_ball
    rid  = robot.robot_id
    ball = snap.ball_position
    dist = math.hypot(ball[0] - robot.position[0], ball[1] - robot.position[1])
    phase = seq_phase.get(rid, "approach")

    if phase == "approach" and dist <= BALL_APPROACH_OFFSET_M:
        seq_phase[rid] = phase = "kick"
    if phase == "kick" and dist > BALL_APPROACH_OFFSET_M * 2:
        seq_phase[rid] = "approach"
        kick_phase.pop(rid, None)
        return move_to_ball(snap, robot, None)
    if phase == "approach":
        return move_to_ball(snap, robot, None)

    prep = kick_sequence(snap, robot, kick_target)
    if prep is not None:
        return prep
    seq_phase.pop(rid, None)
    return IntentKick(target_pos=kick_target)
