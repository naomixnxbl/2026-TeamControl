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

# ── Attack direction (side-awareness) ─────────────────────────────────────────
# +1.0 means we attack toward +x (opponent goal at +HALF_LEN_M, own goal at
# -HALF_LEN_M); -1.0 flips it. The Skill Lab sets this per selected team from the
# configured us_positive/us_yellow so every skill attacks/defends the correct
# goal. Default +1.0 keeps the historical convention (and unit tests) unchanged.
_ATTACK_SIGN: float = 1.0


def set_attack_sign(sign: float) -> None:
    """Set the global attack direction (+1 = attack +x, -1 = attack -x)."""
    global _ATTACK_SIGN
    _ATTACK_SIGN = 1.0 if sign >= 0 else -1.0


def attack_sign() -> float:
    """Current attack direction: +1 attacking +x, -1 attacking -x."""
    return _ATTACK_SIGN


def opp_goal() -> tuple[float, float]:
    """Centre of the opponent goal we attack."""
    return (_ATTACK_SIGN * HALF_LEN_M, 0.0)


def own_goal() -> tuple[float, float]:
    """Centre of our own goal we defend."""
    return (-_ATTACK_SIGN * HALF_LEN_M, 0.0)


def forward_progress(x: float) -> float:
    """Signed progress of an x-coordinate toward the opponent goal (higher = further forward)."""
    return x * _ATTACK_SIGN


def in_own_half(x: float) -> bool:
    """True when x is on our defensive half (the own-goal side of centre)."""
    return x * _ATTACK_SIGN < 0.0

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
# Robot centre → ball centre distance at which the strike phase fires the kick.
# The kicker plate sits at the robot's front, so the closest a robot centre can
# get to the ball centre is ROBOT_RADIUS (0.09 m) + ball radius (~0.02 m) ≈
# 0.11 m — a value below that (the old 0.05 m) can NEVER be reached, so the kick
# never fired. 0.13 m clears the contact distance with margin while staying
# under the Skill Lab's 0.15 m command gate so the kick actually triggers.
KICK_READY_DIST_M:     float = 0.13

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

# ── Forward-outlet (counter-attack release) tuning ────────────────────────────
# A pass outlet must be at least this far ahead (toward the opponent goal) of the
# carrier — a genuine forward pass, not a square/back ball.
FORWARD_MIN_ADVANCE_M: float = 0.5
# ...and its nearest opponent at least this far away — not tightly marked.
OUTLET_MARK_CLEAR_M:   float = 0.5
# Pass-lane half-width that must be clear of any robot.
OUTLET_LANE_CLEAR_M:   float = 0.18

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


# ── Attack targeting helpers (shared by the counter-attack skills) ────────────

def point_to_segment_dist(p: tuple[float, float],
                          a: tuple[float, float],
                          b: tuple[float, float]) -> float:
    """Shortest distance from point *p* to the segment A→B."""
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def best_goal_target(snap: Snapshot) -> tuple[float, float]:
    """Most open aim point across the opponent goal mouth (away from the keeper).

    Samples points across the mouth and picks the one whose ball→point line is
    farthest from any opponent; ties break toward the centre. Side-aware via the
    global attack direction. Falls back to the goal centre when no opponents.
    """
    goal = opp_goal()
    if not snap.enemy_robots:
        return goal
    ball = snap.ball_position
    best_pt, best_score = goal, None
    for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):
        pt = (goal[0], goal[1] + frac * GOAL_HW_M)
        clearance = min(
            point_to_segment_dist(o.position, ball, pt) for o in snap.enemy_robots
        )
        score = (clearance, -abs(frac))
        if best_score is None or score > best_score:
            best_score, best_pt = score, pt
    return best_pt


def forward_outlet(snap: Snapshot, robot: RobotState) -> tuple[float, float] | None:
    """Most-advanced open teammate ahead of the carrier on a clear lane, or None.

    "Ahead" and "advanced" are measured toward the opponent goal via the global
    attack direction, so this works on either side of the field.
    """
    from TeamControl.bt.tactics.line_of_sight import line_of_sight_clear

    ball = snap.ball_position
    carrier_progress = forward_progress(robot.position[0])
    best_pos: tuple[float, float] | None = None
    best_progress = -math.inf
    for tm in snap.own_robots:
        if tm.robot_id == robot.robot_id:
            continue
        progress = forward_progress(tm.position[0])
        if progress < carrier_progress + FORWARD_MIN_ADVANCE_M:
            continue
        nearest_opp = min(
            (
                math.hypot(tm.position[0] - o.position[0], tm.position[1] - o.position[1])
                for o in snap.enemy_robots
            ),
            default=math.inf,
        )
        if nearest_opp < OUTLET_MARK_CLEAR_M:
            continue
        obstacles = list(snap.enemy_robots) + [
            r for r in snap.own_robots if r.robot_id not in (robot.robot_id, tm.robot_id)
        ]
        if not line_of_sight_clear(
            ball, tm.position, obstacles, clearance=OUTLET_LANE_CLEAR_M
        ):
            continue
        if progress > best_progress:
            best_progress = progress
            best_pos = tm.position
    return best_pos


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
