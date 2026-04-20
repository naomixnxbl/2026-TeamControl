"""
Shared kick engine — reliable kick-to-target for any robot behavior.

Usage:
    ks = KickState()

    # Each tick:
    result = kick_tick(ks, me, ball, aim, now, rel_ball, d_ball)
    vx, vy, w       = result.vx, result.vy, result.w
    kick, dribble    = result.kick, result.dribble
    if result.kick_started:
        print("kick burst started!")

The engine handles:
  - Contact detection (ball touching dribbler)
  - Hold-and-rotate to face aim before kicking
  - Blended approach (face ball -> face aim as we close in)
  - Sustained kick burst (kick=1 for multiple ticks so grSim registers)
  - Force kick timeout (last resort if alignment takes too long)
  - Cooldown tracking
"""

import math
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import clamp, move_toward, compute_arc_nav
from TeamControl.robot.constants import (
    KICK_RANGE, BALL_NEAR, BEHIND_DIST, AVOID_RADIUS,
    CRUISE_SPEED, CHARGE_SPEED, DRIBBLE_SPEED,
    MAX_W, TURN_GAIN,
    KICK_COOLDOWN,
)

# -- Tuning (shared across all users) ----------------------------------
CONTACT_DIST    = 130       # mm — ball touching dribbler (90 robot + 21 ball + margin)
KICK_ALIGN_TOL  = 0.12      # rad (~7 deg) — must face aim before kick
KICK_BURST_T    = 0.35      # s — sustain kick=1 for this long
FORCE_KICK_TIME = 2.5       # s — force kick if stuck near ball this long
DRIBBLE_SPD     = DRIBBLE_SPEED
APPROACH_SPD    = CRUISE_SPEED


class KickState:
    """Persistent state for the kick engine — one per robot."""
    __slots__ = (
        "near_ball_since", "last_kick", "committed_side",
        "kick_end_time", "bursting",
    )

    def __init__(self):
        self.near_ball_since = 0.0
        self.last_kick = 0.0
        self.committed_side = None
        self.kick_end_time = 0.0
        self.bursting = False

    def reset(self):
        self.near_ball_since = 0.0
        self.committed_side = None
        self.bursting = False


class KickResult:
    """Output of a single kick_tick call."""
    __slots__ = (
        "vx", "vy", "w", "kick", "dribble",
        "kick_started", "burst_done", "committed_side",
    )

    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.w = 0.0
        self.kick = 0
        self.dribble = 0
        self.kick_started = False
        self.burst_done = False
        self.committed_side = None


def kick_tick(ks, me, ball, aim, now, rel_ball=None, d_ball=None):
    """
    Run one tick of the kick engine.

    Parameters
    ----------
    ks       : KickState — persistent state (mutated in place)
    me       : (x, y, orientation) — robot pose
    ball     : (x, y) — ball world position, or None
    aim      : (x, y) — world target to kick toward (mate, goal, etc.)
    now      : float — current time
    rel_ball : (rx, ry) — ball in robot-local coords (optional, computed if None)
    d_ball   : float — distance to ball (optional, computed if None)

    Returns
    -------
    KickResult with vx, vy, w, kick, dribble, and status flags.
    """
    r = KickResult()

    if ball is None:
        ks.bursting = False
        return r

    # Compute robot-local ball if not provided
    if rel_ball is None:
        rel_ball = world2robot(me, ball)
    if d_ball is None:
        d_ball = math.hypot(rel_ball[0], rel_ball[1])

    ang_ball = math.atan2(rel_ball[1], rel_ball[0])
    rel_aim = world2robot(me, aim)
    ang_aim = math.atan2(rel_aim[1], rel_aim[0])

    # ==================================================================
    #  BURST MODE — sustained kick already in progress
    # ==================================================================
    if ks.bursting:
        r.dribble = 1
        r.w = clamp(ang_aim * TURN_GAIN, -MAX_W, MAX_W)

        if d_ball < CONTACT_DIST + 30:
            r.kick = 1
        # Drive toward ball to maintain contact
        r.vx, r.vy = move_toward(rel_ball, CHARGE_SPEED,
                                  ramp_dist=100, stop_dist=0)

        if now > ks.kick_end_time:
            ks.bursting = False
            r.burst_done = True

        r.committed_side = ks.committed_side
        return r

    # ==================================================================
    #  P1: Ball touching dribbler — hold + rotate to face aim
    # ==================================================================
    if d_ball < CONTACT_DIST and rel_ball[0] > 0:
        r.dribble = 1
        # Gentle forward pressure to hold ball, mostly rotate
        r.vx = DRIBBLE_SPD * 0.3
        r.vy = 0.0
        r.w = clamp(ang_aim * TURN_GAIN * 1.5, -MAX_W, MAX_W)

        if ks.near_ball_since == 0.0:
            ks.near_ball_since = now
        force_kick = (now - ks.near_ball_since) > FORCE_KICK_TIME
        can_kick = (now - ks.last_kick) > KICK_COOLDOWN
        aligned = abs(ang_aim) < KICK_ALIGN_TOL

        if can_kick and (aligned or force_kick):
            # Start kick burst
            ks.bursting = True
            ks.kick_end_time = now + KICK_BURST_T
            ks.last_kick = now
            ks.near_ball_since = 0.0
            ks.committed_side = None
            r.kick = 1
            r.kick_started = True
            r.vx, r.vy = move_toward(rel_ball, CHARGE_SPEED,
                                      ramp_dist=100, stop_dist=0)

        r.committed_side = ks.committed_side
        return r

    # ==================================================================
    #  P2: Ball close + in front — drive in with dribbler, start aligning
    # ==================================================================
    if d_ball < KICK_RANGE and rel_ball[0] > 0:
        r.dribble = 1
        r.vx, r.vy = move_toward(rel_ball, DRIBBLE_SPD,
                                  ramp_dist=150, stop_dist=0)
        # Blend rotation: face ball far out, face aim when close
        blend = max(0.0, 1.0 - d_ball / KICK_RANGE)
        w_ball = ang_ball * TURN_GAIN
        w_aim = ang_aim * TURN_GAIN
        r.w = clamp(w_ball * (1.0 - blend) + w_aim * blend,
                     -MAX_W, MAX_W)

        if ks.near_ball_since == 0.0:
            ks.near_ball_since = now

        r.committed_side = ks.committed_side
        return r

    # ==================================================================
    #  P3: Arc approach — get behind ball relative to aim, drive in
    # ==================================================================
    ks.near_ball_since = 0.0

    nav, ks.committed_side, is_behind = compute_arc_nav(
        robot_xy=(me[0], me[1]),
        ball=ball, aim=aim,
        behind_dist=BEHIND_DIST,
        avoid_radius=AVOID_RADIUS,
        committed_side=ks.committed_side,
    )

    rel_nav = world2robot(me, nav)
    d_nav = math.hypot(rel_nav[0], rel_nav[1])

    if is_behind and d_nav < 300 and d_ball < BALL_NEAR:
        r.dribble = 1
        r.vx, r.vy = move_toward(rel_ball, DRIBBLE_SPD,
                                  ramp_dist=300, stop_dist=0)
        r.w = clamp(ang_aim * TURN_GAIN * 0.7, -MAX_W, MAX_W)
    else:
        r.vx, r.vy = move_toward(rel_nav, APPROACH_SPD,
                                  ramp_dist=400, stop_dist=10)
        r.w = clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)

        # Push away from ball while arcing to avoid knocking it
        if d_ball < AVOID_RADIUS:
            push_strength = APPROACH_SPD * 0.8 * (1.0 - d_ball / AVOID_RADIUS)
            if d_ball > 1.0:
                r.vx += (-rel_ball[0] / d_ball) * push_strength
                r.vy += (-rel_ball[1] / d_ball) * push_strength

    r.committed_side = ks.committed_side
    return r
