"""
Elite Team AI coordinator for 6v6 SSL.

Every tick: read world → detect possession → assign roles → compute targets → send commands.

Tactical system:
  ATTACK       – we control ball → push up, triangles, through-balls, shoot/pass decision
  COUNTER      – just won the ball → sprint forward, exploit gaps before defense resets
  DEFEND       – opponents control → fall back, mark, coordinated pressing
  CONTEST      – loose ball → balanced shape, sprint to intercept

Key improvements:
  - Counter-attack detection: fast transitions when we gain possession
  - Coordinated pressing: 2-robot press when opponent has ball in their half
  - Through-ball passing: passes into space ahead of running teammates
  - Dynamic formation: shape adapts to ball position and possession state
  - Smarter support: better spacing, more passing triangles
  - Aggressive goalie distribution to teammates
"""

import time
import math

from TeamControl.network.robot_command import RobotCommand
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import compute_arc_nav, predict_ball as _predict_ball
from TeamControl.robot.constants import (
    FIELD_LENGTH, FIELD_WIDTH, HALF_LEN, HALF_WID,
    GOAL_WIDTH, GOAL_HW, GOAL_DEPTH,
    PENALTY_DEPTH, PENALTY_HW,
    DEFENSE_DEPTH, DEFENSE_HALF_WIDTH,
    MAX_W, TURN_GAIN,
    KICK_RANGE, BALL_NEAR, BEHIND_DIST, AVOID_RADIUS,
    FRICTION, INTERCEPT_STEPS,
    PRESSURE_DIST, KICK_COOLDOWN, BALL_HISTORY_SIZE,
    MAX_ADVANCE, SHOT_SPEED, CLEAR_BALL_SPEED, CLEAR_BALL_DIST,
    LOOP_RATE, FRAME_INTERVAL,
)

# ════════════════════════════════════════════════════════════════════
#  ALIASES (team.py naming convention → central constants)
# ════════════════════════════════════════════════════════════════════

FIELD_LEN    = FIELD_LENGTH
FIELD_WID    = FIELD_WIDTH
GOAL_W       = GOAL_WIDTH
DEF_DEPTH    = DEFENSE_DEPTH
DEF_HW       = DEFENSE_HALF_WIDTH
AVOID_R      = AVOID_RADIUS
LOOP_DT      = LOOP_RATE
FRAME_DT     = FRAME_INTERVAL
BHIST_N      = BALL_HISTORY_SIZE
GK_MAX_ADV   = MAX_ADVANCE
GK_SHOT_SPEED  = int(SHOT_SPEED)
GK_CLEAR_SPEED = int(CLEAR_BALL_SPEED)
GK_CLEAR_DIST  = int(CLEAR_BALL_DIST)

DEF_MARGIN   = 200
DEF_MARGIN_D = 450
FIELD_MARGIN = 180

# ════════════════════════════════════════════════════════════════════
#  SPEEDS  (m/s) — team-specific role speeds
# ════════════════════════════════════════════════════════════════════

SPD_SPRINT   = 2.5
SPD_CRUISE   = 1.8
SPD_APPROACH = 1.4
SPD_DRIBBLE  = 1.1
SPD_POSSESS  = 1.6
SPD_POSITION = 1.5
SPD_SAVE     = 2.5
SPD_CLEAR    = 1.4
SPD_COUNTER  = 2.3
SPD_PRESS    = 2.0

# ════════════════════════════════════════════════════════════════════
#  BALL PHYSICS (team-specific extras)
# ════════════════════════════════════════════════════════════════════

PREDICT_STEPS = INTERCEPT_STEPS
PREDICT_DT    = 0.10
BALL_STOP_V   = 50

# ════════════════════════════════════════════════════════════════════
#  TACTICAL  (mm / seconds / scores)
# ════════════════════════════════════════════════════════════════════

PASS_LANE_CLR     = 320
PASS_SPEED        = 3200
INTERCEPT_MARGIN  = 0.05
MAX_ROBOT_SPD     = 2400
RECV_MARKED_DIST  = 380
MAX_SHOT_DIST     = 4200
HAS_SPACE_DIST    = 800

SHOOT_THRESH      = 0.16
PASS_THRESH       = 0.05

SUPPORT_MIN_BALL  = 2000
SUPPORT_MIN_MATE  = 1200
SUPPORT_MIN_ATK   = 1300
DEFEND_LINE       = 1800
DEFEND_SPREAD     = 1400
DEFEND_MARK_RATIO = 0.48

WINNER_HYST       = 0.45      # seconds advantage needed to steal winner role
POSSESS_DIST      = 500
WINNER_HOLD_MIN   = 0.6       # seconds — minimum time before winner can change

COUNTER_WINDOW    = 2.0
COUNTER_ADVANCE   = 1800

PRESS_ZONE        = 0.3
PRESS_DIST        = 600

THROUGH_LEAD      = 800

# ════════════════════════════════════════════════════════════════════
#  GOALIE (team-specific extras)
# ════════════════════════════════════════════════════════════════════

GK_DANGER      = 2500
GK_PASS_CLR    = 450


# ════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ════════════════════════════════════════════════════════════════════

def _cl(v, lo, hi):
    return max(lo, min(hi, v))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _norm(dx, dy):
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 1e-9 else (0.0, 0.0)


def _dot(ax, ay, bx, by):
    return ax * bx + ay * by


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = _cl(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _seg_closest(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return ax, ay
    t = _cl(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
    return ax + t * dx, ay + t * dy


# ════════════════════════════════════════════════════════════════════
#  FIELD HELPERS
# ════════════════════════════════════════════════════════════════════

def _in_def(x, y, gx):
    return abs(x - gx) < DEF_DEPTH and abs(y) < DEF_HW


def _in_penalty(x, y, gx):
    """True if (x, y) is inside the penalty box around goal at gx."""
    return abs(x - gx) < PENALTY_DEPTH and abs(y) < PENALTY_HW


def _field_clamp(x, y):
    return (_cl(x, -HALF_LEN + FIELD_MARGIN, HALF_LEN - FIELD_MARGIN),
            _cl(y, -HALF_WID + FIELD_MARGIN, HALF_WID - FIELD_MARGIN))


def _push_out(x, y, gx, margin=DEF_MARGIN):
    if not _in_def(x, y, gx):
        return x, y
    inward = -1 if gx > 0 else 1
    return gx + inward * (DEF_DEPTH + margin), y


def _push_out_def(x, y, our_gx):
    inward = -1 if our_gx > 0 else 1
    if _in_def(x, y, our_gx):
        x = our_gx + inward * (DEF_DEPTH + DEF_MARGIN_D)
        if abs(y) > DEF_HW * 0.5:
            x = our_gx + inward * (DEF_DEPTH + DEF_MARGIN_D + 350)
    else:
        if abs(y) > DEF_HW * 0.55 and abs(x - our_gx) < DEF_DEPTH + DEF_MARGIN_D:
            x = our_gx + inward * (DEF_DEPTH + DEF_MARGIN_D + 350)
    return x, y


def _gk_clamp(x, y, gx):
    """Clamp goalie position to stay inside the penalty box."""
    margin = 50
    if gx > 0:
        x = _cl(x, gx - PENALTY_DEPTH + margin, gx - margin)
    else:
        x = _cl(x, gx + margin, gx + PENALTY_DEPTH - margin)
    return x, _cl(y, -PENALTY_HW + margin, PENALTY_HW - margin)


def _safe_pos(x, y, our_gx, opp_gx):
    x, y = _push_out(x, y, our_gx)
    x, y = _push_out(x, y, opp_gx)
    return _field_clamp(x, y)


def _safe_pos_def(x, y, our_gx, opp_gx):
    x, y = _push_out_def(x, y, our_gx)
    x, y = _push_out(x, y, opp_gx)
    return _field_clamp(x, y)


# ════════════════════════════════════════════════════════════════════
#  BALL PHYSICS
# ════════════════════════════════════════════════════════════════════


def _optimal_intercept(rpos, ball, bvel):
    spd = math.hypot(bvel[0], bvel[1])
    d0 = _dist(rpos, ball)
    best_pt, best_t = ball, d0 / max(MAX_ROBOT_SPD, 1)
    if spd < BALL_STOP_V:
        return best_pt, best_t
    for i in range(1, PREDICT_STEPS + 1):
        dt = i * PREDICT_DT
        bp = _predict_ball(ball, bvel, dt)
        tr = _dist(rpos, bp) / max(MAX_ROBOT_SPD, 1)
        if tr <= dt and tr < best_t:
            best_pt, best_t = bp, tr
    return best_pt, best_t


# ════════════════════════════════════════════════════════════════════
#  TACTICAL EVALUATION
# ════════════════════════════════════════════════════════════════════

def _lane_clear(a, b, opps, margin=PASS_LANE_CLR):
    for ox, oy in opps:
        if _seg_dist(ox, oy, a[0], a[1], b[0], b[1]) < margin:
            return False
    return True


def _would_intercept(ball, recv, opps):
    d = _dist(ball, recv)
    if d < 1:
        return False
    t_ball = d / PASS_SPEED
    for ox, oy in opps:
        cx, cy = _seg_closest(ox, oy, ball[0], ball[1], recv[0], recv[1])
        t_opp = _dist((ox, oy), (cx, cy)) / MAX_ROBOT_SPD
        if t_opp < t_ball + INTERCEPT_MARGIN:
            return True
    return False


def _recv_marked(recv, opps):
    return any(_dist(recv, o) < RECV_MARKED_DIST for o in opps)


def _pass_safe(ball, recv, opps, margin=PASS_LANE_CLR):
    if not _lane_clear(ball, recv, opps, margin):
        return False
    if _would_intercept(ball, recv, opps):
        return False
    if _recv_marked(recv, opps):
        return False
    return True


def _shot_score(pos, opp_gx, opps, gk_pos):
    dx = abs(pos[0] - opp_gx)
    if dx > MAX_SHOT_DIST:
        return 0.0
    dx = max(dx, 100)

    angle = 2.0 * math.atan2(GOAL_HW, dx)
    angle_s = min(angle / math.radians(30), 1.0)
    dist_s = max(0.0, 1.0 - dx / MAX_SHOT_DIST)

    block = 1.0
    for ox, oy in opps:
        d = _seg_dist(ox, oy, pos[0], pos[1], opp_gx, 0)
        if d < 170:
            block *= 0.10
        elif d < 260:
            block *= 0.25
        elif d < 450:
            block *= 0.50

    gk_pen = 0.15 if (gk_pos and abs(gk_pos[0] - opp_gx) < 500) else 0.0

    sc = _cl(angle_s * 0.30 + dist_s * 0.35 + block * 0.35 - gk_pen, 0.0, 1.0)
    return 0.0 if block < 0.18 else sc


def _pass_score(ball, recv, opps, opp_gx):
    if not _pass_safe(ball, recv, opps):
        return 0.0
    d = _dist(ball, recv)
    if d < 450 or d > FIELD_LEN * 1.1:
        return 0.0

    dist_s = 1.0 if 800 < d < 3500 else 0.40

    atk = 1 if opp_gx > 0 else -1
    advance = (recv[0] - ball[0]) * atk
    adv_s = _cl(0.5 + advance / 3000, 0, 1)

    recv_shot = _shot_score(recv, opp_gx, opps, None)

    near_opp = min((_dist(recv, o) for o in opps), default=9999)
    space = _cl((near_opp - RECV_MARKED_DIST) / 1500, 0, 0.30)

    return _cl(dist_s * 0.18 + adv_s * 0.30 + recv_shot * 0.37 + space + 0.05, 0.0, 1.0)


def _through_ball_score(ball, mate_pos, mate_vel_est, opps, opp_gx):
    """Score a through-ball into space ahead of a running teammate."""
    atk = 1 if opp_gx > 0 else -1
    # Target: ahead of teammate in attacking direction
    lead_x = mate_pos[0] + atk * THROUGH_LEAD
    lead_y = mate_pos[1] * 0.8  # drift toward center
    lead = _safe_pos(lead_x, lead_y, -HALF_LEN, HALF_LEN)

    if not _pass_safe(ball, lead, opps, PASS_LANE_CLR * 0.9):
        return 0.0, None

    d = _dist(ball, lead)
    if d < 600 or d > FIELD_LEN:
        return 0.0, None

    # Bonus for advancing position
    advance = (lead[0] - ball[0]) * atk
    if advance < 300:
        return 0.0, None

    adv_s = _cl(advance / 3000, 0, 1)
    shot = _shot_score(lead, opp_gx, opps, None)
    space = min((_dist(lead, o) for o in opps), default=9999)
    space_s = _cl((space - 300) / 1500, 0, 0.3)

    sc = _cl(adv_s * 0.35 + shot * 0.40 + space_s + 0.05, 0.0, 1.0)
    return sc, lead


def _pick_shot_target(ball, opp_gx, gk_pos):
    # Aim past the goal line into the net so the ball enters through the
    # open face, not into the side walls of the [ shaped goal.
    aim_inward = 1 if opp_gx > 0 else -1
    aim_x = opp_gx + aim_inward * (GOAL_DEPTH * 0.5)
    h = GOAL_HW * 0.35
    if gk_pos:
        return (aim_x, -h) if gk_pos[1] > 0 else (aim_x, h)
    return (aim_x, -h * 0.6) if ball[1] > 0 else (aim_x, h * 0.6)


def _best_pass(ball, rid, our, opps, opp_gx):
    best_sc, best_tgt, best_tid = 0.0, None, None

    for tid, tp in our.items():
        if tid == rid:
            continue
        # Standard pass
        sc = _pass_score(ball, tp, opps, opp_gx)
        if sc > best_sc:
            best_sc, best_tgt, best_tid = sc, tp, tid

        # Through-ball: pass into space ahead
        tb_sc, tb_tgt = _through_ball_score(ball, tp, None, opps, opp_gx)
        if tb_sc > best_sc:
            best_sc, best_tgt, best_tid = tb_sc, tb_tgt, tid

    return best_sc, best_tgt, best_tid


# ════════════════════════════════════════════════════════════════════
#  POSSESSION & GAME STATE DETECTION
# ════════════════════════════════════════════════════════════════════

class _PossessionTracker:
    """Track possession state with hysteresis for counter-attack detection."""
    __slots__ = ('state', 'gained_time', 'lost_time')

    def __init__(self):
        self.state = 'CONTEST'  # ATTACK, DEFEND, CONTEST, COUNTER
        self.gained_time = 0.0
        self.lost_time = 0.0

    def update(self, we_have, opp_has, now):
        prev = self.state

        if we_have:
            if prev == 'DEFEND' or prev == 'CONTEST':
                # Just won the ball → counter-attack window
                self.state = 'COUNTER'
                self.gained_time = now
            elif prev == 'COUNTER':
                if now - self.gained_time > COUNTER_WINDOW:
                    self.state = 'ATTACK'
            # else already ATTACK, stay
        elif opp_has:
            self.state = 'DEFEND'
            self.lost_time = now
        else:
            if prev == 'COUNTER' and now - self.gained_time < COUNTER_WINDOW:
                pass  # maintain counter
            else:
                self.state = 'CONTEST'


# ════════════════════════════════════════════════════════════════════
#  ROLE ASSIGNMENT
# ════════════════════════════════════════════════════════════════════

def _pick_winner(ball, bvel, our, gk, prev_winner, winner_since, now):
    """Pick the field robot that should chase the ball.

    Uses intercept-time ranking with hysteresis **and** a minimum hold timer
    so that the winner doesn't flip-flop every tick when two robots are
    roughly equidistant from the ball.

    Returns (winner_id, winner_since).
    """
    fids = [r for r in our if r != gk]
    if not fids:
        return None, now
    times = {}
    for r in fids:
        _, t = _optimal_intercept(our[r], ball, bvel)
        times[r] = t
    best = min(fids, key=lambda r: times[r])

    if prev_winner is not None and prev_winner in times:
        # Enforce minimum hold time — don't even consider switching yet
        if (now - winner_since) < WINNER_HOLD_MIN:
            return prev_winner, winner_since

        tb = times[best]
        tp = times[prev_winner]
        # Only switch if the new candidate is clearly better
        if tb < tp - WINNER_HYST:
            return best, now
        if tp > tb + 0.7:
            return best, now
        return prev_winner, winner_since
    return best, now


def _split_field(ball, our, gk, winner, our_gx, opp_gx, poss_state):
    """Split field players into support / defender lists.
    Counter-attack mode sends more players forward."""
    atk = 1 if opp_gx > 0 else -1
    fids = [r for r in our if r != gk and r != winner]
    if not fids:
        return [], []
    adv = (ball[0] - our_gx) * atk
    ratio = adv / FIELD_LEN

    if poss_state == 'COUNTER':
        n_sup = min(3, len(fids))  # Everyone forward on counter!
    elif poss_state == 'ATTACK':
        n_sup = 2 if ratio > -0.05 else 1
    elif poss_state == 'DEFEND':
        n_sup = 1 if ratio > 0.1 else 0
    else:
        n_sup = 2 if ratio > 0 else 1

    n_sup = min(n_sup, len(fids))
    fids.sort(key=lambda r: -our[r][0] * atk)
    return fids[:n_sup], fids[n_sup:]


# ════════════════════════════════════════════════════════════════════
#  SUPPORT POSITIONING — potential-field scored candidates
# ════════════════════════════════════════════════════════════════════

_SUP_OFFSETS = [
    (1800,  1600), (1800, -1600), (2000,  0),
    (2400,  1200), (2400, -1200),
    (1500,  2000), (1500, -2000),
    (2200,  800),  (2200, -800),
    (2800,  500),  (2800, -500),
    (1300,  1400), (1300, -1400),
    (1700,  1000), (1700, -1000),
    (3000,  200),  (3000, -200),
    (2600,  1600), (2600, -1600),
    (1100,  2400), (1100, -2400),
]

# Counter-attack: more aggressive forward positions
_COUNTER_OFFSETS = [
    (2500,  1000), (2500, -1000),
    (3000,  500),  (3000, -500),
    (3500,  0),
    (2800,  1500), (2800, -1500),
    (3200,  800),  (3200, -800),
    (2000,  1800), (2000, -1800),
]


def _support_targets(ball, our, opps, our_gx, opp_gx, sids, winner, poss_state):
    atk = 1 if opp_gx > 0 else -1

    we_have = False
    if winner is not None and winner in our:
        we_have = _dist(our[winner], ball) < POSSESS_DIST

    offsets = _COUNTER_OFFSETS if poss_state == 'COUNTER' else _SUP_OFFSETS

    cands = []
    for dx, dy in offsets:
        cx, cy = ball[0] + dx * atk, ball[1] + dy
        cx, cy = _safe_pos(cx, cy, our_gx, opp_gx)
        cands.append((cx, cy))

    others = [our[r] for r in our if r not in sids]

    def score(cx, cy):
        s = 0.0
        db = _dist(ball, (cx, cy))

        if db < 1000:
            s -= 6.0
        elif db < SUPPORT_MIN_BALL:
            s -= 2.5

        if we_have and winner in our:
            da = _dist((cx, cy), our[winner])
            if da < SUPPORT_MIN_ATK:
                s -= 4.5

        if _lane_clear(ball, (cx, cy), opps, PASS_LANE_CLR):
            s += 4.5
        else:
            s -= 2.5

        s += _shot_score((cx, cy), opp_gx, opps, None) * 4.0

        if 1600 < db < 3500:
            s += 3.0
        elif 1000 < db < FIELD_LEN * 0.5:
            s += 1.2

        # Reward positions that advance toward opponent goal
        s += max(0, 1 - abs(cx - opp_gx) / (FIELD_LEN * 0.6)) * 2.5

        # Counter-attack bonus for very forward positions
        if poss_state == 'COUNTER':
            fwd = (cx - ball[0]) * atk
            if fwd > 1500:
                s += 3.0
            elif fwd > 800:
                s += 1.5

        for tp in others:
            td = _dist((cx, cy), tp)
            if td < SUPPORT_MIN_MATE:
                s -= 3.5

        for ox, oy in opps:
            if _dist((cx, cy), (ox, oy)) < 400:
                s -= 2.0

        return s

    scored = sorted([(score(cx, cy), cx, cy) for cx, cy in cands], reverse=True)

    out, placed = {}, []
    for sid in sids:
        for _, cx, cy in scored:
            if not any(_dist((cx, cy), p) < SUPPORT_MIN_MATE for p in placed):
                placed.append((cx, cy))
                out[sid] = (cx, cy)
                break
        if sid not in out:
            fy = 1800 * (1 if len(placed) % 2 == 0 else -1)
            fb = _safe_pos(ball[0] + atk * 2500, ball[1] + fy, our_gx, opp_gx)
            out[sid] = fb
            placed.append(fb)
    return out


# ════════════════════════════════════════════════════════════════════
#  DEFENDER POSITIONING
# ════════════════════════════════════════════════════════════════════

def _defender_targets(ball, our, opps, our_gx, opp_gx, dids, we_have, poss_state):
    atk = 1 if opp_gx > 0 else -1
    out = {}

    if we_have and dids:
        # Push defenders up when we have the ball
        line_dist = DEFEND_LINE if poss_state != 'COUNTER' else DEFEND_LINE * 0.6
        for i, did in enumerate(dids):
            yo = DEFEND_SPREAD * (1 if i % 2 == 0 else -1)
            dx = our_gx - atk * line_dist
            dx, _ = _field_clamp(dx, 0)
            pos = _safe_pos_def(dx, yo, our_gx, opp_gx)
            out[did] = pos
        return out

    sorted_opp = sorted(opps, key=lambda o: abs(o[0] - our_gx))
    ball_in_ours = (ball[0] - our_gx) * atk < 0
    opp_on_ball = any(_dist(ball, o) < 350 for o in opps)

    # ── Coordinated pressing ──────────────────────────────────
    press_ids = []
    if poss_state == 'DEFEND' and dids:
        ball_advance = (ball[0] - our_gx) * atk / FIELD_LEN
        if ball_advance > PRESS_ZONE:
            # Ball in opponent's territory — press aggressively
            sorted_by_ball = sorted(dids, key=lambda r: _dist(our[r], ball))
            press_ids.append(sorted_by_ball[0])
            # First presser: go directly to ball carrier
            px = ball[0] + (our_gx - ball[0]) * 0.10
            py = ball[1]
            out[sorted_by_ball[0]] = _safe_pos_def(px, py, our_gx, opp_gx)

            if len(sorted_by_ball) > 1:
                # Second presser: cut off passing lane
                press_ids.append(sorted_by_ball[1])
                cut_x = ball[0] + (our_gx - ball[0]) * 0.30
                cut_y = ball[1] * 0.4  # drift toward center
                out[sorted_by_ball[1]] = _safe_pos_def(cut_x, cut_y, our_gx, opp_gx)
        elif ball_in_ours and opp_on_ball:
            # Ball in our territory — single press
            press_id = min(dids, key=lambda r: _dist(our[r], ball))
            press_ids.append(press_id)
            px = ball[0] + (our_gx - ball[0]) * 0.18
            py = ball[1]
            out[press_id] = _safe_pos_def(px, py, our_gx, opp_gx)

    # ── Remaining defenders: mark or hold shape ───────────────
    used = set()
    for did in dids:
        if did in out:
            continue
        pos = None
        for j, (ox, oy) in enumerate(sorted_opp):
            if j in used:
                continue
            if _dist((ox, oy), ball) > FIELD_LEN * 0.7:
                continue
            mx = ox + (our_gx - ox) * DEFEND_MARK_RATIO
            my = oy * 0.60
            mx, my = _field_clamp(mx, my)
            pos = _safe_pos_def(mx, my, our_gx, opp_gx)
            used.add(j)
            break
        if pos is None:
            idx = list(dids).index(did)
            yo = DEFEND_SPREAD * (1 if idx % 2 == 0 else -1)
            dx = our_gx - atk * DEFEND_LINE
            dx, _ = _field_clamp(dx, 0)
            pos = _safe_pos_def(dx, yo, our_gx, opp_gx)
        out[did] = pos
    return out


# ════════════════════════════════════════════════════════════════════
#  COMMAND BUILDER
# ════════════════════════════════════════════════════════════════════

_WALL_BRAKE_DIST = 400   # mm from field edge to start braking
_WALL_BRAKE_MIN  = 0.10  # minimum speed factor near walls


def _cmd(rid, rpos, target, face, speed, kick, dribble, yellow,
         ramp_dist=350.0, stop_dist=20.0):
    rel_t = world2robot(rpos, target)
    rel_f = world2robot(rpos, face)
    d = math.hypot(rel_t[0], rel_t[1])

    if d < stop_dist:
        vx, vy = 0.0, 0.0
    else:
        s = speed
        # Linear deceleration ramp — prevents overshoot at target
        if d < ramp_dist:
            t = (d - stop_dist) / max(ramp_dist - stop_dist, 1.0)
            s = max(speed * t, 0.10)
        ux, uy = rel_t[0] / d, rel_t[1] / d
        vx, vy = ux * s, uy * s

    # Wall braking — slow down near field edges
    dist_to_wall = min(
        HALF_LEN - abs(rpos[0]),
        HALF_WID - abs(rpos[1]),
    )
    if dist_to_wall < _WALL_BRAKE_DIST:
        wf = max(dist_to_wall / _WALL_BRAKE_DIST, _WALL_BRAKE_MIN)
        vx *= wf
        vy *= wf

    ang = math.atan2(rel_f[1], rel_f[0])
    w = 0.0 if abs(ang) < 0.04 else _cl(ang * TURN_GAIN, -MAX_W, MAX_W)

    return RobotCommand(robot_id=rid, vx=vx, vy=vy, w=w,
                        kick=kick, dribble=dribble, isYellow=yellow)


# ════════════════════════════════════════════════════════════════════
#  ATTACKER BEHAVIOUR
# ════════════════════════════════════════════════════════════════════

def _attacker(rid, rpos, ball, bvel, our, opps, our_gx, opp_gx, gk_pos,
              yellow, lkt, now, poss_state, committed_side=None):
    """Returns (RobotCommand, updated_committed_side)."""
    rel_ball = world2robot(rpos, ball)
    d_ball = math.hypot(rel_ball[0], rel_ball[1])
    atk = 1 if opp_gx > 0 else -1

    # Ball in a penalty box → field players can't enter
    for gx in (our_gx, opp_gx):
        if _in_penalty(ball[0], ball[1], gx):
            inw = -1 if gx > 0 else 1
            wait = (gx + inw * (PENALTY_DEPTH + 120),
                    _cl(ball[1], -PENALTY_HW, PENALTY_HW))
            return _cmd(rid, rpos, wait, ball, SPD_CRUISE, 0, 0, yellow), None

    # Evaluate options
    shoot_sc = _shot_score(ball, opp_gx, opps, gk_pos)
    pass_sc, pass_tgt, _ = _best_pass(ball, rid, our, opps, opp_gx)
    shot_tgt = _pick_shot_target(ball, opp_gx, gk_pos)

    near_opp = min((_dist(ball, o) for o in opps), default=9999)
    pressed = near_opp < PRESSURE_DIST

    # ── Decide action ─────────────────────────────────────────
    action = 'dribble'
    aim = (opp_gx, 0)

    if poss_state == 'COUNTER':
        if shoot_sc >= SHOOT_THRESH * 0.8:
            action, aim = 'shoot', shot_tgt
        elif pass_tgt is not None and pass_sc > PASS_THRESH * 0.7:
            action, aim = 'pass', pass_tgt
        else:
            aim = (opp_gx, ball[1] * 0.3)
            aim = _safe_pos(aim[0], aim[1], our_gx, opp_gx)
    elif shoot_sc >= SHOOT_THRESH and (pass_tgt is None or shoot_sc >= pass_sc * 0.80):
        action, aim = 'shoot', shot_tgt
    elif pass_tgt is not None and pass_sc > PASS_THRESH:
        action, aim = 'pass', pass_tgt
    elif shoot_sc > 0.10 and abs(ball[0] - opp_gx) < 2800:
        action, aim = 'shoot', shot_tgt
    else:
        aim = (ball[0] + atk * 1500, ball[1] * 0.4)
        aim = _safe_pos(aim[0], aim[1], our_gx, opp_gx)

    intercept_pt, _ = _optimal_intercept((rpos[0], rpos[1]), ball, bvel)
    cd_ok = (now - lkt) > KICK_COOLDOWN

    # ══════════════════════════════════════════════════════════
    #  POSSESS — ball in kicker range
    # ══════════════════════════════════════════════════════════
    if d_ball < KICK_RANGE and rel_ball[0] > -60:
        ang_ball = math.atan2(rel_ball[1], rel_ball[0])
        if abs(ang_ball) > 0.45:
            return _cmd(rid, rpos, ball, ball, SPD_DRIBBLE, 0, 1, yellow), None

        ra = world2robot(rpos, aim)
        ang = abs(math.atan2(ra[1], ra[0]))

        if action == 'shoot':
            if ang < 0.36 and cd_ok:
                return _cmd(rid, rpos, ball, aim, SPD_DRIBBLE, 1, 0, yellow), None
            return _cmd(rid, rpos, aim, aim, SPD_POSSESS, 0, 1, yellow), None

        if action == 'pass' and pass_tgt:
            still_safe = _pass_safe(ball, pass_tgt, opps)
            if ang < 0.48 and cd_ok and still_safe:
                return _cmd(rid, rpos, ball, pass_tgt, SPD_DRIBBLE, 1, 0, yellow), None
            if still_safe:
                return _cmd(rid, rpos, pass_tgt, pass_tgt, SPD_POSSESS, 0, 1, yellow), None
            if shoot_sc >= SHOOT_THRESH:
                ra2 = world2robot(rpos, shot_tgt)
                if abs(math.atan2(ra2[1], ra2[0])) < 0.36 and cd_ok:
                    return _cmd(rid, rpos, ball, shot_tgt, SPD_DRIBBLE, 1, 0, yellow), None
                return _cmd(rid, rpos, shot_tgt, shot_tgt, SPD_POSSESS, 0, 1, yellow), None

        if pressed and cd_ok:
            if pass_tgt and _pass_safe(ball, pass_tgt, opps):
                ra2 = world2robot(rpos, pass_tgt)
                if abs(math.atan2(ra2[1], ra2[0])) < 0.55:
                    return _cmd(rid, rpos, ball, pass_tgt, SPD_DRIBBLE, 1, 0, yellow), None
            elif ang < 0.7:
                return _cmd(rid, rpos, ball, aim, SPD_DRIBBLE, 1, 0, yellow), None

        return _cmd(rid, rpos, aim, aim, SPD_DRIBBLE, 0, 1, yellow), None

    # ══════════════════════════════════════════════════════════
    #  APPROACH — arc around ball, then drive through
    # ══════════════════════════════════════════════════════════
    tb = intercept_pt if d_ball > BALL_NEAR else ball

    # Use arc navigation with committed side for smooth approach
    nav, committed_side, is_behind = compute_arc_nav(
        robot_xy=(rpos[0], rpos[1]),
        ball=tb,
        aim=aim,
        behind_dist=BEHIND_DIST,
        avoid_radius=AVOID_R,
        committed_side=committed_side,
    )

    nav = _safe_pos(nav[0], nav[1], our_gx, opp_gx)

    d_nav = _dist((rpos[0], rpos[1]), nav)

    if is_behind and d_nav < 200 and d_ball < BALL_NEAR:
        # Lined up behind ball — approach gently so kicker contacts, not body
        return _cmd(rid, rpos, ball, ball, SPD_DRIBBLE, 0, 1, yellow,
                    ramp_dist=350), committed_side

    # Speed selection — smooth transitions, slower near ball
    if poss_state == 'COUNTER':
        speed = SPD_COUNTER if d_ball > 800 else SPD_CRUISE
    elif d_ball > 1600:
        speed = SPD_SPRINT
    elif d_ball > BALL_NEAR:
        # Smooth interpolation from SPRINT to DRIBBLE
        t = (d_ball - BALL_NEAR) / (1600.0 - BALL_NEAR)
        speed = SPD_DRIBBLE + (SPD_SPRINT - SPD_DRIBBLE) * t
    else:
        speed = SPD_DRIBBLE
    drib = 1 if d_ball < BALL_NEAR else 0

    return _cmd(rid, rpos, nav, ball, speed, 0, drib, yellow,
                ramp_dist=400), committed_side


# ════════════════════════════════════════════════════════════════════
#  SUPPORT / DEFENDER BEHAVIOUR
# ════════════════════════════════════════════════════════════════════

def _support(rid, rpos, ball, target, our_gx, opp_gx, yellow, poss_state):
    tx, ty = _safe_pos(target[0], target[1], our_gx, opp_gx)
    d = _dist((rpos[0], rpos[1]), (tx, ty))
    if poss_state == 'COUNTER':
        sp = SPD_COUNTER if d > 1000 else (SPD_CRUISE if d > 400 else 0.6)
    else:
        sp = SPD_CRUISE if d > 1500 else (SPD_POSITION if d > 400 else 0.5)
    return _cmd(rid, rpos, (tx, ty), ball, sp, 0, 0, yellow)


def _defender(rid, rpos, ball, target, our_gx, opp_gx, yellow, poss_state):
    tx, ty = _safe_pos_def(target[0], target[1], our_gx, opp_gx)
    d = _dist((rpos[0], rpos[1]), (tx, ty))
    if poss_state == 'DEFEND':
        # Faster repositioning when defending
        sp = SPD_PRESS if d > 1000 else (SPD_CRUISE if d > 400 else 0.6)
    else:
        sp = SPD_CRUISE if d > 1500 else (SPD_POSITION if d > 400 else 0.5)
    return _cmd(rid, rpos, (tx, ty), ball, sp, 0, 0, yellow)


# ════════════════════════════════════════════════════════════════════
#  GOALIE BEHAVIOUR
# ════════════════════════════════════════════════════════════════════

def _goalie(rid, rpos, ball, bvel, bspeed, opps, our, our_gx, opp_gx,
            yellow, lkt, now, smooth):
    rel_ball = world2robot(rpos, ball)
    d_ball = math.hypot(rel_ball[0], rel_ball[1])
    sign = 1 if our_gx > 0 else -1

    # ── Shot detection (with wall bounce) ──────────────────────
    shot, pred_y = False, 0.0
    if bvel[0] * sign > 70 and bspeed > GK_SHOT_SPEED and abs(bvel[0]) > 35:
        tc = (our_gx - ball[0]) / bvel[0]
        if 0 < tc < 2.0:
            py = ball[1] + bvel[1] * tc
            if abs(py) < GOAL_HW + 250:
                shot, pred_y = True, _cl(py, -GOAL_HW, GOAL_HW)

        # Wall bounce check
        if not shot and bspeed > GK_SHOT_SPEED * 1.2:
            bx, by = ball
            vx, vy = bvel
            t = 0.0
            while t < 2.0:
                bx += vx * 0.01
                by += vy * 0.01
                if by > HALF_WID:
                    by = 2 * HALF_WID - by
                    vy = -vy
                elif by < -HALF_WID:
                    by = -2 * HALF_WID - by
                    vy = -vy
                f = max(1.0 - FRICTION * 0.01, 0.0)
                vx *= f
                vy *= f
                t += 0.01
                if math.hypot(vx, vy) < 40:
                    break
                if sign > 0 and bx >= our_gx:
                    if abs(by) < GOAL_HW + 200:
                        shot, pred_y = True, _cl(by, -GOAL_HW, GOAL_HW)
                    break
                elif sign < 0 and bx <= our_gx:
                    if abs(by) < GOAL_HW + 200:
                        shot, pred_y = True, _cl(by, -GOAL_HW, GOAL_HW)
                    break

    in_box = _in_def(ball[0], ball[1], our_gx)
    ball_slow = bspeed < GK_CLEAR_SPEED
    kick, drib = 0, 0

    # ── CLEAR / DISTRIBUTE ─────────────────────────────────────
    if in_box and ball_slow and d_ball < GK_CLEAR_DIST:
        target, speed, face = ball, SPD_CLEAR, ball

        if d_ball < KICK_RANGE and rel_ball[0] > 0:
            # Smart distribution: pass to teammate if possible
            best_tgt, best_sc = None, -1
            for tid, tp in our.items():
                if tid == rid or _in_def(tp[0], tp[1], our_gx):
                    continue
                if not _pass_safe(ball, tp, opps, margin=GK_PASS_CLR):
                    continue
                sc = abs(tp[0] - our_gx) / FIELD_LEN
                # Bonus for teammates with space
                nearest_opp = min((_dist(tp, o) for o in opps), default=9999)
                if nearest_opp > 600:
                    sc += 0.2
                if sc > best_sc:
                    best_sc, best_tgt = sc, tp

            if best_tgt and (now - lkt) > KICK_COOLDOWN:
                ra = world2robot(rpos, best_tgt)
                if abs(math.atan2(ra[1], ra[0])) < 0.40:
                    kick, face = 1, best_tgt
                else:
                    drib, face = 1, best_tgt
            else:
                side_y = HALF_WID if ball[1] > 0 else -HALF_WID
                outward = -1 if our_gx > 0 else 1
                clr = (ball[0] + outward * 1500, side_y)
                rc = world2robot(rpos, clr)
                if abs(math.atan2(rc[1], rc[0])) < 0.38 and (now - lkt) > KICK_COOLDOWN:
                    kick, face = 1, clr
                else:
                    drib, face = 1, clr
        elif d_ball < 350:
            drib = 1

    # ── SAVE ───────────────────────────────────────────────────
    elif shot:
        sx = our_gx + (ball[0] - our_gx) * 0.10
        target = _gk_clamp(sx, pred_y, our_gx)
        speed, face = SPD_SAVE, ball

    # ── POSITION ───────────────────────────────────────────────
    else:
        gx = ball[0] - our_gx
        gy = ball[1]
        gd = math.hypot(gx, gy)
        bd = abs(ball[0] - our_gx)
        if gd > 1:
            ratio = 1.0 - _cl(bd / (FIELD_LEN * 0.5), 0, 1)
            if bd < GK_DANGER:
                ratio = min(1.0, ratio * 1.5)
            adv = ratio * GK_MAX_ADV
            tx = our_gx + (gx / gd) * adv
            ty = (gy / gd) * adv
            ty = _cl(ty, -(GOAL_HW + 200), GOAL_HW + 200)
            target = _gk_clamp(tx, ty, our_gx)
        else:
            target = _gk_clamp(our_gx, 0, our_gx)
        speed = SPD_CRUISE if bd < GK_DANGER else SPD_POSITION
        face = ball

    # ── Smooth ─────────────────────────────────────────────────
    alpha = 0.93 if shot else (0.78 if in_box else 0.25)
    if smooth[0] is None:
        smooth[0], smooth[1] = target[0], target[1]
    else:
        smooth[0] += alpha * (target[0] - smooth[0])
        smooth[1] += alpha * (target[1] - smooth[1])
    final = _gk_clamp(smooth[0], smooth[1], our_gx)

    return _cmd(rid, rpos, final, face, speed, kick, drib, yellow)


# ════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def run_team(is_running, dispatch_q, wm, is_yellow, goalie_id=0):
    frame       = None
    last_ft     = 0.0
    bhist       = []
    last_bxy    = None
    last_kick   = {}
    gk_smooth   = [None, None]
    prev_winner = None
    winner_since = 0.0
    poss        = _PossessionTracker()
    committed_sides = {}   # robot_id → committed arc side (+1/-1 or None)

    while is_running.is_set():
        now = time.time()

        # ── Fetch frame ────────────────────────────────────────
        if now - last_ft > FRAME_DT:
            try:
                f = wm.get_latest_frame()
                if f is not None:
                    frame = f
            except Exception:
                pass
            last_ft = now

        if frame is None or frame.ball is None:
            time.sleep(LOOP_DT)
            continue

        bp = frame.ball.position
        ball = (float(bp[0]), float(bp[1]))

        # ── Ball velocity ──────────────────────────────────────
        if last_bxy is None or ball != last_bxy:
            bhist.append((now, ball[0], ball[1]))
            if len(bhist) > BHIST_N:
                bhist.pop(0)
            last_bxy = ball

        bvx = bvy = bspeed = 0.0
        if len(bhist) >= 2:
            dt = bhist[-1][0] - bhist[0][0]
            if dt > 0.02:
                bvx = (bhist[-1][1] - bhist[0][1]) / dt
                bvy = (bhist[-1][2] - bhist[0][2]) / dt
                bspeed = math.hypot(bvx, bvy)
        bvel = (bvx, bvy)

        # ── Robot positions ────────────────────────────────────
        our_team = frame.get_yellow_robots(isYellow=is_yellow)
        opp_team = frame.get_yellow_robots(isYellow=not is_yellow)

        our_full = {}
        our_2d = {}
        for robot in our_team:
            rp = robot.position
            our_full[robot.id] = (float(rp[0]), float(rp[1]), float(rp[2]))
            our_2d[robot.id] = (float(rp[0]), float(rp[1]))

        opp_2d = {}
        for robot in opp_team:
            rp = robot.position
            opp_2d[robot.id] = (float(rp[0]), float(rp[1]))
        opps = list(opp_2d.values())

        if not our_full:
            time.sleep(LOOP_DT)
            continue

        # ── Goal positions ─────────────────────────────────────
        try:
            usp = wm.us_positive()
        except Exception:
            usp = True

        if is_yellow:
            our_gx = HALF_LEN if usp else -HALF_LEN
            opp_gx = -HALF_LEN if usp else HALF_LEN
        else:
            our_gx = -HALF_LEN if usp else HALF_LEN
            opp_gx = HALF_LEN if usp else -HALF_LEN

        # Opponent goalie
        gk_opp = None
        for op in opp_2d.values():
            if abs(op[0] - opp_gx) < DEF_DEPTH:
                gk_opp = op
                break

        gk = goalie_id if goalie_id in our_full else next(iter(our_full))

        # ── Role assignment ────────────────────────────────────
        winner, winner_since = _pick_winner(
            ball, bvel, our_2d, gk, prev_winner, winner_since, now)
        prev_winner = winner

        we_have = (winner is not None and winner in our_2d and
                   _dist(our_2d[winner], ball) < POSSESS_DIST)
        opp_has = any(_dist(ball, o) < 350 for o in opps)

        # Update possession tracker
        poss.update(we_have, opp_has, now)

        sup_ids, def_ids = _split_field(ball, our_2d, gk, winner,
                                         our_gx, opp_gx, poss.state)

        sup_tgts = _support_targets(ball, our_2d, opps, our_gx, opp_gx,
                                    sup_ids, winner, poss.state)
        def_tgts = _defender_targets(ball, our_2d, opps, our_gx, opp_gx,
                                     def_ids, we_have, poss.state)

        # ── Generate commands ──────────────────────────────────
        for rid, rpos in our_full.items():
            lk = last_kick.get(rid, 0.0)

            if rid == gk:
                c = _goalie(rid, rpos, ball, bvel, bspeed, opps,
                            our_2d, our_gx, opp_gx, is_yellow,
                            lk, now, gk_smooth)
            elif rid == winner:
                cs = committed_sides.get(rid)
                c, cs = _attacker(rid, rpos, ball, bvel, our_2d, opps,
                                  our_gx, opp_gx, gk_opp, is_yellow,
                                  lk, now, poss.state, cs)
                committed_sides[rid] = cs
            elif rid in sup_ids:
                fb = _safe_pos(ball[0], ball[1] + 1500, our_gx, opp_gx)
                tgt = sup_tgts.get(rid, fb)
                c = _support(rid, rpos, ball, tgt, our_gx, opp_gx,
                             is_yellow, poss.state)
                committed_sides.pop(rid, None)  # reset when not attacking
            elif rid in def_ids:
                fb = _safe_pos_def(our_gx, 0, our_gx, opp_gx)
                tgt = def_tgts.get(rid, fb)
                c = _defender(rid, rpos, ball, tgt, our_gx, opp_gx,
                              is_yellow, poss.state)
                committed_sides.pop(rid, None)
            else:
                continue

            if c.kick:
                last_kick[rid] = now
            dispatch_q.put((c, 0.15))

        time.sleep(LOOP_DT)
