# Attack Handoff — What We Do When We Hold the Ball

**Scope:** our in-possession strategy (GegenPressing yellow side), read straight
from the code. GegenPress/defence is considered good; **today's work is the
attack.** This is the map of what happens the moment we win/hold the ball.

**One-line summary:** when we secure the ball we flip the whole team forward,
the carrier **shoots if there's a good spot**, else plays the **most direct
forward pass** to the most-advanced open teammate, else **dribbles forward** —
minimal passes, keep it advancing, keep possession on their half, finish when
open.

---

## 1. Team shape the instant we win the ball

`Coordinator._apply_counter_attack_roles` — [coordinator.py](../src/TeamControl/bt/coordinator.py)

- Triggered by `_we_have_secure_possession` (our nearest robot is ≥ `secure_margin`
  = 0.30 m closer to the ball than any opponent). This **breaks the press
  instantly** (no exit debounce) — see `_update_gegenpress_state`.
- Roles become: **carrier = ATTACKER**, every other field robot = **SUPPORTER**.
- Supporters `RepositionToSpace` → spread into open space toward the opponent
  goal (not clustering on the carrier). So we commit numbers forward for the
  counter.
- Only active when `gegenpress.enabled` (our yellow side). Config:
  [sim_gegenpress.yaml](../src/TeamControl/utils/sim_gegenpress.yaml).

## 2. The ball-carrier's decision (the ATTACKER tree)

`AttackerTree` PossessionAction selector — [attacker.py](../src/TeamControl/bt/trees/attacker.py) `_build_tree`.

Runs only after **`HasBallControl`** (ball within `POSSESSION_DIST` = 0.11 m and
in front of the kicker). With `counter_attack: true` (our config) the priority is:

1. **ShootSequence** — shoot if it's a good spot.
2. **CounterReleaseSequence** — minimal forward pass to advance.
3. **HoldPossession** — dribble toward goal (fallback when no good pass).

> With `counter_attack: false` (default/other teams) branch 2 is the older
> situational `PassSequence` (pass only when blocked/under pressure) instead.

### 2a. Shoot — `ShootSequence`
`HasSettledPossession` → `HasClearShot` → `ShootAtGoal`.
- **Settle:** need `SHOT_SETTLE_TICKS` = **5** continuous ticks of control (shoots
  quickly after receiving — was 30, tightened to 5).
- **Clear shot:** within `SHOOT_DIST_THRESHOLD` = **2.0 m** of goal, facing the
  aim point within `SHOT_HEADING_TOL` = **0.25 rad (~14°)**, and the corridor
  (`SHOT_CORRIDOR_RADIUS` 0.20 m) clear.
- **Aim:** `_best_goal_target` samples across the goal mouth
  (`GOAL_MOUTH_HALF_WIDTH` 0.45 m) and picks the point **farthest from the
  keeper** — we shoot at the open side, not always centre.

### 2b. Forward release — `CounterReleaseSequence`
`ShouldCounterRelease` (just checks `counter_attack` flag) → `FindForwardOutlet`
→ `DribbleTowardPassTarget` → `PassToOpenTeammate`.
- `_find_forward_outlet` picks the **most-advanced open teammate** that is:
  - at least `counter_min_advance_frac` = **10%** of the field *ahead of the
    carrier* toward goal (genuine forward pass, no square/back balls),
  - not tightly marked (`counter_outlet_marked_frac` 0.05 clearance),
  - on a clear lane (line-of-sight).
- `DribbleTowardPassTarget` turns the carrier to face the outlet (`pass_orient_tol`),
  then `PassToOpenTeammate` fires `IntentPass`.
- **No forward outlet → the sequence fails → falls to HoldPossession (dribble).**

### 2c. Dribble — `HoldPossession`
- `IntentDribble` toward goal (the default when no shot and no forward pass).
- Motion-level: `MotionExecutor` adds a **left-right nudge** while dribbling to
  retain the ball on the dribbler ([adapter.py](../src/TeamControl/bt/adapter.py),
  `DRIBBLE_NUDGE_*`).
- **SSL 1 m dribble limit** enforced by `DribbleLimitTracker` → forces a kick if
  we carry too far.

## 3. Off-ball (supporters) while we attack

- **Break forward + spread:** SUPPORTER `RepositionToSpace` maximizes distance
  from teammates *and* opponents, trending toward the opponent goal.
- **Reception in sync:** when the carrier passes, `Coordinator._apply_pass_receive_sync`
  latches the receiver onto the reception point (faces the ball, steps onto it
  as it arrives) so passer/receiver aren't out of sync.
- **Anti-cluster:** `_apply_teammate_spacing` pushes apart any teammates whose
  targets bunch (goalie + ball-chaser exempt).

## 4. Where to tune the attack

| Want to change | File / constant |
|---|---|
| Shoot sooner / later | `SHOT_SETTLE_TICKS`, `SHOT_HEADING_TOL`, `SHOOT_DIST_THRESHOLD` in [attacker.py](../src/TeamControl/bt/trees/attacker.py) (or `behavior_tree.attacker` in [bt_tuning.yaml](../src/TeamControl/utils/bt_tuning.yaml)) |
| How "forward" a pass must be | `COUNTER_MIN_ADVANCE_FRAC` (0.10), `COUNTER_OUTLET_MARKED_FRAC` (0.05) |
| Open-goal aim width | `GOAL_MOUTH_HALF_WIDTH` (0.45) |
| Dribble nudge feel | `DRIBBLE_NUDGE_*` in [adapter.py](../src/TeamControl/bt/adapter.py) |
| Commit-forward trigger | `gegenpress.secure_margin` in [sim_gegenpress.yaml](../src/TeamControl/utils/sim_gegenpress.yaml) |

## 5. Known gaps / candidates for today

- **`ball_velocity` is hardcoded `(0,0)`** in [adapter.py](../src/TeamControl/bt/adapter.py)
  → receivers can't lead a moving ball; interception/first-touch can't anticipate.
  Biggest single lever for attack quality.
- **Pass target = teammate's current position** (no lead) → moving receivers can be
  passed behind.
- **Only the carrier attacks; supporters are positional** — no coordinated give-and-go
  / overlapping runs yet.
- **Shot aim ignores the keeper's momentum** and doesn't chip/curve.
- Set-piece attack (our free kicks/corners) is handled separately in the
  Coordinator, not by this tree.

---
*Run it: start grSim → `python ui_main.py` → mode `gegenpress` → Start. Yellow =
us (this attack), blue = competition opponent.*
