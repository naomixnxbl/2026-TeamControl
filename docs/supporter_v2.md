# Supporter v2 — Behaviour Tree Design

The v2 supporter replaces the static v1 behaviour (move to a hardcoded
position and stay there) with a reactive tree that handles ball chasing,
pass distribution, receiving, and dynamic repositioning.

Source: `src/TeamControl/bt/trees/supporter.py`

---

## Tree topology

```
SupporterRoot (Selector, memory=False)
├── PossessionSequence (Sequence)
│   ├── InPossession
│   └── DistributeSelector (Selector)
│       ├── PassSequence (Sequence)
│       │   ├── FindOpenTeammate
│       │   ├── DribbleTowardTarget
│       │   └── PassToTeammate
│       ├── ShootIfClose
│       └── DribbleToGoal
├── ReceivePassSequence (Sequence)
│   ├── IsPassTarget
│   └── HoldForPass
├── BallPossessionSequence (Sequence)
│   ├── IsClosestToBall
│   └── GoToBall
└── RepositionToSpace
```

The Selector evaluates branches top-to-bottom and takes the first one
that succeeds. This ordering encodes priority:

1. **Have the ball?** → distribute it (pass / shoot / dribble)
2. **Being passed to?** → hold position and face the ball
3. **Closest to ball?** → chase it
4. **None of the above?** → find open space

---

## How passing works

### Selecting a pass target (`FindOpenTeammate`)

For each own robot (excluding the goalie and self), compute a "marking
pressure" score:

```
score(teammate) = min( dist(teammate, opp) for each opponent )
```

The teammate with the **highest** score — i.e. the most space from any
opponent — is selected. If no opponent is on the field, all teammates
score infinity and the first eligible one wins.

If the best score is below `MARKED_THRESHOLD` (0.5 m), every teammate
is considered marked and the pass branch fails. The tree falls through
to `ShootIfClose` or `DribbleToGoal`.

### Dribble-turn before passing (`DribbleTowardTarget`)

After selecting a target, the robot must face the teammate before
kicking. `DribbleTowardTarget` writes `IntentDribble(target_pos=teammate)`
which keeps the dribbler active while the robot rotates. This is
physically correct — the ball stays attached during the turn.

The node returns `RUNNING` while the heading error exceeds
`PASS_ORIENT_TOL` (0.2 rad ≈ 11°). Once aligned, it returns `SUCCESS`
and `PassToTeammate` fires `IntentKick` toward the teammate on the same
tick.

### The `_pass_committed` flag

A naive implementation oscillates: the robot orients toward the
teammate, but on the next tick `InPossession` re-checks whether the
ball is in front of the kicker. Since the robot now faces the teammate
(not the ball), the heading check fails, possession is "lost", and the
robot falls through to chase mode — only to regain possession and try
again.

Fix: when `FindOpenTeammate` succeeds, it sets `_pass_committed = True`
on the shared tree instance. While committed, `InPossession` only checks
distance (skips the heading check), allowing the robot to freely rotate
without losing possession status. The flag resets at the start of every
`tick()` and is only re-set if `FindOpenTeammate` succeeds again. If the
ball physically escapes (distance > `POSSESSION_DIST`), the robot
correctly drops out regardless of the flag.

### Pass signal coordination (`_active_pass_target`)

When `PassToTeammate` fires, it writes `_active_pass_target = target_id`
on the shared `SupporterTree` instance. Since all supporters (robots
2–5) share the same tree object, this acts as a side-channel signal.

On subsequent ticks, when the target robot is evaluated:

- `IsPassTarget` checks `_active_pass_target == my_id` → SUCCESS
- `HoldForPass` writes `IntentMove(current_pos, face_ball)` — the
  receiver stays put and watches the ball arrive

The signal clears when:

- **Timeout**: after `PASS_SIGNAL_TIMEOUT_TICKS` (100) ticks (~1 sec)
- **Receiver transitions**: the receiver does something other than hold
  (e.g. gains possession and enters `PossessionSequence`, or becomes
  closest and chases)

Non-target robots ignore the signal entirely and continue with their
normal tree evaluation.

---

## How ball chasing works

### `IsClosestToBall`

Computes `dist(robot, ball)` for every own robot excluding the goalie.
Returns `SUCCESS` only if this robot has the smallest distance.
Tie-break: lowest `robot_id` wins. This prevents multiple supporters
from chasing the same ball.

### Attacker deference

The attacker tree (`attacker.py`) also chases the ball via `ChaseBall`,
but checks whether it is the closest non-goalie robot first. If a
supporter is closer, the attacker approaches at `CHASE_SLOW_SPEED`
(0.2 m/s) instead of full speed, giving supporters priority to reach
the ball.

---

## How repositioning works (`RepositionToSpace`)

When a supporter is not chasing, not receiving, and doesn't have the
ball, it finds open space on the field using a grid search:

1. Divide the attacking half into a grid at `GRID_STEP` (0.5 m)
   intervals (roughly 100 cells)
2. Score each cell: `min(dist_to_nearest_opponent, dist_to_nearest_own_robot)`
3. Pick the highest-scoring cell (most space from everyone)
4. Tie-break: closer to opponent goal (forward bias)
5. Orient toward the ball (ready for a pass)

The grid bounds are mirrored by `us_positive`: when attacking toward
−x, the grid covers `[−4.0, 1.0] × [−2.5, 2.5]`; when attacking
toward +x, it covers `[−1.0, 4.0] × [−2.5, 2.5]`.

---

## Tuneable constants

| Constant | Value | Purpose |
|---|---|---|
| `POSSESSION_DIST` | 0.122 m | Ball-to-robot distance for "in possession" |
| `POSSESSION_HEADING_TOL` | 0.3 rad | Max heading error to ball for possession |
| `PASS_ORIENT_TOL` | 0.2 rad | Max heading error to target before passing |
| `SHOOT_DIST_THRESHOLD` | 2.0 m | Max distance to goal to allow shooting |
| `MARKED_THRESHOLD` | 0.5 m | Teammate closer than this to an opponent = marked |
| `CHASE_SLOW_SPEED` | 0.2 m/s | Attacker speed when not closest to ball |
| `GRID_STEP` | 0.5 m | Reposition grid resolution |
| `PASS_SIGNAL_TIMEOUT_TICKS` | 100 | Ticks before pass signal expires |

---

## Known limitations

- **Attacker conflict**: the attacker still chases the ball (slowly)
  when a supporter is closer. Coordinator-level "who chases" arbitration
  is a separate task.
- **No possession hysteresis**: `InPossession` uses the same flickering
  `POSSESSION_DIST` as the attacker. A proper fix (separate acquire/lose
  thresholds) applies to both trees.
- **`ball_velocity` is `(0, 0)`**: repositioning and pass-target
  selection don't factor in ball motion because velocity isn't wired yet.
- **Sequential tree sharing**: the `_active_pass_target` and
  `_pass_committed` fields rely on the coordinator ticking supporters
  sequentially on the same tree instance. If ticking were ever
  parallelised, these would need to move to a shared data structure.
