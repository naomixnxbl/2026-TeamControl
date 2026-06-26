# PD Controller Design

## Big Picture

The movement controller turns a target into robot commands:

```text
Vision / Game Controller / Robot Receiver
        -> Behaviour Tree
        -> Intent Executor
        -> motion/controller.py  (RobotMotionController)
        -> RobotCommand(vx, vy, w, kick, dribble)
```

The current shipping strategy is **Motion Strategy Option A**:

1. Rotate until the robot faces the target direction.
2. Drive toward the target position.

`general_motion()` is available as Option C (guarded combined movement).
Switch to Option C only after Option A is stable and validated in grSim — see
the Option C section below.

---

## Key Variables

| Name | Meaning |
|---|---|
| `vx` | Forward/backward speed in robot frame, m/s |
| `vy` | Sideways speed in robot frame, m/s |
| `w` | Angular speed, rad/s |
| `current_pos` | Robot pose: `(x, y, theta)` |
| `target_xy` | Target position: `(x, y)` |
| `target_theta` | Target heading angle |
| `deadline` | Absolute time the robot should try to arrive by |

---

## Time-Based Speed

The behaviour tree does not directly choose speed. It chooses a deadline.

Each tick, the controller asks:

> How fast do I need to move from here to arrive by the deadline?

```python
time_remaining = max(deadline - time.monotonic(), 0.001)
speed = min(dist_mm / 1000.0 / time_remaining, MAX_SPEED)
w_limit = min(angle_rad / time_remaining, MAX_W)
```

`strategy.py` exposes named time budgets and a helper to build the deadline:

| Constant | Time budget | Meaning |
|---|---:|---|
| `FAST` | 0.2 s | Move as fast as allowed |
| `NORMAL` | 0.5 s | Normal movement |
| `SLOW` | 0.8 s | Slower, careful approach |

```python
from TeamControl.robot.motion.strategy import get_deadline, NORMAL

deadline = get_deadline(NORMAL)  # time.monotonic() + 0.5
```

---

## Main API

Use one persistent `RobotMotionController` per robot. This matters because the
PD controller remembers the previous tick — creating a new one each tick throws
away the derivative history and disables the D term.

```python
from TeamControl.robot.motion.controller import get_motion_controller
from TeamControl.robot.motion.strategy import get_deadline, NORMAL

mv = get_motion_controller(robot_id, is_yellow)
```

`get_motion_controller` is a factory that returns the same instance for a given
`(robot_id, is_yellow)` pair, so calling it multiple times is safe.

Important functions:

```python
# Check if robot is close enough to the position target
mv.is_close_to_target(current_xy, target_xy, threshold_mm=100.0)

# Check if robot is facing the target heading
mv.is_facing_dir(current_theta, target_theta, threshold_rad=0.1)

# Rotate only. Returns w in rad/s.
mv.rotational_motion(current_theta, target_theta, deadline)

# Drive only. Returns (vx, vy) in robot frame, m/s.
# stay_in_field=True activates dynamic braking near the field boundary.
mv.translational_motion(current_pos, target_xy, deadline, stay_in_field=True)

# Option C: guarded combined movement. Returns (vx, vy, w).
mv.general_motion(current_pos, target_xy, target_theta, deadline)
```

---

## Field-Boundary Enforcement (`stay_in_field`)

Pass `stay_in_field=True` to `translational_motion` whenever the robot should
stay on the field. (Renamed from `field_limit` — same parameter, clearer
name.) This routes through `ball_nav.apply_boundary_braking` (the shared
motion-rule layer used by every controller, not just the PD one), which
applies four stages **after** the accel limiter, all tunable in
`field_config.py`:

### Stage 1 — Decel zone (inside field, within the zone)

When the robot is inside the field but within `VORONOI_BOUNDARY_DECEL_ZONE_MM`
(default 400 mm) of any boundary edge, speed is capped with a **linear
ramp** — full speed at the zone's outer edge, down to
`VORONOI_BOUNDARY_NEAR_SPEED_SCALE` (default 0.05, i.e. 5% of `MAX_SPEED`)
right at the wall:

```text
t = dist_to_boundary / VORONOI_BOUNDARY_DECEL_ZONE_MM
v_max = MAX_SPEED × (NEAR_SPEED_SCALE + t × (1 − NEAR_SPEED_SCALE))
```

The PD controller may compute a higher speed; this cap overrides it. This
is a simple linear ramp, not a physics-derived stopping-distance curve —
`regulate_speed_to_target`'s `sqrt(2 × LINEAR_AMAX × dist)` cap (see below)
is the one place that formula is actually used, for never-overshooting a
*target*, not the field boundary.

### Stage 2 — Out-of-field crawl (outside field)

If the robot is already outside the field, velocity is multiplied by
`VORONOI_OUT_OF_FIELD_SPEED_SCALE` (default 0.1).  This is a safety backstop
so momentum is killed quickly; the navigator also overrides the movement target
to the nearest boundary point to actively return the robot.

### Stage 3 — Hard stop

Regardless of which stage above fired, any velocity component pointing
further into a boundary the robot is already within
`VORONOI_BOUNDARY_HARD_STOP_MM` (default 30 mm) of is zeroed outright — the
final guarantee the robot never drives further out.

### Stage 4 — Goal-post zone

Past either end line, within `GOAL_HALF_WIDTH_MM + ROBOT_RADIUS` of the
center line (i.e. lined up with the goal mouth), the x-component driving
further into the physical goal structure is zeroed — it's a hard obstacle,
not a soft boundary like the rest of the field edge.

---

## Recommended Executor Logic

This is Option A from `motion-strategy.md`.  The helper `option_a_movement()`
in `strategy.py` packages this up and returns a `RobotCommand` directly:

```python
from TeamControl.robot.motion.strategy import option_a_movement, get_deadline, NORMAL

cmd = option_a_movement(mv, current_pos, target_xy, target_theta, is_yellow)
dispatch_q.put((cmd, 0.15))
```

If you need to call the controller directly:

```python
deadline = get_deadline(NORMAL)
if not mv.is_facing_dir(current_pos[2], target_theta):
    w = mv.rotational_motion(current_pos[2], target_theta, deadline)
    dispatch_q.put((RobotCommand(id, 0, 0, w, 0, 0, yellow), 0.15))
else:
    vx, vy = mv.translational_motion(current_pos, target_xy, deadline)
    dispatch_q.put((RobotCommand(id, vx, vy, 0, 0, 0, yellow), 0.15))
```

Why this is the default:

- Simple to debug.
- Turning and driving problems are separated.
- Good enough for Division B reliability.

---

## Option C — Guarded Combined Movement

`general_motion()` is Option C. Use it after Option A is stable in grSim.

```python
from TeamControl.robot.motion.strategy import option_c_movement

cmd = option_c_movement(mv, current_pos, target_xy, target_theta, is_yellow)
dispatch_q.put((cmd, 0.15))
```

Or directly:

```python
vx, vy, w = mv.general_motion(current_pos, target_xy, target_theta, deadline)
```

It does this:

```text
if heading_error > 60 degrees:
    rotate only (same as Option A first step)
else:
    drive and rotate together with scaling
```

Scaling means:

- If the robot is facing the wrong way, reduce `vx/vy`.
- If the robot is far from the target position, reduce `w`.

This helps reduce drift and local spinning, but it is still harder to debug
than Option A because `vx`, `vy`, and `w` all change at the same time.

**When to upgrade from Option A to Option C:**

1. Option A works reliably in grSim (robot reaches targets without oscillation).
2. You observe that the sequential turn-then-drive path is too slow for your
   match scenario.
3. You are ready to retune `BLEND_DIST` in `constants.py` (default 300 mm) —
   increase it if the robot spins too much during combined movement.

---

## Legacy API (Option B — removed)

The old `velocity_to_target()` combined-movement API (Option B) has been
removed. Do not use it. If you find a reference to it in old code, replace it
with Option A (`option_a_movement`) or Option C (`option_c_movement`).

---

## Tuning

There are two levels of gain storage — a global default layer and a per-robot
override layer.  They stack and are fully compatible.

### Layer 1 — Global defaults

`tuning.json` at the project root → loaded into `constants.py` at startup:

```json
{
  "turn_kp": 1.0,
  "turn_kd": 0.1,
  "linear_kp": 0.002,
  "linear_kd": 0.0005
}
```

The constants file also sets:

```python
MIN_V = 0.0    # m/s   — minimum linear command (dead-zone floor)
MIN_W = 0.0    # rad/s — minimum angular command (dead-zone floor)
```

These are the fallback for every robot that has no per-robot calibration.

### Layer 2 — Per-robot overrides

`movement_calibration.json` at the project root, written by the PD Calibration
page or the calibration harness.  Keys are `"yellow/0"`, `"blue/3"`, etc.

```json
{
  "yellow/0": {
    "turn_kp": 1.2,
    "turn_kd": 0.08,
    "linear_kp": 0.0025,
    "linear_kd": 0.0004,
    "speed_scale": 0.95,
    "lateral_drift_per_m": 5.0,
    "stop_overshoot_mm": 20.0,
    "min_v": 0.05,
    "min_w": 0.03
  }
}
```

Hardware-compensation fields (Layer 2 only):

| Field | Meaning |
|---|---|
| `speed_scale` | Actual/commanded speed ratio from calibration. < 1 means robot is slow. |
| `lateral_drift_per_m` | Side drift in mm per metre of forward travel. |
| `stop_overshoot_mm` | How far the robot rolls past the target; aim this much short. |
| `min_v` | Minimum linear command to overcome dead-zone. |
| `min_w` | Minimum angular command to overcome dead-zone. |

When `get_motion_controller(robot_id, is_yellow)` is called, it loads Layer 2
if available, otherwise falls back to Layer 1.  The PD Calibration page shows
a banner indicating which layer is active ("Source: movement_calibration.json"
in orange vs "Source: constants.py" in grey).

### Compatibility with legacy tuning

`tuning.json` also stores `angular_normal_speed`, `angular_fast_speed`, and
related proportional-layer values used by `ball_nav` / `Movement.py`.  These
are **separate from the PD gains** and do not interact with Layers 1–2.

Important ceiling: `MAX_W = max_w_raw * w_clamp_pct = 0.5 * 0.60 = 0.30 rad/s`.
The PD controller's angular output is hard-capped here.  The legacy values
`angular_normal_speed = 0.5` and `angular_fast_speed = 0.6` exceed this cap —
they only apply in the proportional behaviour layer and never reach the PD
controller.  To raise the PD ceiling, edit `max_w_raw` or `w_clamp_pct` in
`tuning.json`; the proportional layer is unaffected.

Plain meaning of PD gains:

- `kp`: how hard the robot pushes toward the target.
- `kd`: how much the robot damps/brakes as the error changes.
- Higher values are not always better; too high causes oscillation.

Option C also uses:

```python
BLEND_DIST = 300.0  # mm — in constants.py
```

Increase `BLEND_DIST` if the robot spins too much during combined movement.

---

## Not Yet Integrated

| Piece | Current status |
|---|---|
| `RobotIntent` dataclass | Not built; only `Intent` enum exists |
| `IntentExecutor` | Missing |
| Behaviour tree outputs intents | Not yet; striker still writes direct velocities |
| Logger in executor | Missing |

Until these are built, treat this document as the movement design target, not
as a fully integrated architecture.

---

## Calibration Logs

PD auto-tune (`PDCalibration.auto_tune_turn` / `auto_tune_linear` in
`pd_calibration.py`) writes a full per-candidate history for every sweep to
`calibration_logs/<team>/<letter>/<timestamp>_<kind>_autotune.json` (see
`robot/motion/calibration_log.py`). `movement_calibration.json` only ever
keeps the latest winning gains, so this is the place to look if you need to
see what a sweep actually tried.

**Not yet done:** the hardware Auto-Calibrate / Speed Sweep flow in
`ui/calibration_page.py` still logs into the flat `"runs"` array inside
`calibration.json` instead of this per-robot folder. Migrating it to the
same `calibration_logs/<team>/<letter>/` convention is a follow-up.

---

## Wheel-Aware Speed/Accel Limits

`MAX_SPEED`/`MAX_W`/`LINEAR_AMAX`/`ANGULAR_AMAX` cap speed/acceleration as
an isotropic circle — the same in every direction. The real robot (and
grSim's own physics model, see `SSL/grSim/config_files/TurtleRabbit.ini`)
is a 4-omniwheel robot with **asymmetric** wheel angles
(60°/135°/225°/300°), so the true achievable envelope is direction-
dependent. `robot/motion/wheel_kinematics.py` adds an opt-in wheel-aware
limiter: enter a robot's wheel angles, wheel/robot radius, and measured
max wheel speed/accel in the PD Calibration page's "Wheel Geometry" card,
and `RobotMotionController` automatically switches that robot from the
isotropic limiter to the true wheel-feasible envelope. Leaving
`max_wheel_speed_mps`/`max_wheel_accel_mps2` at 0 (→ `None`) keeps a robot
on the old isotropic behaviour — this is purely additive, nothing changes
until those two are measured and entered.

**Keeping grSim in sync (manual step):** this repo doesn't launch or
control grSim (`harness/grSim_runner.py` only talks to an already-running
instance over UDP), so there's no live sync. When a robot's wheel geometry
changes, manually copy the same numbers into `TurtleRabbit.ini`
(`wheel*_angle_deg` → `WheelXAngle`, `wheel_radius_mm` → `WheelRadius` in
metres, `robot_radius_mm` → `Radius` in metres) and restart grSim, so the
simulated physics matches what the limiter assumes. The newer
`ssl_simulation_config.proto` (`RobotSpecs`/`SimulatorConfig`) could
automate this via a live config channel, but its sender is commented out
in `network/ssl_sockets.py` and wiring it up is out of scope for now.
