# Motion Strategy - TeamControl Division B

## Motion Systems

The PD controller (`motion/controller.py`, `RobotMotionController`) was
removed at one point because it didn't work on the server-deployed/
competition side, but has since been **restored** and is back in active use
— see `docs/pd-controller-design.md`. The three systems in play today:

| System | Module | Used by | Best for |
|---|---|---|---|
| **A — Proportional ramp** | `ball_nav.move_toward()` | striker, goalie, navigator, team, voronoi_navigator (bare integrator) | Reactive ball-chasing where the target moves every frame |
| **C — PD controller** | `motion/controller.py` (`RobotMotionController.translational_motion`) | voronoi_game_navigator (match/team play), pd_test mode (`voronoi_pd_test_navigator.py`), PD Calibration page | Production game navigation and hardware-calibrated movement |
| **B — PD with zone caps** | `Movement.py / RobotMovement` | behaviour tree, sandbox | Legacy/frozen — do not add new callers |

### When to use each

**System A (`ball_nav.move_toward`)** — use for direct reactive tracking.
The target (ball, enemy, open space) moves every frame, so a derivative
term would wrongly brake on target motion rather than robot motion. The
proportional ramp handles this well.

**System C (`RobotMotionController.translational_motion`)** — use for the
production game navigator and anywhere per-robot hardware calibration
(`movement_calibration.json` gains, wheel kinematics, overshoot
compensation) matters. `stay_in_field=True` (renamed from `field_limit`)
applies the same shared boundary rules as System A:
`ball_nav.clamp_for_role()` (penalty-box/goal clamping) and
`ball_nav.apply_boundary_braking()` (decel zone, hard stop, goal-post zone,
out-of-field crawl) — both systems go through this one shared rule layer in
`ball_nav.py` so field-boundary behaviour can't drift between them.

**System B (`Movement.py`)** — do not add new callers. The `RobotMovement`
class is the historical middle ground; the zone caps are useful for the
behaviour tree's structured decision flow. It has its own embedded copy of
a PD-style controller (kept as-is, separate from the `motion/` package).
`calculateBallVelocity()` in this file has speed levels that are 10x too
low (0.02–0.10 m/s instead of 0.2–1.0 m/s) — do not use it for real robot
speeds.

---

## Angular Velocity Ceiling

`MAX_W = MAX_W_RAW * W_CLAMP_PCT = 0.5 * 0.60 = 0.30 rad/s`

This is the angular-velocity ceiling used directly by `ball_nav`-based
navigation (e.g. the `clamp(ang_ball * TURN_GAIN, -MAX_W, MAX_W)` pattern in
`voronoi_navigator.py`/`voronoi_game_navigator.py`). The legacy tuning
values `angular_normal_speed = 0.5` and `angular_fast_speed = 0.6` (in
`tuning.json`) are used only by System B's proportional layer — they
exceed `MAX_W` and are intentionally separate. To raise the ceiling,
increase `max_w_raw` or `w_clamp_pct` in `tuning.json`.

---

## Student Summary

- Ball-chasing game logic → `ball_nav.move_toward()`
- Voronoi-planner-driven navigation (test or match mode) → `ball_nav.move_toward()` +
  `ball_nav.apply_boundary_braking()`
- Behaviour tree (legacy) → `Movement.py (RobotMovement)` as-is, no new code there
