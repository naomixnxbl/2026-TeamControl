# Motion Package Summary

Date: 2026-05-21

## What Was Implemented Today

The new motion package gives the team a cleaner path than the legacy
`Movement.py` file. It separates motion control, PD math, gain storage,
hardware compensation, strategy wrappers, and calibration tests.

## Files

| File | Purpose |
|---|---|
| `pd.py` | Reusable PD controller for scalar turning and 2D translation errors. |
| `controller.py` | Main robot motion controller. Owns PD instances and generates robot commands. |
| `settings.py` | Loads, saves, clears, and applies per-robot gains from JSON. |
| `hardware.py` | Real-life compensation helpers for speed scale, drift, overshoot, and deadzone. |
| `strategy.py` | Team-facing movement strategies: Option A and Option C. |
| `pd_calibration.py` | Calibration test runner for angular turn and linear forward tests. |
| `__init__.py` | Public imports for other modules. |

## Current Public Interface

```python
from TeamControl.robot.motion import get_motion_controller, turn_then_go

motion = get_motion_controller(robot_id, is_yellow)
cmd = turn_then_go(motion, current_pos, target_xy, target_theta, is_yellow)
dispatch_q.put((cmd, 0.15))
```

For direct controller use:

```python
motion.rotational_motion(current_theta, target_theta, deadline, use_pd=True)
motion.translational_motion(
    current_pos,
    target_xy,
    deadline,
    use_pd=True,
    use_hardware=True,
)
motion.general_motion(
    current_pos,
    target_xy,
    target_theta,
    deadline,
    use_pd=True,
    use_hardware=True,
)
```

## Gains

The saved gains now include both PD gains and simple hardware compensation:

```json
{
  "turn_kp": 1.0,
  "turn_kd": 0.1,
  "linear_kp": 0.002,
  "linear_kd": 0.0005,
  "speed_scale": 1.0,
  "lateral_drift_per_m": 0.0,
  "stop_overshoot_mm": 0.0,
  "min_v": 0.0,
  "min_w": 0.0
}
```

Untuned robots use defaults. Tuned robots load their own saved gains by team
color and robot ID.

## Calibration Modes

The calibration runner can test four useful combinations:

| `use_pd` | `use_hardware` | Meaning |
|---|---|---|
| `True` | `False` | PD only |
| `False` | `True` | deadline movement plus hardware compensation |
| `True` | `True` | PD plus hardware compensation |
| `False` | `False` | raw deadline movement |

This is useful because real-life compensation should be proven separately from
PD before combining both.

## Evaluation

| Area | Score | Notes |
|---|---:|---|
| Team-facing API | 8/10 | Simple enough for BT users. Option A is clear and safe. |
| PD implementation | 7/10 | Good reusable PD. Missing filtering and anti-noise handling. |
| Calibration storage | 7/10 | Per-robot JSON works. Path should eventually be anchored to repo/config. |
| Hardware compensation | 6.5/10 | Speed/drift/overshoot/deadzone are simple and optional. |
| grSim readiness | 7/10 | Should be good enough for repeatable first tests. Needs full simulation run. |
| Real robot readiness | 5/10 | Needs pose freshness checks, deadzone testing, and real robot validation. |
| Division B practicality | 7/10 | Conservative Option A fits Division B consistency-first strategy. |

Overall score: 7/10 for a first team-usable motion controller.

## Main Strength

The biggest strength is separation of responsibility:

```text
BT decides target
strategy.py chooses movement style
controller.py computes velocity
pd.py computes correction
hardware.py compensates real robot behavior
settings.py saves per-robot gains
pd_calibration.py tests repeatability
```

That is a good interface for a student team because each person can work on
their part without rewriting the whole robot stack.

## Main Limits

- Calibration still depends on whatever pose source it is given.
- The controller does not yet reject stale vision/world-model data.
- The hardware compensation is linear and simple.
- No automated gain search exists yet.
- Min command/deadzone compensation exists, but must be measured on real robots.
- Angular motion has PD and `min_w`, but no separate angular speed scale yet.
- Option C should stay experimental until Option A is reliable.
