# Motion Test Plan

Date: 2026-05-21

This file lists the tests needed before the motion controller should be trusted
by the BT team.

## Test Matrix

Run each movement test with these four combinations:

| Case | `use_pd` | `use_hardware` | Purpose |
|---|---|---|---|
| Raw | `False` | `False` | Baseline deadline movement. |
| PD only | `True` | `False` | Tune PD without hardware compensation. |
| Hardware only | `False` | `True` | Check speed/drift/overshoot/deadzone compensation. |
| Combined | `True` | `True` | Final expected real movement mode. |

## grSim Tests

### 1. Import And Construction

Pass if this runs without error:

```powershell
python -c "from TeamControl.robot.motion import get_motion_controller; print(get_motion_controller(0, True).get_gains())"
```

Expected:

- Default gains load.
- No JSON file is required.
- Robot ID and team color are accepted.

### 2. Angular Turn Test

Setup:

- Start grSim.
- Start calibration backend.
- Select robot `0`.
- Use yellow unless testing blue.

Run:

- Turn 90 degrees.
- Repeat 10 times.
- Start from different headings if possible.

Pass target:

| Metric | Target |
|---|---|
| Final heading error | `< 0.10 rad` |
| Success rate | `10/10` |
| No runaway spin | required |
| Robot translation during turn | small / acceptable |

Tune rule:

| Symptom | Change |
|---|---|
| Turns too slowly | increase `turn_kp` |
| Overshoots angle | increase `turn_kd` or reduce `turn_kp` |
| Shakes near angle | reduce `turn_kd` or add derivative filtering later |

### 3. Linear Forward Test

Run:

- Move forward 1000 mm.
- Repeat 10 times.

Pass target:

| Metric | Target |
|---|---|
| Final position error | `< 100 mm` |
| Final heading error | `< 0.10 rad` |
| Success rate | `10/10` |
| No large sideways drift | required |

Tune rule:

| Symptom | Change |
|---|---|
| Too slow / stops short | increase `linear_kp` |
| Overshoots | increase `linear_kd` or reduce `linear_kp` |
| Curves sideways | check transform and pose first, then hardware drift |

### 4. Option A Movement Test

Run through `strategy.py`:

```python
cmd = turn_then_go(motion, current_pos, target_xy, target_theta, is_yellow)
```

Pass target:

- Robot turns first.
- Robot drives only after facing target.
- `w = 0` during translation.
- `vx = vy = 0` during rotation.
- Reaches target within 100 mm.

### 5. Option C Smoke Test

Only after Option A passes.

Pass target:

- If heading error is over 60 degrees, robot turns first.
- If heading error is under 60 degrees, robot can translate and rotate together.
- No local spinning.
- No strong drift away from target.

## Real Robot Tests

### 1. Safety Checklist

Before sending movement commands:

- Robot is on a clear field.
- Emergency stop is available.
- Battery is charged.
- Wheel directions are correct.
- Vision pose is visible and stable.
- Team color and robot ID are correct.
- Command sender is targeting the right robot.

### 2. Pose Freshness Test

Pass target:

| Metric | Target |
|---|---|
| Pose update rate | stable enough for control loop |
| Pose age | ideally `< 200 ms` |
| Missing pose handling | calibration stops safely |

This should become automatic once world snapshot integration is finished.

### 3. Real Angular Turn Test

Run:

- 45 degree turn.
- 90 degree turn.
- -90 degree turn.
- Repeat each 5 times.

Pass target:

| Metric | Target |
|---|---|
| Final heading error | `< 0.15 rad` |
| Success rate | `8/10` or better |
| No continuous spin | required |

### 4. Real Linear Forward Test

Run:

- 500 mm forward.
- 1000 mm forward.
- 1000 mm backward if safe.

Pass target:

| Metric | Target |
|---|---|
| Final position error | `< 150 mm` |
| Success rate | `8/10` or better |
| Drift | measured and recorded |

### 5. Speed Scale Calibration

Run:

- Command a straight movement.
- Measure actual distance and time from vision.
- Compute:

```text
speed_scale = actual_speed / commanded_speed
```

Interpretation:

| Result | Meaning |
|---|---|
| `< 1.0` | Robot is slower than commanded. |
| `> 1.0` | Robot is faster than commanded. |
| `1.0` | No speed correction needed. |

### 6. Deadzone Calibration

Run small commands while the robot is still.

For linear deadzone:

```text
vx = 0.01, 0.02, 0.03, ...
```

Save the smallest value that reliably moves the robot:

```text
min_v = smallest reliable linear command
```

For angular deadzone:

```text
w = 0.02, 0.04, 0.06, ...
```

Save the smallest value that reliably rotates the robot:

```text
min_w = smallest reliable angular command
```

Defaults are zero, so deadzone compensation is off until measured.

### 7. Lateral Drift Calibration

Run:

- Drive straight along a known line.
- Measure sideways error at the end.
- Compute:

```text
lateral_drift_per_m = sideways_error_mm / distance_traveled_m
```

Run both directions if possible and average the result.

### 8. Stop Overshoot Calibration

Run:

- Command a fixed straight movement.
- Measure how far past the target the robot stops.

Save:

```text
stop_overshoot_mm = average overshoot distance
```

Use only positive overshoot. If the robot stops short, tune PD first.

### 9. Save And Reload Test

Run:

1. Apply tuned gains.
2. Save last result.
3. Stop GUI.
4. Restart GUI.
5. Select same robot and team.
6. Check gains loaded from JSON.

Pass target:

- Saved values reload correctly.
- Different robot IDs keep separate gains.
- Yellow and blue teams keep separate gains.
- Clear tuned gains returns to defaults.

## Suggested Acceptance Criteria

The motion package is ready for BT handoff when:

- grSim Option A passes all tests.
- At least one real robot passes angular and linear tests.
- Saved gains survive restart.
- Untuned robot defaults are safe.
- Calibration can run without changing BT code.

The motion package is ready for match testing when:

- Robots 0-5 each have saved gains or verified defaults.
- Pose freshness is enforced.
- Real robots pass 8/10 angular and linear tests.
- Option A works from multiple field positions.
- Logs show no repeated missing-pose or stale-pose failures.
