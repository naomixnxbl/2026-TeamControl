# Motion Package Future Work

Date: 2026-05-21

## Current Verdict

This is good enough to hand to the team as a first PD controller and calibration
interface for RoboCup Division B, as long as Option A is treated as the default.

It does not guarantee perfect motion. It gives the team a stable way to ask for
predictable motion and a repeatable way to tune it.

## Recommended Strategy

Ship this order:

1. Use Option A by default: rotate first, then drive.
2. Calibrate PD gains in grSim.
3. Calibrate PD gains on real robots.
4. Add hardware compensation only after PD-only behavior is measured.
5. Let BT use the motion API once Option A passes repeatable tests.
6. Keep Option C as an experiment after Option A is stable.

## What Is Left To Do

| Priority | Task | Why |
|---|---|---|
| P0 | Use a clean world snapshot as calibration pose source | Calibration results are only as good as pose data freshness. |
| P0 | Add stale-pose rejection | Prevents tuning against old vision data. |
| P0 | Run grSim angular and linear tests for robots 0-5 | Confirms the API works before real hardware. |
| P0 | Run real robot angular and linear tests | Confirms the controller handles real response. |
| P1 | Measure min command/deadzone gains | `min_v` and `min_w` exist but need real values. |
| P1 | Add angular scale if needed | `min_w` exists, but angular speed scale is not separate yet. |
| P1 | Save calibration result history | Makes tuning decisions easier to compare. |
| P1 | Add automatic gain sweep | Reduces manual tuning time. |
| P2 | Add derivative filtering | Vision noise can make the D term jump. |
| P2 | Add acceleration limiting | Prevents sudden command changes on real robots. |
| P2 | Clean old calibration UI and `ball_nav` calibration path | Avoids two calibration systems living forever. |
| P2 | Replace old modes with BT mode plus calibration mode | Matches the planned system architecture. |

## Suggested Next Gains

Do not add full PID yet. These already exist or are next:

```json
{
  "min_v": 0.0,
  "min_w": 0.0,
  "angular_scale": 1.0
}
```

Meaning:

| Gain | Meaning |
|---|---|
| `min_v` | Smallest useful linear command that actually moves the robot. |
| `min_w` | Smallest useful angular command that actually rotates the robot. |
| `angular_scale` | Future per-robot correction if the robot turns too fast or too slow. |

Only add `ki` after repeated tests show steady-state error that PD and hardware
compensation cannot remove.

## Score Breakdown

| Category | Score | Improvement Needed |
|---|---:|---|
| Sim movement structure | 8/10 | Needs grSim pass/fail data. |
| Real robot compensation | 6.5/10 | Measure deadzone and add angular scale only if needed. |
| Calibration workflow | 6/10 | Add automatic sweeps and result history. |
| BT integration readiness | 8/10 | Good enough if BT uses Option A. |
| Reliability | 5/10 | Needs world snapshot, freshness, and logging. |
| Maintainability | 7/10 | Good split, but old calibration code still exists elsewhere. |

## Main Recommendation

Finish world-model snapshot integration before making the controller more
complex. A simple controller with clean pose data will tune better than an
advanced controller using stale or inconsistent pose data.

## Suggested Architecture Target

```text
Vision / grSim / real feedback
        |
        v
Clean World Snapshot
        |
        +--> BT
        +--> PD Calibration
        +--> GUI
        |
        v
Motion Controller
        |
        v
Dispatcher / robot sender
```

When BT exists, the old role modes can be reduced to:

| Mode | Purpose |
|---|---|
| `bt` | Main gameplay mode. |
| `calibration` | Robot movement and tuning tests. |
| `vision_only` | Debug input pipeline. |

## When This Is Good Enough For Division B

Call it ready when:

- Option A reaches target within 100 mm in grSim for 10/10 runs.
- Option A reaches target within 150 mm on real robots for 8/10 runs.
- Angular turn finishes within 0.10 rad in grSim for 10/10 runs.
- Angular turn finishes within 0.15 rad on real robots for 8/10 runs.
- Saved gains reload correctly after restarting the GUI.
- Untuned robots safely fall back to default gains.
- Calibration stops if pose data is missing or stale.
