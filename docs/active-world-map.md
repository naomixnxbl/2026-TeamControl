# Active World Map

## Purpose

`WorldMap` sits between SSL Vision and path planning. It turns delayed frame
observations into a lightweight tracked world:

```text
SSL Vision detection frame
    -> Frame
    -> WorldSnapshot
    -> WorldMap
    -> predicted planning obstacles
    -> Voronoi planner and local navigator
```

The map stores observations in world coordinates. Convert to a robot-relative
frame only when a controller needs to issue movement commands.

## Activation And API

The world map is activated by starting the TeamControl backend. Both entry
points create a manager-backed `WorldModel` and start `WMWorker`, which feeds
vision/game-controller updates into `WorldModel.world_map`:

```shell
python ui_main.py
python main.py --mode voronoi_test
```

In code, the live shared model is created through `WorldModelManager`:

```python
from TeamControl.world.model_manager import WorldModelManager

wm_manager = WorldModelManager()
wm_manager.start()
wm = wm_manager.WorldModel()
```

Robot behaviours do not need to construct `WorldMap` directly. They receive the
shared `wm` proxy from the process launcher and use its public methods:

```python
frame = wm.get_latest_frame()
snapshot = wm.snapshot()
planning_obstacles = wm.get_planning_obstacles(
    now_s=now_s,
    horizon_ms=250,
    ignore_robots=((is_yellow, robot_id),),
)
```

For planner integration, pass those frozen obstacle snapshots into
`PlannerInput.obstacles`. See [Voronoi Planner Rules](voronoi-planner-rules.md)
for the planner-side API and [Multiprocessing](Multiprocessing.md) for the
process boundary.

## Vision Timestamps

SSL Vision detection frames provide:

```proto
required double t_capture = 2;
required double t_sent = 3;
```

Both values are seconds. `t_capture` is the important timestamp for tracking
because it describes when the camera observed the field. `t_sent` is useful for
measuring vision processing and network delay.

`Frame` preserves both values. When multiple cameras contribute to one frame,
the combined frame keeps the latest values.

Robot velocity uses differences between SSL `t_capture` values. Observation
freshness uses a separate local receipt timestamp from `time.time()`. Do not
subtract an SSL capture timestamp directly from the local Unix clock: SSL
sources are not required to use the same clock origin.

## Robot Tracking

Each fresh robot observation looks backward at the previous observation for the
same `(isYellow, robot_id)`:

```python
new_obs.update_vel_from(old_obs)
```

Velocity is estimated in world coordinates:

```python
vx = (new_x - old_x) / dt_s
vy = (new_y - old_y) / dt_s
```

For short-horizon planning, predicted position is:

```python
predicted_x = x + vx * horizon_s
predicted_y = y + vy * horizon_s
```

## Planning Obstacles

Use:

```python
obstacles = wm.get_planning_obstacles(
    now_s=time.time(),
    horizon_ms=200,
    ignore_robots={(True, 0)},
)
```

Each immutable `PlanningObstacle` contains:

```text
robot_id
isYellow
pos_mm
radius_mm
vel_mmps
observation_age_ms
prediction_horizon_ms
```

The planning horizon includes observation delay:

```python
prediction_horizon_ms = observation_age_ms + requested_horizon_ms
```

The effective radius expands with speed:

```python
radius_mm = safe_radius_mm + speed_mmps * prediction_horizon_s
```

The planner should usually consume this frozen obstacle view through
`PlannerInput.obstacles`. That keeps one control tick deterministic even though
the underlying `WorldModel` is shared through a multiprocessing manager.

## Ball Tracking

`Frame` preserves raw ball candidates from every camera. For compatibility with
older callers, `frame.ball` still exposes the first raw observation. New
tracking code should use all candidates through `WorldSnapshot.ball_candidates`.

`WorldMap` validates and ranks candidates before accepting one:

1. Reject candidates below the confidence threshold.
2. Reject candidates outside the received field dimensions.
3. Predict the previous ball position at the new capture timestamp.
4. Reject candidates that are too far from the prediction.
5. Select the highest-confidence remaining candidate.
6. Use distance from the predicted position as a confidence tie-breaker.
7. Preserve the last valid position when the ball briefly disappears.

The current tuning values are:

```python
BALL_MIN_CONFIDENCE = 0.1
BALL_BASE_TOLERANCE_MM = 150.0
BALL_TOLERANCE_RATE_MMPS = 7000.0
```

Useful map state:

```text
ball
ball_vel_mmps
ball_visible
ball_last_seen_s
last_rejected_ball_pos_mm
last_ball_rejection_reason
possible_ball_left_field_pos_mm
ball_left_field_pos_mm
```

`possible_ball_left_field_pos_mm` comes from an out-of-bounds vision
observation. `ball_left_field_pos_mm` comes from a confirmed game-controller
event. Keep those meanings separate.

Do not sort purely by distance. A false detection slightly closer to the
prediction should not automatically beat a much stronger observation.

## Qt Debug Renderer

The Qt command center has a `World Map` tab and matching Dashboard controls for
inspecting tracked state. Their checkboxes are generated from serializable
`RenderLayer` objects, so layers can be hidden independently. The built-in
layers are:

```text
Robots
Velocity vectors
Predicted clearance
Ball
```

Velocity arrows show `250 ms` of travel. Predicted-clearance circles are hidden
by default and include both the requested horizon and current observation age.

Render producers provide their own layers without importing Qt:

```python
voronoi = RenderLayer(
    "Voronoi edges",
    polylines=(RenderPolyline(points_mm=edge, color="#ffffff"),),
)
render_data = world_map.get_render_data(extra_layers=(voronoi,))
```

The canvases automatically add a checkbox for the new layer.

The canvas starts with local field defaults, then switches to the latest
`SSL_GeometryFieldSize` received from vision. A changed geometry updates the
home field, world-map field, and calibration field without restarting the UI.
Debug render frames are requested at `10 Hz` while either the Dashboard or
`World Map` tab is visible. The Dashboard hides duplicate `Robots` and `Ball`
debug layers by default because the live field already draws those objects.

## Dashboard Field Actions

The dashboard field owns the click-driven manual action layer:

- `Click on Field` in ball placement mode sends a grSim ball replacement and
  leaves an orange `X` at the requested point. Use this marker to manually align
  the requested placement with the ball reported by vision. Once vision sees
  the ball within the configured tolerance for `0.5 s`, the marker is treated as
  confirmed and removed.
- Left-clicking a robot selects that robot as the target for dashboard field
  actions such as `Go to Ball`, `Go to Ball & Kick`, and `Go to Point`.
- Right-click field actions reuse the Hardware Test action loop, but the main
  window routes them through the same dashboard guard before any command starts.

Dashboard placement and action commands are intentionally disabled unless the
engine is running, the mode is not `6v6`, and `Send Commands to grSim` is on.
This keeps the convenience layer out of competition mode and prevents field
clicks from becoming an accidental real-robot command path. The Hardware Test
tab remains the deliberate path for direct hardware testing.

Potential bug to watch for: while visible, the orange placement `X` marks the
requested grSim replacement point, not a verified vision measurement. If the X
and the rendered ball disagree after vision catches up, check grSim/vision
geometry and network ports before debugging the dashboard drawing.

## Voronoi Integration

The current Voronoi integration is split across two paths:

- `PlannerAPI` runs inside the robot/skill process and owns per-robot waypoint
  state.
- `WorldMapRenderWorker` runs as a background process for UI/debug render data.
- `WorldMap.get_planning_obstacles()` provides predicted obstacle snapshots for
  both paths.

The global planner proposes a route. The local navigator remains responsible
for braking and replanning when moving obstacles invalidate that route.

## Field-Edge Targets

Movement code sanitizes world-frame targets before driving. A target outside
the playable field is offset inward to an inset box, rather than slowing every
command merely because the robot is close to a wall. Callers that cannot use an
offset target can explicitly reject it with:

```python
sanitize_field_target(target, reject_outside=True)
```

Distance-to-target deceleration and obstacle avoidance remain separate safety
behaviors.
