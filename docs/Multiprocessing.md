# Multiprocessing

## Shared World Model

`WorldModel` is exposed through `WorldModelManager`, a multiprocessing manager.
Processes should receive the manager-backed `wm` object and use its public
methods instead of copying internal state.

Use accessor methods:

```python
frame = wm.get_latest_frame()
snapshot = wm.snapshot()
obstacles = wm.get_planning_obstacles(now_s=now_s, horizon_ms=250)
version = wm.get_version()
```

Avoid reaching into internals such as `wm.frame_list.latest` from another
process. Manager proxies only keep the shared model coherent when callers go
through the methods registered on the model.

## Process Shape

The normal backend starts separate processes for vision, game-controller state,
world-model updates, dispatch, robot receive, and selected robot behaviours.
The Qt engine also starts `WorldMapRenderWorker` when it needs debug map render
data.

The planner is different: `PlannerAPI` is not a multiprocessing worker. Robot
or skill processes create a local `PlannerAPI` instance and call
`planner.plan(PlannerInput(...))` inside their control loop. The planner stores
its waypoint cache in that local process, keyed by `(is_yellow, robot_id)`.

## Versioning

`WorldModel` maintains an integer version counter. Vision frames update the
internal world map every frame and bump the public version on the model's
configured `update_interval`. Game-controller, robot-receiver, field, and
manual update paths bump the version when they change the shared model.
Consumers can poll `wm.get_version()` to decide whether they need to refresh
local cached data for the current tick.

## Planning Data Flow

For robot planning, prefer this pattern:

```python
planning_obstacles = wm.get_planning_obstacles(
    now_s=now_s,
    horizon_ms=250,
    ignore_robots=((is_yellow, robot_id),),
)

planner_output = planner.plan(
    PlannerInput(
        robot_id=robot_id,
        is_yellow=is_yellow,
        current_pose=current_pose,
        target_pose=target_pose,
        obstacles=planning_obstacles,
        now_s=now_s,
    )
)
```

This keeps the expensive/shared world access at the edge of the tick and lets
the planner work from a frozen obstacle snapshot. Passing `world_map=wm` to
`PlannerInput` is supported when the caller wants live `WorldMap` path checks
inside the planner call, but the planner still remains local to the calling
process.

## Debug Map Rendering

`WorldMapRenderWorker` is a background process for the UI's Map Debug data. It
can generate robot layers, predicted-clearance circles, planned-path polylines,
and the optional full Voronoi graph layer. This worker should not be treated as
the robot planner service.

## Common Issues

If a robot lookup returns a placeholder such as `0` instead of a robot object,
the robot probably was not observed in the latest usable frame. Common causes
are missing camera coverage, the robot not being on the field, or a dropped
vision frame.

If a real robot drives toward a strange point, check SSL-Vision ball detection
and colour calibration. A false orange detection can produce a fake ball
position, and any ball-following behaviour will then chase the wrong target.
