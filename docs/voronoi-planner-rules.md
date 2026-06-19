# Voronoi Planner Rules

This document describes the Dijkstra-based planner that sits on top of the
bounded Voronoi map generator in `src/TeamControl/world/map/`.

The implementation is separate from the legacy
`src/TeamControl/voronoi_planner/` package:

```text
src/TeamControl/planner/
src/TeamControl/planner/voronoi_dijkstra.py
src/TeamControl/robot/voronoi_navigator.py
```

## Shared Tunables

Active Voronoi and navigator behaviour tunables live in
`src/TeamControl/world/field_config.py`. Prefer changing the `VORONOI_*`
constants there instead of adding local literals in planner, renderer, or robot
modules.

Important groups:

- map/planner defaults: `VORONOI_HORIZON_MS`,
  `VORONOI_DENSITY_PERCENT`, `VORONOI_MAX_DENSITY_NODES`,
  `VORONOI_BOUNDARY_INSET_MM`, `VORONOI_OBSTACLE_COST_WEIGHT`
- endpoint and steal approach: `VORONOI_ENDPOINT_REACH_MM`
- start-inside-obstacle escape: `VORONOI_ESCAPE_MARGIN_MM`,
  `VORONOI_MIN_ESCAPE_STEP_MM`
- navigator behaviour: `VORONOI_WAYPOINT_REACHED_MM`,
  `VORONOI_TARGET_STOP_MM`, `VORONOI_FIELD_TARGET_MARGIN_MM`,
  `VORONOI_POSSESSION_*`, `VORONOI_STEAL_FRONT_*`, and precision speed/ramp
  constants

The UI/debug Voronoi overlay intentionally has separate render defaults:
`VORONOI_RENDER_DENSITY_PERCENT` and `VORONOI_RENDER_MAX_DENSITY_NODES`. These
are lower than live planner defaults so the map-debug worker stays responsive.

## Current Entry Point

Use the public facade in `src/TeamControl/planner/` for robot behaviours and
skill execution:

```python
from TeamControl.planner import PlannerAPI, PlannerInput

planner = PlannerAPI()
planner_output = planner.plan(
    PlannerInput(
        robot_id=robot_id,
        is_yellow=is_yellow,
        current_pose=current_pose,
        target_pose=target_pose,
        obstacles=planning_obstacles,
        robot_reached_current_waypoint=reached_current_waypoint,
        now_s=now_s,
    )
)
```

`PlannerAPI` is a normal in-process object. It is not a multiprocessing worker
or shared service. Keep one planner instance in the robot/skill process that is
using it so its per-robot waypoint cache stays local and predictable.

For how the live `WorldMap` is started and which `wm` methods expose planning
data, see [Active World Map](active-world-map.md#activation-and-api).

## Inputs

The planner receives:

- current robot position
- target point
- either explicit planning obstacles or a `WorldMap`-compatible object
- optional previous planner state
- optional controlled robot identity for ignored obstacles

The low-level planner builds or receives a bounded Voronoi map that already
includes:

- virtual grid sites
- predicted robot obstacles from `WorldMap.get_planning_obstacles()`
- safe graph nodes
- safe graph edges
- obstacle-aware edge costs

## Rule 0: Edge Weighting Near Obstacles

Each edge cost should increase when obstacles are close to that edge.

The generator and planner weight safe edges by obstacle proximity. Unsafe edges
are still removed first; weighting only ranks the remaining choices.

Base weighting:

```text
cost = edge_length * (1 + obstacle_cost_weight * min_clearance / edge_clearance)
```

Additional nearby-obstacle risk is cumulative, so two obstacles near the same
corridor make that corridor more expensive than one obstacle:

```text
cost = edge_length * (1 + obstacle_cost_weight * (clearance_risk + obstacle_risk_sum))
```

Where each nearby obstacle contributes:

```text
obstacle_penalty = weight * min_clearance / obstacle_edge_clearance
```

Notes:

- Unsafe edges are still rejected before planning.
- Weighting should only rank safe choices.
- Wider corridors should naturally win over narrow corridors.

## Rule 1: Sanitise Target Into Playable Area

The margin is applied once in `VoronoiWaypointManager.update()` (the primary
enforcement point) **before** the `is_path_free` check, so it applies to every
planning decision — free-path returns, cached-route returns, and Dijkstra runs.
`VoronoiDijkstraPlanner.plan()` also applies it internally (idempotent when the
input is already margined).

```text
effective_x = clamp(raw_x, FIELD_X_MIN + m, FIELD_X_MAX - m)
effective_y = clamp(raw_y, FIELD_Y_MIN + m, FIELD_Y_MAX - m)
where m = VORONOI_FIELD_TARGET_MARGIN_MM  (see field_config.py)
```

Examples with m = 150 mm:

```text
(5000, 1000)  -> (4350, 1000)
(1000, 4000)  -> (1000, 2850)
(4400, 200)   -> (4350, 200)   # already close to boundary
(1000, 1000)  -> (1000, 1000)  # well inside — unchanged
```

`VORONOI_FIELD_TARGET_MARGIN_MM` is tunable in `field_config.py`.  The same
constant also gates the intermediate-node validity check in Rule 3.

> **Bug fix (applied):** Before this was corrected, the margin was only applied
> inside Dijkstra.  When the direct path was clear (`is_path_free = True`),
> `active_target_pose` was the exact field boundary with no inset, and the robot
> drove all the way to the edge.  The fix moves enforcement to
> `waypoint_manager.py` so every path return is margined.

## Rule 2: Always Check Direct Path

Before planning through the graph, the planner checks direct path safety.

Use the existing world-map path check:

```python
world_map.is_path_free(start_pos, target_pos, ...)
```

This keeps the planner aligned with the existing obstacle model in
`src/TeamControl/world/map/`.

If the direct path is free:

```text
active_target_pose = target_pos
waypoints = []
```

No Dijkstra search is needed.

In the current `PlannerAPI` flow, `VoronoiWaypointManager.update()` also returns
this decision as:

```python
planner_output.is_path_free
```

If `is_path_free` is `True`, `active_target_pose` is the sanitised target
(clamped to field inset by `VORONOI_FIELD_TARGET_MARGIN_MM`) and no replan is
performed. The planner also clears any previous planned
waypoints in this case. If it is `False`, the manager checks whether an
existing waypoint path can still be used before running a fresh Dijkstra plan.

## Rule 2a: Escape When Already Inside Clearance

If the controlled robot starts inside another obstacle's inflated clearance,
normal path checks can reject every segment that begins at the robot. Without a
special case, the planner cannot connect the start node to the graph, returns no
waypoints, and the navigator stops.

The low-level planner now handles that invalid starting state before normal
direct-path and Dijkstra planning. It asks the path map for planning obstacles,
finds obstacles whose clearance radius already contains the start point, and
returns one short waypoint pointing outward from those obstacles. Once the robot
has moved out of the containing clearance, later ticks return to the usual
direct-path, cached-route, or Dijkstra behavior.

The escape step is intentionally conservative and tunable through
`VORONOI_ESCAPE_MARGIN_MM` and `VORONOI_MIN_ESCAPE_STEP_MM`. If these are too
small, the robot may remain inside clearance for multiple ticks and repeatedly
request escape waypoints. If they are too large, the robot can take a jerky
first step away from the ball or desired route.

## Rule 2b: Endpoint Inside Clearance

The final target may be inside another robot's safety clearance, especially when
the target is the ball and the ball is possessed. Treating that as a hard
blocked target makes the chaser stop just when it should press.

Before direct-path and Dijkstra decisions, `VoronoiWaypointManager` resolves the
endpoint against nearby obstacles:

1. Clamp the requested target to the field.
2. Check whether the target is inside an obstacle envelope using
   `VORONOI_ENDPOINT_REACH_MM` rather than the moving robot's full body
   clearance.
3. If contained, offset the endpoint along the obstacle-to-target direction to
   the close-reach circle.
4. Mark the output with:
   - `endpoint_was_adjusted=True`
   - `endpoint_precision_mode=True`

`endpoint_precision_mode` is a control hint, not a claim that the whole robot
body has normal clearance. The navigator uses it to keep approaching slowly with
precision speed and a tighter stop distance. This is what lets a robot keep
pressing a possessed ball instead of giving up.

Use `PlannerAPI.check_target_clearance()` when a caller needs to classify a
target before planning. It reports both:

- `in_safety_clearance`: target lies inside normal safety clearance
- `in_reach_clearance`: target lies inside the tighter close-reach envelope

### Endpoint Edge Cases To Watch

- If `VORONOI_ENDPOINT_REACH_MM` is too small, the robot may press too deeply
  into another robot and cause contact.
- If it is too large, the robot may stop too far from the ball and fail to
  steal.
- If an adjusted endpoint is clamped by the field boundary and still lies in a
  clearance zone, precision mode remains enabled. The robot should creep in,
  but route generation may still return no waypoints.
- Tuple obstacles and `PlanningObstacle` objects can represent already-inflated
  radii. Real `Obstacle` objects expose both physical radius and safe radius.
  The clearance API separates safety clearance from close-reach clearance, but
  tests should use the obstacle type that matches the scenario being modeled.

## Rule 2c: Reuse Existing Path When Valid

If the direct path is not free, the planner may reuse a previous planned path
instead of rebuilding from scratch.

The cached path is stored as a queue of remaining waypoints. When the controller
reports `robot_reached_current_waypoint`, the manager pops the first waypoint
from that queue before deciding the next active target.

Every tick then checks the direct path to the final goal:

- If the direct path is free, clear all queued waypoints and return the final
  target directly.
- If the direct path is blocked and the next queued waypoint is still reachable,
  keep following that queued waypoint.
- If the direct path becomes blocked again after a direct-free tick, the queue
  will be empty, so the planner generates a fresh route.

The cached queue is valid only if all of these are true:

1. The new target is still within the dead-zone radius of the last target.
2. There are remaining queued waypoints for the similar target.
3. The direct path from the current position to the next waypoint is still free.

If any check fails, the cached path is invalid and the planner must generate a
new path.

Suggested state:

```python
@dataclass
class PlannerState:
    last_target_mm: tuple[float, float] | None
    waypoints_mm: tuple[tuple[float, float], ...]
    generated_at_s: float
```

Suggested configuration:

```python
target_dead_zone_mm = 150.0
```

## New Path Plan

When direct path and cached path both fail:

1. Generate or refresh the obstacle-aware Voronoi map.
2. Build the obstacle-aware navigation graph.
3. Append the final target point if it is not already reached.
4. Cache the target and generated waypoints in planner state.

## Expected Public API

Low-level graph-search class:

```python
class VoronoiDijkstraPlanner:
    def plan(
        self,
        world_map,
        start_pos_mm: tuple[float, float],
        target_pos_mm: tuple[float, float],
        *,
        now_s: float | None = None,
        ignore_robots: set[tuple[bool, int]] | None = None,
        previous_state: PlannerState | None = None,
        stay_in_field: bool = True,
    ) -> PlanResult:
        ...
```

`stay_in_field=True` (default): applies Rule 1 target clamping and Rule 3
intermediate-node validation.  Goal-zone crossing checks are always active
regardless of this flag.  Pass `stay_in_field=False` only when the target is
intentionally outside the field (e.g. tracking a ball rolling out of bounds)
and the caller handles field enforcement separately.

`VoronoiDijkstraPlanner` is the low-level graph search. Robot behavior and the
future Skill Intent Executor should normally call `PlannerAPI` instead:

```python
planner_output = planner.plan(planner_input)
```

The planner API owns the per-robot route cache in a local dict keyed by
`(is_yellow, robot_id)`. `WorldModel` remains the source of world snapshots and
planning obstacle snapshots.

## Waypoint Manager Adapter

The Skill Intent Executor can use the task-PDF shaped adapter:

```python
from TeamControl.planner import PlannerAPI, PlannerInput
from TeamControl.world.field_config import VORONOI_HORIZON_MS

planner = PlannerAPI()
obstacles = wm.get_planning_obstacles(
    now_s=now_s,
    horizon_ms=VORONOI_HORIZON_MS,
    ignore_robots=((is_yellow, robot_id),),
)

planner_output = planner.plan(
    PlannerInput(
        robot_id=robot_id,
        is_yellow=is_yellow,
        current_pose=current_pose,
        target_pose=target_pose,
        obstacles=obstacles,
        clearance_mm=200,
        robot_reached_current_waypoint=nav_output.robot_reached_target,
    )
)
```

`planner_output.active_target_pose` is the target the movement layer
(`ball_nav.move_toward()`) should track. If there is an active waypoint, it
is returned first. If the direct path is free, the field-clamped target is
returned.

`WorldModel` should provide world snapshots, obstacle snapshots, and render
data. The planner API owns route state and waypoint decisions. This keeps the
world layer from turning into a behavior/planning service.

When using the live shared `WorldModel`, callers commonly snapshot planning
obstacles first:

```python
planning_obstacles = wm.get_planning_obstacles(
    now_s=now_s,
    horizon_ms=VORONOI_HORIZON_MS,
    ignore_robots=((is_yellow, robot_id),),
)
```

Passing explicit obstacles makes the planner tick deterministic from the
caller's point of view. Passing `world_map=wm` is also supported, but that keeps
path checks and obstacle reads behind the world-map proxy for that call.

## Rule 3: Path Validity — Reject Corrupt Intermediate Nodes

After Dijkstra returns a path, every **intermediate** Voronoi graph node
(everything between start and target) is validated.  The target itself is not
checked here because the caller controls it via Rule 1.

A path is discarded (empty waypoints returned) if any intermediate node:

1. **Is outside the field** by more than `VORONOI_FIELD_TARGET_MARGIN_MM`.
   The Voronoi graph is built inside the inset boundary, so an out-of-bounds
   node indicates a graph defect.

   ```text
   valid_x: FIELD_X_MIN - m  ..  FIELD_X_MAX + m
   valid_y: FIELD_Y_MIN - m  ..  FIELD_Y_MAX + m
   ```

2. **Is inside a physical goal box**.  The goal structure has walls on three
   sides — no robot can traverse through it.  A node is in the goal zone when:

   ```text
   |y| <= GOAL_HALF_WIDTH_MM  AND  (x > FIELD_X_MAX OR x < FIELD_X_MIN)
   ```

   `GOAL_HALF_WIDTH_MM = 500 mm` (defined in `field_config.py`).

Discarding the path rather than clamping individual nodes preserves safety: a
path with one corrupt node may have more corruption elsewhere, and clamping it
silently could produce a physically impossible route.

## Ball-Steal Clearance Exception

> **Note:** these rules were removed from `voronoi_navigator.py` when it was
> simplified to a bare planner integrator, and are now implemented in the
> production game navigator, `voronoi_game_navigator.py`
> (`_steal_ignore_keys` / `_robot_is_in_front_of_possessor`).  See
> [voronoi-navigator-stripped.md](voronoi-navigator-stripped.md) for the
> full list of what moved there.

The game navigator keeps clearance rules enabled while following the ball.
The only exception is a narrow ball-steal case.

An obstacle can be ignored at the ball target only when it is the specific robot
that possesses the ball. Possession is defined as:

```text
distance(robot, ball) < VORONOI_POSSESSION_DIST_MM
abs(angle_to_ball_in_robot_frame) <= VORONOI_POSSESSION_ANGLE_RAD
ball is in front of the robot
```

The chasing robot must also be in front of the possessor before the possessor
obstacle is ignored. This prevents the planner from globally disabling
clearance whenever the ball happens to sit inside another robot's radius.

## Planned Path Debug Layer

`voronoi_test` publishes each robot's current planner route to the UI engine.
The background `WorldMapRenderWorker` renders those routes as a `Planned paths`
layer on the Map Debug canvas.

The path layer contains:

- the robot's current position
- the active waypoint list, if a Voronoi route is being followed

This layer is separate from the yellow Voronoi graph edges. It shows the actual
route the robot process is currently using.

When `planner_output.is_path_free` is `True`, the robot is using the direct
free path, so the planned-path debug layer receives an empty point list for that
robot. This clears any stale reroute polyline instead of drawing a direct-target
line as a planned route.

For display only, planned-path polylines are clipped to the current field
rectangle before rendering. This prevents off-field robot observations or stale
route points from drawing large floating blue/yellow shapes outside the field.

## voronoi\_test Mode — Integrator Behaviour

`voronoi_navigator.py` in `voronoi_test` mode is a **bare integrator** for
testing the planner in isolation.  It intentionally contains no game logic.

Each tick:

1. Refresh world-model cache.
2. If ball is not visible (including when outside field bounds — see Field
   Enforcement below), send stop command.
3. Call `PlannerAPI.plan()` with no steal-ignore keys and zero clearance.
   The waypoint manager applies Rule 1 (target margin) before returning
   `active_target_pose`.
4. Drive toward `active_target_pose` using `ball_nav.move_toward(rel_target, CHASE_SPEED)`,
   then `ball_nav.apply_boundary_braking(current_pos, vx, vy)`.
   The latter applies the dynamic-braking cap (see Field Enforcement section
   below) — ported from the old PD controller's `field_limit=True` option
   when the PD controller was removed for this competition.
5. Face the ball with a proportional angular controller (`ang_ball * TURN_GAIN`).
6. **Field override**: if the robot is outside the field, `movement_target` is
   overridden to the nearest boundary clamp point (takes priority over planner output).
7. **Ball stop**: stop translating once within `VORONOI_TARGET_OFFSET_MM` of
   the ball.  The robot still rotates to face the ball.
8. **Face-target stop**: stop translating when within `FACE_TARGET_DIST_MM` and
   angle error > `FACE_TARGET_ANGLE_RAD` (dribble alignment).

Stripped behaviours (possession, steal, precision approach, exponential smoothing,
penalty-box guard) are documented in
[voronoi-navigator-stripped.md](voronoi-navigator-stripped.md) for the future
game navigator.

## Field Enforcement — Layers

All active field-enforcement mechanisms, ordered from earliest to latest in the
pipeline:

| Layer | What it does | Where |
|---|---|---|
| **Ball filter** | Out-of-field ball → `ball_visible = False` → navigator stops | `world_map.py:_is_ball_in_field` (falls back to `FIELD_X/Y_MIN/MAX` when field geometry is not received from vision) |
| **Obstacle filter** | Out-of-field robots excluded from planning obstacles | `world_map.py:get_planning_obstacles` |
| **Target margin** | Target clamped to `FIELD_*_MIN/MAX ± VORONOI_FIELD_TARGET_MARGIN_MM` | `waypoint_manager.py` (all cases) + `voronoi_dijkstra.py` (Dijkstra run) |
| **Node validation** | Dijkstra intermediate nodes outside field or in goal zone → path discarded | `voronoi_dijkstra.py` Rule 3 |
| **Segment check** | Any path segment crossing the physical goal mouth → path discarded | `voronoi_dijkstra.py` (always active) |
| **Navigator override** | Robot outside field → movement target overridden to nearest boundary point | `voronoi_navigator.py`, `voronoi_game_navigator.py` |
| **Dynamic braking** | Within `VORONOI_BOUNDARY_DECEL_ZONE_MM` of boundary → speed capped, ramping from `VORONOI_BOUNDARY_NEAR_SPEED_SCALE` at the wall to full speed at the zone edge | `ball_nav.py:apply_boundary_braking()` |
| **Out-of-field scale** | Robot outside field → velocity × `VORONOI_OUT_OF_FIELD_SPEED_SCALE` (0.1) | `ball_nav.py:apply_boundary_braking()` |
| **Hard stop** | Within `VORONOI_BOUNDARY_HARD_STOP_MM` (30 mm) of any edge → the velocity component pointing further out is zeroed outright, regardless of which stage above fired | `ball_nav.py:apply_boundary_braking()` |
| **Goal-post zone** | Past an end line, within `GOAL_HALF_WIDTH_MM + ROBOT_RADIUS` of the centre line → x-component driving into the goal structure is zeroed | `ball_nav.py:apply_boundary_braking()` |
| **Ball-out-of-bounds clearance** | Ball left the field (`wm.last_ball_rejection_reason == "out_of_bounds"`) → robot drives to a point ≥ `OUT_OF_BOUNDS_CLEARANCE_MM` (500 mm) from both the ball's exit point and the boundary line it crossed, instead of stopping in place | `ball_nav.py:compute_out_of_bounds_clearance()`, wired in by `voronoi_game_navigator.py` (not the bare `voronoi_navigator.py` integrator) |
