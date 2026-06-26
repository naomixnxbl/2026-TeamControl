# Bounded Voronoi Map Generator

This document explains the Voronoi map work under `src/TeamControl/world/map/`.

## Goal

Generate a closed, bounded Voronoi navigation map for the SSL field:

- Real field size: `9000 x 6000 mm`
- Default navigation inset: `100 mm`
- Default navigation bounds: `(-4400, 4400, -2900, 2900)`
- Default clearance: `ROBOT_RADIUS_MM + SAFE_MARGIN`, currently `100 mm`
- Default clipped bounds: `(-4280, 4280, -2780, 2780)`
- Output: closed Voronoi cells, graph nodes, and safe graph edges

The inset is separate from the field drawing. The PNG shows the full field, but
the navigation map is generated inside the inset rectangle so a robot following
the map does not plan directly on the field boundary.

Voronoi cells are clipped to the clearance-clipped bounds, not the larger inset
navigation bounds. That keeps the blue outer Voronoi box aligned with the safe
yellow perimeter corridor.

In realtime Map Debug, the Voronoi boundary is generated from the latest
SSL-Vision `FieldSize.field_length` and `FieldSize.field_width`. If vision has
not supplied geometry yet, the generator falls back to the constants in
`field_config.py`.

## Shared Defaults

The active defaults are centralized in `src/TeamControl/world/field_config.py`.
Use the `VORONOI_*` constants there when changing planner, render, or navigator
behaviour.

Key map defaults:

- `VORONOI_MIN_CLEARANCE_MM = ROBOT_RADIUS_MM + SAFE_MARGIN`
- `VORONOI_BOUNDARY_INSET_MM`
- `VORONOI_RENDER_DENSITY_PERCENT`
- `VORONOI_RENDER_MAX_DENSITY_NODES`
- `VORONOI_GENERATOR_MAX_DENSITY_NODES`
- `VORONOI_OBSTACLE_COST_WEIGHT`
- `VORONOI_HORIZON_MS`

Live robot planning uses denser defaults than the UI overlay. The render layer
is intentionally lighter so the debug worker can keep up while the map panel is
open.

## Files

- `src/TeamControl/world/map/voronoi_generator.py`
  Generates virtual sites, closed Voronoi cells, graph nodes, and safe edges.
- `src/TeamControl/planner/voronoi_dijkstra.py`
  Temporarily connects robot/target positions into the graph and runs Dijkstra.
- `src/TeamControl/planner/api.py`
  Public facade used by robot behaviours and future skill execution.
- `src/TeamControl/robot/voronoi_navigator.py`
  Uses the planner in `voronoi_test` mode to chase the ball by waypoint.
- `src/TeamControl/process_workers/voronoi_map_runner.py`
  Generates the realtime debug world-map render data in a background process.
- `src/TeamControl/world/map/voronoi_plot.py`
  Saves a Matplotlib PNG showing field markings plus the Voronoi overlay.
- `scripts/render_voronoi_png.py`
  Local command-line tool for creating PNGs in `png/`.
- `tests/world/test_voronoi_generator.py`
  Tests the bounded cells, clearance checks, inset bounds, density grid, and PNG helper.

## PNG Legend

- Green background: field and field margin
- White lines: field markings
- Gray rectangles: goals
- Dashed yellow rectangle: inset navigation bounds
- Solid orange rectangle: clearance-clipped bounds used for cell clipping and safe perimeter edges
- Blue lines: closed Voronoi cell boundaries, clipped to the orange/safe boundary
- Yellow lines: candidate navigation edges that passed clearance
- Red dots: virtual sites
- Purple dots/circles: obstacle centers and inflated obstacle keep-out regions

Blue does not mean unsafe by itself; it means "cell geometry." A line becomes
yellow only when it is part of the navigation graph. The outer blue cell box is
now clipped onto the same clearance boundary as the yellow safe perimeter.

To make the outer range easier to navigate on clear maps, the generator adds
safe yellow perimeter corridors just inside the navigation bounds. The orange
rectangle shows that clearance-clipped perimeter region explicitly.

## Placement Modes

### Density Grid

`density_grid` is the default for PNG generation.

It creates a deterministic grid based on a density percentage:

- `10%`: coarse map, currently `4 x 2 = 8` virtual sites
- `100%`: tight practical map, capped by `--max-density-nodes`

Why cap it? With `100 mm` clearance, a mathematically tight grid would place
sites roughly every `200 mm`. On an `8800 x 5800 mm` inset field, that can mean
hundreds of sites. The current dependency-free Voronoi builder clips every cell
against every other site, so very high site counts can become slow and visually noisy.

Example:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 10
```

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 100
```

Use a higher cap if you want more detail:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 100 --max-density-nodes 400
```

### Explicit Grid

`grid` mode lets you choose X/Y spacing directly.

The site grid is placed across the inset navigation bounds. If the default inset
is `100 mm`, the grid spans:

```text
x: -4400 to 4400
y: -2900 to 2900
```

Example:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --mode grid --grid-spacing-x 2200 --grid-spacing-y 1450
```

For a tighter map, use spacing near twice the requested clearance. For example,
with `100 mm` clearance:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --mode grid --grid-spacing-x 200 --grid-spacing-y 200
```

That is the tight version, but it can produce a lot of sites.

### Random

`random` mode keeps the earlier behavior for comparison.

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --mode random --nodes 16 --seed 3
```

## Clearance Model

The generator keeps only graph edges that satisfy:

```text
edge clearance >= min_clearance_mm
```

Clearance currently checks distance from the edge to every virtual site and keeps
the edge inside the navigation bounds. When obstacles are provided, it also
checks:

```text
distance(edge, obstacle_center) - obstacle_radius >= min_clearance_mm
```

Edges that sit directly on the inset boundary are usually rejected because they
have no room to move farther outward.

The perimeter corridor is different from those literal boundary edges. It is
offset inward by the requested clearance, so it can be drawn as a solid yellow
navigation edge when the boundary area is clear.

This is conservative. It is better for a candidate map to be sparse and safe
than dense and unsafe.

## Recommended Defaults

For visual exploration:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 10
```

To compare against a specific SSL-Vision field size:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 10 --field-length 12000 --field-width 8000
```

For a denser but still practical map:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 60
```

For a tight map:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 100
```

For exact grid control:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --mode grid --grid-spacing-x 500 --grid-spacing-y 500
```

## Obstacles

Virtual sites are kept. Obstacles are added as an extra site layer.

```text
virtual sites  = stable field coverage
obstacle sites = dynamic avoidance pressure
```

The generator accepts either:

- `VoronoiObstacle(pos_mm=(x, y), radius_mm=r, label="...")`
- `PlanningObstacle` objects returned by `WorldMap.get_planning_obstacles()`

Use the direct helper when you already have explicit obstacle data:

```python
from TeamControl.world.map.voronoi_generator import (
    VoronoiObstacle,
    generate_bounded_voronoi_map,
)

voronoi_map = generate_bounded_voronoi_map(
    placement_mode="density_grid",
    density_percent=60,
    obstacles=(
        VoronoiObstacle((0.0, 0.0), radius_mm=300.0, label="obs0"),
    ),
)
```

Use the `WorldMap` helper when planning from live world data:

```python
from TeamControl.world.field_config import VORONOI_HORIZON_MS
from TeamControl.world.map.voronoi_generator import generate_voronoi_map_from_world_map

voronoi_map = generate_voronoi_map_from_world_map(
    world_map,
    now_s=now_s,
    horizon_ms=VORONOI_HORIZON_MS,
    ignore_robots={(True, controlled_robot_id)},
    placement_mode="density_grid",
    density_percent=60,
)
```

This reuses `WorldMap.get_planning_obstacles()`, so obstacle positions are
already age-compensated and radius-inflated by existing world-map logic. The
path safety checks use `WorldMap.is_path_free()`, which accounts for the moving
robot radius and the obstacle safe radius.

For local PNG debugging, pass obstacles as `x,y,radius[,label]`:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 60 --obstacle 0,0,300,center
```

Multiple obstacles are allowed:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 60 --obstacle 0,0,300,A --obstacle 1200,700,240,B
```

You can also generate random obstacles for visual stress testing:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 60 --random-obstacles 5 --obstacle-radius 240 --seed 3
```

Use a radius range:

```powershell
.\.venv\Scripts\python.exe scripts\render_voronoi_png.py --density 60 --random-obstacles 8 --obstacle-radius-min 180 --obstacle-radius-max 420 --seed 10
```

Random obstacles are placed inside the inset navigation bounds. The same `--seed`
recreates the same obstacle layout.

## Realtime World-Map Layer

The Qt map-debug tab can now include a `Voronoi map` layer. Use the UI mode
`voronoi_test` as a separate testing environment. This starts the vision/world
model pipeline and runs one robot on each team through the live Voronoi
planner.

In the Qt app, the selected `our_id` runs on the configured team color and the
selected `enemy_id` runs on the opposite team color. In `main.py --mode
voronoi_test`, shell `0` is used on each team.

Recommended workflow:

1. Start the UI.
2. Select `voronoi_test`.
3. Click `Start`.
4. Open the `Map Debug` tab.
5. Enable the `Voronoi map` checkbox.
6. Move robots in grSim or place robots manually and watch the purple obstacle
   zones, yellow safe edges, and robot routes update.

The layer is generated only while the map-debug stream is enabled, and the
engine already throttles map render data to about `10 Hz`.

The full debug map render is generated by `WorldMapRenderWorker`, not inside
`WorldModel`. The UI sends world-map snapshots to the worker and emits the
latest completed `MapRenderData`. This keeps robot layers, predicted clearance,
ball layers, and Voronoi generation away from the world-model process.

This render worker is for UI/debug map data. It is not the planner API. Robot
behaviours call `PlannerAPI` inside their own process; the planner may generate
its own bounded Voronoi graph during a `plan()` call when the direct path is
blocked and the cached route is no longer valid.

The layer is hidden by default in the checkbox list. Enable `Voronoi map` in the
map panel to inspect it.

`voronoi_test` also publishes the route currently used by each navigator. The
map renderer displays this as a `Planned paths` layer so you can distinguish the
chosen route from the full Voronoi graph.

The engine logs generation cost roughly once per second:

```text
[map] World map worker generated in 8.80 ms (Voronoi 7.10 ms)
```

The realtime overlay currently uses conservative defaults:

- `density_percent=VORONOI_RENDER_DENSITY_PERCENT`
- `max_density_nodes=VORONOI_RENDER_MAX_DENSITY_NODES`
- `horizon_ms=VORONOI_HORIZON_MS`
- `obstacle_cost_weight=VORONOI_OBSTACLE_COST_WEIGHT`

Those defaults are meant for visualization first. Increase density carefully,
because this Voronoi builder clips each cell against all sites.

## Edge Weighting

Unsafe edges are rejected. Safe but narrow edges can also be made more expensive.

When `obstacle_cost_weight > 0`, edge cost becomes:

```text
cost = edge_length * (1 + obstacle_cost_weight * min_clearance / edge_clearance)
```

This means:

- wide safe edges stay close to normal distance cost
- safe edges near robots become more expensive
- Dijkstra path search can prefer wider corridors without losing legal routes

The debug drawing still shows all safe edges. The planner uses `MapEdge.cost`
when running Dijkstra, so cost weighting affects route choice even though the
rendered graph does not visually hide high-cost edges.

## Current Limitations

- Safe edges are not guaranteed to form a single connected graph.
- Some safe edges may be geometrically safe but not useful for route planning.
- Boundary-adjacent edges are conservative and often removed.
- The dependency-free Voronoi implementation is easy to understand but not optimized for thousands of sites.
- Graph connectivity can still fail in crowded layouts; callers should handle
  an empty waypoint result as "no safe route found this tick."

## Configuration Risks

- Lowering `SAFE_MARGIN` or `VORONOI_MIN_CLEARANCE_MM` makes the map less
  conservative. The tests warn if `SAFE_MARGIN` drops below `10 mm`, but that is
  only a warning; robot contact risk depends on real hardware size and control
  accuracy.
- Raising `VORONOI_RENDER_DENSITY_PERCENT` or
  `VORONOI_RENDER_MAX_DENSITY_NODES` can make Map Debug sluggish because the
  worker regenerates cell geometry while the UI is open.
- Raising live planner density can improve route choices, but it increases the
  cost of Dijkstra fallback ticks. Direct-path and cached-route fast paths are
  still cheap.
- If `VORONOI_BOUNDARY_INSET_MM` plus `VORONOI_MIN_CLEARANCE_MM` gets too large
  relative to the field, the clipped bounds can collapse and map generation will
  fail.
- Render defaults and live planner defaults are intentionally separate. Do not
  assume the debug graph density exactly matches the graph used by a robot on a
  blocked planning tick.
