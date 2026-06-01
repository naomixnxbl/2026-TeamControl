# Codex Context - 2026 TeamControl

Working notes for Codex on this repository. This is a cleaned-up copy of the repo onboarding context with the details that are most useful during implementation.

## What this repo is
- RoboCup SSL team-control software for WSU TurtleRabbits, 2026 season.
- Python codebase for SSL-Vision and Game Controller ingestion, world-model construction, and robot/grSim command output.
- Includes a PySide6 operator UI plus headless launch paths for running behavior modes.

## Entry points
- `main.py` - headless launcher for the multiprocessing pipeline and selected robot mode.
- `ui_main.py` - starts the PySide6 operator UI via `TeamControl.ui.main_window.MainWindow`.
- `run.bat` - Windows convenience launcher.

## Package layout
Source lives under `src/TeamControl/`.

- `SSL/` - SSL-Vision, Game Controller, and grSim protocol/helpers.
- `network/` - UDP sockets, senders/receivers, command packet assembly, generated protobufs in `proto2/`.
- `cache/` - in-memory caches for ball, robot, team, game state, ticks, and onboard-ball observations.
- `robot/` - motion primitives and behavior entry points such as goalie, striker, navigator, coop, team, and ball navigation.
- `Formation/` - formations and strategic positioning helpers.
- `dispatcher/` - converts high-level robot commands into network output.
- `process_workers/` - multiprocessing workers for vision, GC FSM, world model, and robot receive loops.
- `onboard_vision/` - support for per-robot onboard camera observations.
- `world/` - world-model storage, frame history, and geometry/game-state helpers.
- `ui/` - PySide6 dashboard, calibration, network, robot, and test panels.
- `bt/` - behavior tree contracts, skills, trees, and coordinator logic.
- `harness/` - runner and smoke-test scaffolding.
- `voronoi_planner/` - path-planning helpers.

## Runtime shape
- The main process builds queues, a shared `WorldModelManager`, and a set of worker processes.
- `WorldModelManager` is a `multiprocessing.managers.BaseManager` wrapper around `WorldModel`, so state is shared through the manager rather than copied.
- The world model keeps a frame history and a version counter; consumers should use its accessors rather than reaching into internals.
- `BaseWorker` in `src/TeamControl/process_workers/worker.py` provides the common worker loop, error handling, and shutdown pattern.
- The dispatcher sends commands to both real robots and grSim depending on configuration, and rate-limits per robot.

## Configuration files
- `calibration.json` at the repo root contains drive calibration.
- `tuning.json` at the repo root contains runtime tuning values.
- `src/TeamControl/utils/ipconfig.yaml` contains robot IPs, grSim endpoints, and SSL network settings.
- `Config` in `src/TeamControl/utils/yaml_config.py` loads `ipconfig.yaml` relative to the module path, not the current working directory.

## Important operational notes
- Multiprocess workers do not share normal Python state. If a process needs state from another process, use queues, shared manager objects, or an explicit IPC path.
- After editing drive calibration programmatically, call `ball_nav._reload_calibration()` so live motion code picks up the new values.
- `src/TeamControl/network/proto2/` is generated protobuf code. Regenerate from the `.proto` sources; do not hand-edit the generated files.
- The UI and some runtime components depend on PySide6, multiprocessing freeze support, and Windows-friendly startup behavior.
- `main.py` is interactive and waits for `exit` on stdin to stop the process tree.

## Network defaults and expectations
- Default game-controller multicast is `224.5.23.1:10003`.
- Default SSL-Vision multicast is `224.5.23.2:10006`.
- Default grSim vision is also commonly pointed at `224.5.23.2:10020` in the docs, but the actual runtime config comes from `ipconfig.yaml`.
- Default grSim command port in the checked-in `ipconfig.yaml` is `20011`.
- `ipconfig.yaml` also controls team color (`us_yellow`) and field side (`us_positive`).

## Behavior-tree work
- The `bt/` package is test-driven and has explicit contracts for intents, snapshots, blackboards, and role assignment.
- Tests expect coordinator logic to operate on intent objects, not `RobotCommand`.
- If you change coordinator or tree dispatch, review the tests under `tests/bt/` first.

## Practical edit rules
- Prefer `rg` / `rg --files` for search and file discovery.
- Avoid editing generated protobufs directly.
- Be careful with shared config and calibration files because they are read at runtime and can affect both UI and headless runs.
- If a change touches multiprocessing, verify the worker start/shutdown path as well as the code you are directly editing.
- If a change touches networking, confirm the port and address assumptions in `ipconfig.yaml` and `docs/SSL-NetworkPorts.md`.

## Docs worth checking
- `CONTRIBUTING.md`
- `docs/getting-started.md`
- `docs/HowToWriteCode.md`
- `docs/Multiprocessing.md`
- `docs/SSL-NetworkPorts.md`
- `tests/README.md`
- `src/TeamControl/network/proto2/README.md`

## Notes on repo docs
- Some docs contain typos or stale wording; prefer the actual source tree and runtime config over README spelling or path errors.
- The repository currently has a few duplicated or loosely maintained onboarding notes, so this file is the preferred working summary for Codex.
