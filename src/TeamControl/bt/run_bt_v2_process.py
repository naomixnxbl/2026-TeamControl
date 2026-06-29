"""V2 behaviour-tree process runner.

Mirrors ``behaviour_tree/run_bt_process.py`` but drives the TurtleRabbitBT
Coordinator instead of the legacy ``MainTree``. Spawn this from
``SSL/grSim/sandbox.py`` (or any other harness) using
``multiprocessing.Process``.

Pipeline each tick:

    WorldModel  →  build_snapshot_from_world_model
                →  Coordinator.tick(snapshot, robot_ids)
                →  dispatch_coordinator_output → dispatcher_q
"""
from __future__ import annotations

import time
from multiprocessing import Event, Queue

from TeamControl.bt.adapter import (
    DribbleLimitTracker,
    build_snapshot_from_world_model,
    dispatch_coordinator_output,
)
from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.snapshot import GamePhase
from TeamControl.bt.coordinator import Coordinator
from TeamControl.bt.trees.attacker import AttackerTree
from TeamControl.bt.trees.defender import DefenderTree
from TeamControl.bt.trees.goalie import GoalieTree
from TeamControl.bt.trees.supporter import SupporterTree
from TeamControl.network.robot_command import RobotCommand
from TeamControl.utils.yaml_config import Config as _YamlConfig
from TeamControl.world.model import WorldModel

_HALT_PHASES = (GamePhase.HALTED, GamePhase.HALF_TIME)


def _send_stop_commands(
    robot_ids: list[int], is_yellow: bool, dispatcher_q: Queue
) -> None:
    """Send zero-velocity commands for every robot to override stale commands."""
    for rid in robot_ids:
        cmd = RobotCommand(robot_id=rid, vx=0.0, vy=0.0, w=0.0, isYellow=is_yellow)
        if not dispatcher_q.full():
            dispatcher_q.put([cmd, 0.1])


def _intent_debug(intent) -> tuple[str | None, tuple[float, float] | None]:
    """Return the small intent summary consumed by the Qt dashboard."""
    if intent is None:
        return None, None
    intent_type = type(intent).__name__
    if intent_type.startswith("Intent"):
        intent_type = intent_type[len("Intent"):]
    target = getattr(intent, "target_pos", None)
    return intent_type.upper(), target


def _pack_bt_state(
    *,
    tick_count: int,
    is_yellow: bool,
    snapshot,
    coordinator: Coordinator,
    robot_ids: list[int],
) -> dict:
    """Build the lightweight BT state packet displayed in the dashboard."""
    robot_map = {robot.robot_id: robot for robot in snapshot.own_robots}
    robots = []
    for rid in robot_ids:
        bb = coordinator.blackboards.get(rid)
        robot = robot_map.get(rid)
        intent_type, intent_target = _intent_debug(
            None if bb is None else bb.current_intent
        )
        robots.append(
            {
                "id": rid,
                "role": "UNKNOWN" if bb is None else bb.current_role.value,
                "pos": None if robot is None else robot.position,
                "ori": None if robot is None else robot.orientation,
                "intent_type": intent_type,
                "intent_target": intent_target,
                "intent_source": None if bb is None else bb.intent_source,
            }
        )
    return {
        "tick": tick_count,
        "phase": snapshot.referee_state.game_phase.value,
        "ball": snapshot.ball_position,
        "is_yellow": is_yellow,
        "robots": robots,
    }

# Robot ids 0..5 — matches Coordinator.ROLE_ASSIGNMENT.
DEFAULT_ROBOT_IDS: list[int] = [0, 1, 2, 3, 4, 5]

# Target tick period in seconds (100 Hz).
TICK_PERIOD: float = 0.01


class _BtLogger:
    """Per-robot file logger for the BT process.

    Creates log/<YYYY-MM-DD>/bt/coordinator.log and log/.../bt/robot_<id>.log.
    Files are line-buffered so they flush after each write.
    """

    def __init__(self, robot_ids: list[int]) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        log_dir = Path("log") / today / "bt"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._robot_files: dict[int, "TextIO"] = {}
        for rid in robot_ids:
            self._robot_files[rid] = open(log_dir / f"robot_{rid}.log", "a", buffering=1, encoding="utf-8")
        self._proc_file = open(log_dir / "coordinator.log", "a", buffering=1, encoding="utf-8")

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def proc(self, msg: str) -> None:
        self._proc_file.write(f"{self._ts()} {msg}\n")

    def robot(self, robot_id: int, msg: str) -> None:
        f = self._robot_files.get(robot_id)
        if f:
            f.write(f"{self._ts()} {msg}\n")

    def close(self) -> None:
        for f in self._robot_files.values():
            try:
                f.close()
            except Exception:
                pass
        try:
            self._proc_file.close()
        except Exception:
            pass


def _build_coordinator(
    us_positive: bool,
    role_assignment: dict[int, RoleType] | None = None,
    heuristic_role_swap: bool = False,
    movement_safety: dict[str, bool | float] | None = None,
) -> Coordinator:
    c = Coordinator(
        trees={
            RoleType.GOALIE: GoalieTree(us_positive=us_positive),
            RoleType.DEFENDER: DefenderTree(us_positive=us_positive),
            RoleType.SUPPORTER: SupporterTree(us_positive=us_positive),
            RoleType.ATTACKER: AttackerTree(us_positive=us_positive),
        },
        us_positive=us_positive,
        role_assignment=role_assignment,
        heuristic_role_swap=heuristic_role_swap,
        movement_safety=movement_safety,
    )
    print(
        f"[BT] coordinator built: us_positive={us_positive} "
        f"opp_goal={c._opp_goal} attack_sign={c._attack_sign} "
        f"heuristic_role_swap={heuristic_role_swap} "
        f"movement_safety={c.movement_safety}",
        flush=True,
    )
    return c


def _fmt_intent(intent) -> str:
    """Compact one-liner for an intent, e.g. 'Move→(0.25,0.00)' or 'Kick→(-2.0,0.0)'."""
    if intent is None:
        return "None"
    kind = type(intent).__name__.replace("Intent", "")
    if hasattr(intent, "target_pos") and intent.target_pos is not None:
        x, y = intent.target_pos
        return f"{kind}→({x:.2f},{y:.2f})"
    return kind


def run_bt_v2_process(
    is_running: Event,
    wm: WorldModel,
    dispatcher_q: Queue,
    is_yellow: bool | None = None,
    robot_ids: list[int] | None = None,
    role_assignment: dict | None = None,
    heuristic_role_swap: bool = False,
    movement_safety: dict[str, bool | float] | None = None,
    tick_period: float = TICK_PERIOD,
    config_file: str = "ipconfig.yaml",
    bt_state_q: "Queue | None" = None,
    verbose: bool = False,
) -> None:
    """Tick the v2 (TurtleRabbitBT) coordinator in a child process.

    Args:
        is_running: shared Event — clear to stop the loop.
        wm: shared WorldModel proxy.
        dispatcher_q: queue consumed by the dispatcher; items are
            ``[RobotCommand, run_time_seconds]``.
        is_yellow: team perspective for this BT instance. ``None`` falls
            back to config. For 6v6 pass ``True``/``False`` explicitly.
        robot_ids: which robot ids to tick. Defaults to 0..5.
        role_assignment: per-robot RoleType override dict. Defaults to
            the module-level ROLE_ASSIGNMENT in coordinator.py.
        heuristic_role_swap: if true, dynamically assign non-goalie roles
            during RUNNING. If false, keep static role_assignment behaviour.
        movement_safety: optional movement guard rails from sim_6v6.yaml.
        tick_period: seconds to sleep between ticks.
        config_file: path to yaml config (relative to utils/).
        bt_state_q: optional GUI queue for compact BT inspector state.
    """
    if robot_ids is None:
        robot_ids = DEFAULT_ROBOT_IDS

    _cfg = _YamlConfig(config_file)
    if is_yellow is None:
        is_yellow = bool(_cfg.us_yellow)
    # Derive us_positive from config: our team's side is read directly from
    # yaml; the opposing team is on the opposite side.
    cfg_us_positive = bool(_cfg.us_positive)
    cfg_us_yellow   = bool(_cfg.us_yellow)
    _us_positive = cfg_us_positive if (is_yellow == cfg_us_yellow) else not cfg_us_positive
    coordinator = _build_coordinator(
        us_positive=_us_positive,
        role_assignment=role_assignment,
        heuristic_role_swap=heuristic_role_swap,
        movement_safety=movement_safety,
    )
    dribble_tracker = DribbleLimitTracker()
    print(f"[BT] started — yellow={is_yellow}, us_positive={_us_positive}, robot_ids={robot_ids}")

    tag = "[BT-YELLOW]" if is_yellow else "[BT-BLUE]"
    last_phase = None
    tick_count = 0
    _last_printed: dict = {}  # rid -> (intent_type, target_1dp)

    try:
        while is_running.is_set():
            snapshot = build_snapshot_from_world_model(
                wm,
                is_yellow=is_yellow,
                active_robot_ids=robot_ids,
            )
            if snapshot is None:
                time.sleep(tick_period)
                continue

            phase = snapshot.referee_state.game_phase

            # Print whenever the game phase changes
            if phase != last_phase:
                if last_phase is None:
                    print(f"{tag} initial phase: {phase}", flush=True)
                else:
                    print(f"{tag} phase changed: {last_phase} -> {phase}", flush=True)
                if verbose:
                    bx, by = snapshot.ball_position
                    print(f"{tag} t={tick_count} phase -> {phase.value} ball=({bx:.2f},{by:.2f})", flush=True)
                _last_printed.clear()
                last_phase = phase

            coordinator.tick(snapshot, robot_ids)

            tick_count += 1

            if verbose:
                bx, by = snapshot.ball_position
                robot_map = {r.robot_id: r for r in snapshot.own_robots}
                changed = []
                for rid in robot_ids:
                    bb = coordinator.blackboards.get(rid)
                    r = robot_map.get(rid)
                    intent = bb.current_intent if bb else None
                    key = (
                        type(intent).__name__,
                        tuple(round(v, 1) for v in intent.target_pos)
                        if intent and hasattr(intent, "target_pos") and intent.target_pos else None,
                    )
                    if _last_printed.get(rid) != key:
                        _last_printed[rid] = key
                        pos = f"({r.position[0]:.2f},{r.position[1]:.2f})" if r else "?"
                        role = bb.current_role.value if bb else "?"
                        changed.append(f"  R{rid}[{role}] {pos} {_fmt_intent(intent)}")
                if changed:
                    print(f"{tag} t={tick_count} {phase.value} ball=({bx:.2f},{by:.2f})", flush=True)
                    print("\n".join(changed), flush=True)

            if bt_state_q is not None and tick_count % 30 == 0:
                try:
                    bt_state_q.put_nowait(
                        _pack_bt_state(
                            tick_count=tick_count,
                            is_yellow=is_yellow,
                            snapshot=snapshot,
                            coordinator=coordinator,
                            robot_ids=robot_ids,
                        )
                    )
                except Exception:
                    pass

            # Print intents every 100 ticks (~1 sec)
            if tick_count % 100 == 0:
                bx, by = snapshot.ball_position
                print(f"{tag} tick={tick_count} phase={phase.value} ball=({bx:.2f},{by:.2f})", flush=True)
                robot_map = {r.robot_id: r for r in snapshot.own_robots}
                for rid in robot_ids:
                    bb = coordinator.blackboards.get(rid)
                    if bb and bb.current_intent:
                        r = robot_map.get(rid)
                        pos_str = f"pos=({r.position[0]:.2f},{r.position[1]:.2f}) facing={r.orientation:.2f}rad" if r else "pos=N/A"
                        print(f"  robot {rid} ({bb.current_role.value}) {pos_str}: {bb.current_intent}", flush=True)

            if phase in _HALT_PHASES:
                _send_stop_commands(robot_ids, is_yellow, dispatcher_q)
            else:
                dispatch_coordinator_output(
                    coordinator,
                    robot_ids,
                    snapshot,
                    is_yellow,
                    dispatcher_q,
                    dribble_tracker=dribble_tracker,
                )
            time.sleep(tick_period)
    except KeyboardInterrupt:
        print("[BT] KeyboardInterrupt — exiting", flush=True)
