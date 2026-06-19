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
from datetime import datetime
from multiprocessing import Event, Queue
from pathlib import Path

from TeamControl.bt.adapter import (
    VoronoiRouter,
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
            self._robot_files[rid] = open(log_dir / f"robot_{rid}.log", "a", buffering=1)
        self._proc_file = open(log_dir / "coordinator.log", "a", buffering=1)

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


def _build_coordinator(us_positive: bool) -> Coordinator:
    return Coordinator(
        trees={
            RoleType.GOALIE: GoalieTree(us_positive=us_positive),
            RoleType.DEFENDER: DefenderTree(us_positive=us_positive),
            RoleType.SUPPORTER: SupporterTree(us_positive=us_positive),
            RoleType.ATTACKER: AttackerTree(us_positive=us_positive),
        },
        us_positive=us_positive,
    )


def _pack_bt_state(tag, tick, phase, snapshot, robot_ids, coordinator, is_yellow):
    """Pack current BT state into a lightweight dict for the UI queue."""
    bx, by = snapshot.ball_position
    robot_map = {r.robot_id: r for r in snapshot.own_robots}
    robots = []
    for rid in robot_ids:
        bb = coordinator.blackboards.get(rid)
        if bb is None:
            continue
        r = robot_map.get(rid)
        intent = bb.current_intent if bb else None
        robots.append({
            "id":            rid,
            "role":          bb.current_role.value if bb else "?",
            "pos":           tuple(round(v, 3) for v in r.position) if r else None,
            "ori":           round(r.orientation, 3) if r else None,
            "intent_type":   type(intent).__name__.replace("Intent", "") if intent else None,
            "intent_target": tuple(round(v, 3) for v in intent.target_pos)
                             if intent and hasattr(intent, "target_pos") and intent.target_pos else None,
        })
    return {
        "tag": tag, "tick": tick, "phase": phase.value,
        "ball": (round(bx, 3), round(by, 3)),
        "is_yellow": is_yellow,
        "robots": robots,
    }


def run_bt_v2_process(
    is_running: Event,
    wm: WorldModel,
    dispatcher_q: Queue,
    is_yellow: bool | None = None,
    robot_ids: list[int] | None = None,
    role_assignment: dict | None = None,
    tick_period: float = TICK_PERIOD,
    config_file: str = "ipconfig.yaml",
    bt_state_q: "Queue | None" = None,
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
        tick_period: seconds to sleep between ticks.
        config_file: path to yaml config (relative to utils/).
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
    coordinator = _build_coordinator(us_positive=_us_positive)
    router = VoronoiRouter()
    logger = _BtLogger(robot_ids)

    tag = "[BT-YELLOW]" if is_yellow else "[BT-BLUE]"
    logger.proc("=" * 60)
    logger.proc(f"{tag} SESSION START  yellow={is_yellow}  us_positive={_us_positive}  "
                f"robot_ids={robot_ids}  enemy_goal={coordinator._enemy_goal}  "
                f"attack_sign={coordinator._attack_sign}")

    last_phase = None
    tick_count = 0

    try:
        while is_running.is_set():
            snapshot = build_snapshot_from_world_model(wm)
            if snapshot is None:
                time.sleep(tick_period)
                continue

            phase = snapshot.referee_state.game_phase

            if phase != last_phase:
                if last_phase is None:
                    logger.proc(f"{tag} initial phase: {phase}")
                else:
                    logger.proc(f"{tag} phase changed: {last_phase} → {phase}")
                last_phase = phase

            coordinator.tick(snapshot, robot_ids)

            tick_count += 1

            if bt_state_q is not None and tick_count % 30 == 0:
                try:
                    bt_state_q.put_nowait(_pack_bt_state(
                        tag, tick_count, phase, snapshot, robot_ids, coordinator, is_yellow,
                    ))
                except Exception:
                    pass

            if tick_count % 100 == 0:
                bx, by = snapshot.ball_position
                logger.proc(f"{tag} tick={tick_count} phase={phase.value} ball=({bx:.2f},{by:.2f})")
                robot_map = {r.robot_id: r for r in snapshot.own_robots}
                for rid in robot_ids:
                    bb = coordinator.blackboards.get(rid)
                    if bb and bb.current_intent:
                        r = robot_map.get(rid)
                        pos_str = (f"pos=({r.position[0]:.2f},{r.position[1]:.2f}) "
                                   f"facing={r.orientation:.2f}rad") if r else "pos=N/A"
                        logger.robot(rid, f"tick={tick_count} role={bb.current_role.value} "
                                         f"{pos_str} intent={bb.current_intent}")

            if phase in _HALT_PHASES:
                _send_stop_commands(robot_ids, is_yellow, dispatcher_q)
            else:
                dispatch_coordinator_output(
                    coordinator,
                    robot_ids,
                    snapshot,
                    is_yellow,
                    dispatcher_q,
                    wm=wm,
                    router=router,
                )
            time.sleep(tick_period)
    except KeyboardInterrupt:
        logger.proc(f"{tag} KeyboardInterrupt — exiting")
    finally:
        logger.proc(f"{tag} SESSION END  ticks={tick_count}")
        logger.close()
