"""
Backend engine — bridges the multiprocessing system with the Qt UI.

Manages process lifecycle, polls the WorldModel, and emits Qt signals
that drive every widget in the dashboard.
"""

import time
import math
from multiprocessing import Process, Queue, Event
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from TeamControl.process_workers.vision_runner import VisionProcess
from TeamControl.process_workers.gcfsm_runner import GCfsm
from TeamControl.process_workers.wm_runner import WMWorker
from TeamControl.process_workers.robot_recv_runner import RobotRecv
from TeamControl.process_workers.voronoi_map_runner import WorldMapRenderWorker
from TeamControl.world.model_manager import WorldModelManager
from TeamControl.world.field_config import (
    VORONOI_HORIZON_MS,
    VORONOI_OBSTACLE_COST_WEIGHT,
    VORONOI_RENDER_DENSITY_PERCENT,
    VORONOI_RENDER_MAX_DENSITY_NODES,
)
from TeamControl.dispatcher.dispatch import Dispatcher
from TeamControl.utils.yaml_config import Config
from TeamControl.world.recording import AsyncSnapshotRecorder

from TeamControl.robot.goalie import run_goalie
from TeamControl.robot.striker import run_striker
from TeamControl.robot.navigator import run_navigator, WAYPOINTS_A, WAYPOINTS_B
from TeamControl.robot.voronoi_navigator import run_voronoi_navigator
from TeamControl.robot.voronoi_game_navigator import run_voronoi_game_navigator
from TeamControl.robot.voronoi_pd_test_navigator import run_pd_planner_test
from TeamControl.robot.team import run_team
from TeamControl.robot.coop import run_coop
from TeamControl.bt.run_bt_v2_process import run_bt_v2_process
from TeamControl.utils.sim_config import Sim3v3Config, Sim6v6Config

from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.grSimPacketFactory import grSimPacketFactory
from TeamControl.network.robot_command import RobotCommand
from TeamControl.onboard_vision import (
    OnboardObservationStore, build_ip_map,
)

# ── Engine ───────────────────────────────────────────────────────────

class SimEngine(QObject):
    """Manages the multiprocessing backend and emits signals for UI."""

    frame_ready = Signal(object)         # WorldSnapshot
    map_render_ready = Signal(object)    # MapRenderData
    field_geometry_ready = Signal(object)  # FieldSize
    game_state_ready = Signal(object)    # GC status dict | GameState | None
    dispatch_info = Signal(object)       # dict snapshot from dispatcher
    channel_status_ready = Signal(object)  # runtime channel freshness
    engine_started = Signal(str)         # mode name
    engine_stopped = Signal()
    log_message = Signal(str)            # log line
    onboard_packet = Signal(object, object)  # (OnboardObservation, addr)
    bt_state_ready = Signal(object)      # BT state dict from run_bt_v2_process

    MODES = [
        "calibration",
        "vision_only",
        "voronoi_test",
        "pd_test",
        "match",
        "goalie",
        "1v1",
        "obstacle",
        "coop",
        "6v6",
        "btv2",
        "btv2_test",
        "bt_3v3",
        "bt_6v6",
    ]
    COMPETITION_MODES = {"6v6", "btv2", "bt_3v3", "bt_6v6"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config: Config | None = None
        self._wm_manager: WorldModelManager | None = None
        self._wm = None
        self._is_running: Event | None = None
        self._bg_procs: list[Process] = []
        self._fg_procs: list[Process] = []
        self._vision_q: Queue | None = None
        self._gc_q: Queue | None = None
        self._dispatch_q: Queue | None = None
        self._dispatch_info_q: Queue | None = None
        self._recv_q: Queue | None = None
        self._map_render_req_q: Queue | None = None
        self._map_render_resp_q: Queue | None = None
        self._planner_path_q: Queue | None = None
        self._grsim_sender: grSimSender | None = None
        self._field_manual_q: Queue | None = None
        self._snapshot_recorder: AsyncSnapshotRecorder | None = None
        self._channel_options: dict[str, bool] = {}
        self._channel_last_seen: dict[str, float] = {}
        self._channel_display_ms: dict[str, int] = {}
        self._bt_state_q: Queue | None = None
        self._last_frame_number: int = -1

        self._running = False
        self._mode = ""
        self._last_version = -1
        self._last_field_geometry_key = None
        self._last_map_render_emit_s = 0.0
        self._last_voronoi_latency_log_s = 0.0
        self._map_render_enabled = False
        self._latest_map_render_data = None
        self._latest_map_render_generation_ms = None
        self._latest_voronoi_generation_ms = None
        self._latest_planner_paths: dict[tuple[bool, int], dict] = {}
        self._map_render_request_pending = False
        self._map_render_request_id = 0

        self._onboard_store = OnboardObservationStore()
        self._ip_to_robot: dict[str, tuple[bool, int]] = {}
        self._onboard_last_ts: dict[tuple[bool, int], float] = {}

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(16)  # ~60 fps
        self._poll_timer.timeout.connect(self._poll)

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_running(self):
        return self._running

    @property
    def current_mode(self):
        return self._mode

    @property
    def config(self) -> Config | None:
        return self._config

    @property
    def onboard_store(self) -> OnboardObservationStore:
        return self._onboard_store

    @property
    def ip_to_robot(self) -> dict:
        return self._ip_to_robot

    def set_field_manual_control(self, shell_id: int, is_yellow: bool, enabled: bool):
        """
        While enabled, the dispatcher stops sending to this robot so Dashboard /
        Hardware Test field commands are not overwritten by AI (goalie, 6v6, …).
        """
        if not self._running or self._field_manual_q is None:
            return
        try:
            self._field_manual_q.put_nowait(
                ("on" if enabled else "off", int(shell_id), bool(is_yellow)))
        except Exception:
            pass

    def send_robot_command(self, command: RobotCommand, runtime: float = 0.20):
        """Queue a UI-originated command through the normal dispatcher path."""
        if not self._running or self._dispatch_q is None:
            return False
        try:
            self._dispatch_q.put_nowait((command, float(runtime), "manual"))
            return True
        except Exception:
            return False

    def dashboard_action_block_reason(self) -> str | None:
        """Return why field-click dashboard actions are disabled, if blocked."""
        if not self._running:
            return "engine is not running"
        if self._mode in self.COMPETITION_MODES:
            return f"competition mode is active ({self._mode})"
        if not self._channel_options.get("send_grsim", False):
            return "Send Commands to grSim is off"
        return None

    def dashboard_actions_allowed(self) -> bool:
        return self.dashboard_action_block_reason() is None

    def set_map_render_enabled(self, enabled: bool):
        """Enable the optional debug-map stream while its tab is visible."""
        self._map_render_enabled = bool(enabled)
        if enabled:
            self._last_map_render_emit_s = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def reload_config(self):
        self._config = Config()
        return self._config

    def start(
        self,
        mode: str = "goalie",
        our_id: int = 0,
        enemy_id: int = 0,
        channel_options: dict | None = None,
    ):
        if self._running:
            self.stop()

        self._config = Config()
        preset = self._config
        defaults = {
            "vision": True,
            "gc": True,
            "robot_recv": True,
            "use_grsim": bool(preset.use_grSim_vision),
            "send_grsim": bool(preset.send_to_grSim),
            "record_wm": bool(getattr(preset, "record_world_snapshots", False)),
        }
        if channel_options:
            defaults.update({k: bool(v) for k, v in channel_options.items()})
        self._channel_options = defaults
        self._channel_last_seen = {}
        self._channel_display_ms = {}

        preset.use_grSim_vision = defaults["use_grsim"]
        preset.send_to_grSim = defaults["send_grsim"]
        preset.record_world_snapshots = defaults["record_wm"]
        self._ip_to_robot = build_ip_map(preset)

        self._vision_q = Queue()
        self._gc_q = Queue()
        self._dispatch_q = Queue()
        self._dispatch_info_q = Queue()
        self._recv_q = Queue()
        self._map_render_req_q = Queue()
        self._map_render_resp_q = Queue()
        self._planner_path_q = Queue()
        self._field_manual_q = Queue()
        self._is_running = Event()

        self._wm_manager = WorldModelManager()
        self._wm_manager.start()
        self._wm = self._wm_manager.WorldModel()

        self._bg_procs = []
        if defaults["vision"]:
            self._bg_procs.append(
                Process(target=VisionProcess.run_worker,
                        args=(self._is_running, None, self._vision_q,
                              preset.use_grSim_vision, preset.vision[1]),
                        daemon=True))
        if defaults["gc"]:
            self._bg_procs.append(
                Process(target=GCfsm.run_worker,
                        args=(self._is_running, None, self._gc_q,
                              preset.us_yellow, preset.us_positive, preset.team_name),
                        daemon=True))
        self._bg_procs.append(
            Process(target=WMWorker.run_worker,
                    args=(self._is_running, None, self._wm,
                          self._vision_q, self._gc_q,
                          self._recv_q, dict(self._ip_to_robot)),
                    daemon=True))
        self._bg_procs.append(
            Process(target=Dispatcher.run_worker,
                    args=(self._is_running, None, self._dispatch_q, preset,
                          self._dispatch_info_q, self._field_manual_q),
                    daemon=True))
        self._bg_procs.append(
            Process(target=WorldMapRenderWorker.run_worker,
                    args=(self._is_running, None, self._map_render_req_q,
                          self._map_render_resp_q),
                    daemon=True))
        if defaults["robot_recv"]:
            self._bg_procs.append(
                Process(target=RobotRecv.run_worker,
                        args=(self._is_running, None, preset.robot_ip,
                              self._recv_q),
                        daemon=True))

        self._bt_state_q = Queue(maxsize=4)
        self._last_frame_number = -1
        self._fg_procs = self._build_foreground(mode, preset, our_id, enemy_id)

        self._is_running.set()
        for p in self._bg_procs:
            p.start()
        for p in self._fg_procs:
            p.start()

        try:
            self._grsim_sender = grSimSender(*preset.grSim_addr)
        except Exception:
            self._grsim_sender = None

        if preset.record_world_snapshots:
            replay_dir = (
                Path(preset.record_world_snapshot_dir)
                / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}"
            )
            self._snapshot_recorder = AsyncSnapshotRecorder(replay_dir)
            self.log_message.emit(f"[record] World snapshots -> {replay_dir}")
        else:
            self._snapshot_recorder = None

        self._running = True
        self._mode = mode
        self._last_version = -1
        self._last_field_geometry_key = None
        self._last_map_render_emit_s = 0.0
        self._last_voronoi_latency_log_s = 0.0
        self._latest_map_render_data = None
        self._latest_map_render_generation_ms = None
        self._latest_voronoi_generation_ms = None
        self._latest_planner_paths = {}
        self._map_render_request_pending = False
        self._map_render_request_id = 0
        self._poll_timer.start()

        self.engine_started.emit(mode)
        self.log_message.emit(f"[engine] Started mode: {mode}")
        self._emit_channel_status()

    def stop(self):
        if not self._running:
            return
        self._poll_timer.stop()
        if self._is_running:
            self._is_running.clear()

        for p in self._fg_procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        for p in self._bg_procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()

        self._fg_procs.clear()
        self._bg_procs.clear()

        if self._snapshot_recorder is not None:
            dropped = self._snapshot_recorder.dropped
            self._snapshot_recorder.close()
            self.log_message.emit(
                f"[record] Snapshot recorder stopped; dropped={dropped}"
            )
            self._snapshot_recorder = None

        if self._wm_manager:
            try:
                self._wm_manager.shutdown()
            except Exception:
                pass

        self._wm = None
        self._wm_manager = None
        self._bt_state_q = None
        self._last_frame_number = -1
        self._recv_q = None
        self._map_render_req_q = None
        self._map_render_resp_q = None
        self._planner_path_q = None
        self._dispatch_info_q = None
        self._field_manual_q = None
        self._running = False
        self._mode = ""
        self._grsim_sender = None
        self._channel_options = {}
        self._channel_last_seen = {}
        self._channel_display_ms = {}

        self.engine_stopped.emit()
        self.log_message.emit("[engine] Stopped")

    # ── Foreground builder ────────────────────────────────────────

    def _build_foreground(self, mode, preset, our_id=0, enemy_id=0):
        procs = []
        wm = self._wm
        dq = self._dispatch_q
        ev = self._is_running

        self.log_message.emit(
            f"[engine] Building {mode}: our shell={our_id} "
            f"({'yellow' if preset.us_yellow else 'blue'}), "
            f"enemy shell={enemy_id} "
            f"({'blue' if preset.us_yellow else 'yellow'})")

        if mode == "calibration":
            self.log_message.emit(
                "[engine] Calibration mode - no robot behaviours running")
            return procs

        if mode == "vision_only":
            self.log_message.emit(
                "[engine] Vision-only mode — no robot models running")
            return procs

        if mode == "voronoi_test":
            us_y = preset.us_yellow
            for rid in (0, 1, 2):
                procs.append(Process(
                    target=run_voronoi_navigator,
                    args=(ev, dq, wm, rid, us_y, self._planner_path_q),
                    kwargs=dict(kick_at_ball=True),
                    daemon=True,
                ))
            team = "yellow" if us_y else "blue"
            self.log_message.emit(
                f"[engine] Voronoi test mode — 3 {team} robots (0,1,2) chasing "
                "ball with kick on arrival")
            return procs

        if mode == "pd_test":
            # Single robot only -- live integration test for
            # RobotMotionController's rule set (motion/controller.py),
            # which otherwise only the PD calibration harness exercises.
            procs.append(Process(target=run_pd_planner_test,
                                 args=(ev, dq, wm, our_id, preset.us_yellow,
                                       self._planner_path_q),
                                 daemon=True))
            self.log_message.emit(
                "[engine] PD test mode — running one robot through the live "
                "Voronoi planner, driven by RobotMotionController (PD)")
            return procs

        if mode == "match":
            us_y = preset.us_yellow
            enemy_y = not us_y
            our_is_goalie = bool(
                our_id == preset.goalie_yellow_id if us_y
                else our_id == preset.goalie_blue_id
            )
            enemy_is_goalie = bool(
                enemy_id == preset.goalie_yellow_id if enemy_y
                else enemy_id == preset.goalie_blue_id
            )
            procs.append(Process(target=run_voronoi_game_navigator,
                                 args=(ev, dq, wm, our_id, us_y,
                                       self._planner_path_q),
                                 kwargs=dict(is_goalie=our_is_goalie),
                                 daemon=True))
            procs.append(Process(target=run_voronoi_game_navigator,
                                 args=(ev, dq, wm, enemy_id, enemy_y,
                                       self._planner_path_q),
                                 kwargs=dict(is_goalie=enemy_is_goalie),
                                 daemon=True))
            self.log_message.emit(
                "[engine] Match mode — running one yellow and one blue robot "
                "through the full game navigator (penalty-box guard, steal, "
                "precision approach, smoothing)")
            return procs

        if mode == "goalie":
            procs.append(Process(target=run_goalie,
                                 args=(ev, dq, wm, our_id, preset.us_yellow),
                                 daemon=True))
            procs.append(Process(target=run_striker,
                                 args=(ev, dq, wm, enemy_id, not preset.us_yellow),
                                 daemon=True))
        elif mode == "1v1":
            procs.append(Process(target=run_striker,
                                 args=(ev, dq, wm, our_id, True),
                                 daemon=True))
            procs.append(Process(target=run_striker,
                                 args=(ev, dq, wm, enemy_id, False),
                                 daemon=True))
        elif mode == "obstacle":
            procs.append(Process(target=run_navigator,
                                 args=(ev, dq, wm, our_id,
                                       preset.us_yellow, WAYPOINTS_A),
                                 daemon=True))
            procs.append(Process(target=run_navigator,
                                 args=(ev, dq, wm, enemy_id,
                                       preset.us_yellow, WAYPOINTS_B),
                                 daemon=True))
        elif mode == "coop":
            # Cross-team coop: our bot + enemy bot cooperate left → right
            us_y = preset.us_yellow
            enemy_y = not us_y
            procs.append(Process(target=run_coop,
                                 args=(ev, dq, wm, our_id, enemy_id, us_y),
                                 kwargs=dict(mate_is_yellow=enemy_y,
                                             attack_positive=True,
                                             grsim_addr=preset.grSim_addr),
                                 daemon=True))
            procs.append(Process(target=run_coop,
                                 args=(ev, dq, wm, enemy_id, our_id, enemy_y),
                                 kwargs=dict(mate_is_yellow=us_y,
                                             attack_positive=True,
                                             grsim_addr=preset.grSim_addr),
                                 daemon=True))
        elif mode == "6v6":
            procs.append(Process(target=run_team,
                                 args=(ev, dq, wm, True, our_id),
                                 daemon=True))
            procs.append(Process(target=run_team,
                                 args=(ev, dq, wm, False, enemy_id),
                                 daemon=True))

        if mode == "btv2":
            procs.append(Process(
                target=run_bt_v2_process,
                args=(ev, wm, dq),
                kwargs=dict(is_yellow=preset.us_yellow, bt_state_q=self._bt_state_q),
                daemon=True,
            ))
            team = "yellow" if preset.us_yellow else "blue"
            self.log_message.emit(
                f"[engine] BT v2 mode — {team} team, robots 0-5 via Coordinator"
            )
            return procs

        if mode == "btv2_test":
            procs.append(Process(
                target=run_bt_v2_process,
                args=(ev, wm, dq),
                kwargs=dict(
                    is_yellow=preset.us_yellow,
                    robot_ids=[our_id],
                    bt_state_q=self._bt_state_q,
                ),
                daemon=True,
            ))
            team = "yellow" if preset.us_yellow else "blue"
            self.log_message.emit(
                f"[engine] BT v2 test mode — {team} robot #{our_id} via Coordinator"
            )
            return procs

        if mode in ("bt_3v3", "bt_6v6"):
            sim = Sim3v3Config() if mode == "bt_3v3" else Sim6v6Config()
            roles = {rid: role.name for rid, role in sim.roles.items()}
            self.log_message.emit(
                f"[engine] {mode} mode - yellow={sim.yellow_ids} "
                f"blue={sim.blue_ids} roles={roles} "
                f"heuristic_role_swap={sim.heuristic_role_swap}"
            )
            for is_yellow, robot_ids, label in (
                (True, sim.yellow_ids, "yellow"),
                (False, sim.blue_ids, "blue"),
            ):
                procs.append(
                    Process(
                        target=run_bt_v2_process,
                        args=(ev, wm, dq),
                        kwargs=dict(
                            is_yellow=is_yellow,
                            robot_ids=robot_ids,
                            role_assignment=sim.roles,
                            heuristic_role_swap=sim.heuristic_role_swap,
                            movement_safety=sim.movement_safety,
                            tick_period=sim.tick_period,
                            bt_state_q=self._bt_state_q,
                        ),
                        daemon=True,
                        name=f"{mode}_{label}",
                    )
                )
            return procs

        return procs

    # ── Polling ───────────────────────────────────────────────────

    def _poll(self):
        if not self._wm:
            return
        try:
            frame = self._wm.get_latest_frame()
            if frame is not None:
                self._sync_field_geometry()
                snap = self._wm.snapshot()
                fn = getattr(snap, "frame_number", -1)
                if fn != self._last_frame_number:
                    self._last_frame_number = fn
                    self._mark_channel_seen("vision")
                    self.frame_ready.emit(snap)
                monotonic_s = time.monotonic()
                if (
                    self._map_render_enabled
                    and monotonic_s - self._last_map_render_emit_s >= 0.1
                ):
                    self._last_map_render_emit_s = monotonic_s
                    now_s = time.time()
                    self._drain_planner_paths(now_s)
                    self._drain_map_render_worker()
                    self._request_map_render_data(now_s)
                    if self._latest_map_render_data is not None:
                        self.map_render_ready.emit(self._latest_map_render_data)
                    if monotonic_s - self._last_voronoi_latency_log_s >= 1.0:
                        render_ms = self._latest_map_render_generation_ms
                        voronoi_ms = self._latest_voronoi_generation_ms
                        if render_ms is not None:
                            self.log_message.emit(
                                "[map] World map worker generated in "
                                f"{render_ms:.2f} ms"
                                + (
                                    f" (Voronoi {voronoi_ms:.2f} ms)"
                                    if voronoi_ms is not None
                                    else ""
                                )
                            )
                            self._last_voronoi_latency_log_s = monotonic_s

            ver = self._wm.get_version()
            if ver != self._last_version:
                self._last_version = ver
                gc_status = self._wm.get_gc_status()
                if gc_status.get("received_at") is not None:
                    self._mark_channel_seen("gc")
                self.game_state_ready.emit(gc_status)
                if self._snapshot_recorder is not None:
                    self._record_world_snapshot()
        except Exception as exc:
            self.log_message.emit(f"[engine] poll error: {exc}")

        self._sync_onboard_from_wm()
        self._drain_dispatch_info()
        self._emit_channel_status()
        self._drain_bt_state_q()

    def _drain_bt_state_q(self):
        if self._bt_state_q is None:
            return
        latest = None
        try:
            while not self._bt_state_q.empty():
                latest = self._bt_state_q.get_nowait()
        except Exception:
            pass
        if latest is not None:
            self.bt_state_ready.emit(latest)

    def _request_map_render_data(self, now_s: float):
        if (
            self._map_render_request_pending
            or self._map_render_req_q is None
            or self._wm is None
        ):
            return
        try:
            obstacles = self._wm.get_obstacles()
        except Exception as exc:
            self.log_message.emit(f"[map] render worker obstacle error: {exc}")
            obstacles = ()
        try:
            planning_obstacles = self._wm.get_planning_obstacles(
                now_s=now_s,
                horizon_ms=VORONOI_HORIZON_MS,
            )
        except Exception as exc:
            self.log_message.emit(f"[map] render worker obstacle error: {exc}")
            planning_obstacles = ()

        try:
            snap = self._wm.snapshot()
            ball = snap.ball.position if snap.ball is not None else None
            ball_visible = bool(snap.ball and snap.ball.visible)
            ball_vel_mmps = self._wm.get_ball_trajectory(horizon_ms=0)
            ball_vel_mmps = ball_vel_mmps[1] if ball_vel_mmps else (0.0, 0.0)
            field = self._wm.get_field_size()
            field_length_mm = _positive_float(getattr(field, "field_length", None))
            field_width_mm = _positive_float(getattr(field, "field_width", None))
            self._map_render_request_id += 1
            request = {
                "request_id": self._map_render_request_id,
                "obstacles": tuple(obstacles),
                "planning_obstacles": tuple(planning_obstacles),
                "ball": ball,
                "ball_visible": ball_visible,
                "ball_vel_mmps": ball_vel_mmps,
                "planner_paths": tuple(self._latest_planner_paths.values()),
                "include_voronoi": True,
                "density_percent": VORONOI_RENDER_DENSITY_PERCENT,
                "max_density_nodes": VORONOI_RENDER_MAX_DENSITY_NODES,
                "obstacle_cost_weight": VORONOI_OBSTACLE_COST_WEIGHT,
            }
            if field_length_mm is not None and field_width_mm is not None:
                request["field_length_mm"] = field_length_mm
                request["field_width_mm"] = field_width_mm
            self._map_render_req_q.put_nowait(request)
            self._map_render_request_pending = True
        except Exception as exc:
            self.log_message.emit(f"[map] render worker request error: {exc}")

    def _drain_map_render_worker(self):
        if self._map_render_resp_q is None:
            return
        latest = None
        while True:
            try:
                latest = self._map_render_resp_q.get_nowait()
            except Exception:
                break
        if latest is None:
            return
        self._map_render_request_pending = False
        if latest.get("error"):
            self.log_message.emit(f"[map] render worker error: {latest['error']}")
            return
        self._latest_map_render_data = latest.get("render_data")
        self._latest_map_render_generation_ms = latest.get("generation_ms")
        self._latest_voronoi_generation_ms = latest.get("voronoi_generation_ms")

    def _drain_planner_paths(self, now_s: float):
        if self._planner_path_q is None:
            return
        while True:
            try:
                update = self._planner_path_q.get_nowait()
            except Exception:
                break
            key = (bool(update.get("is_yellow", True)), int(update.get("robot_id", 0)))
            self._latest_planner_paths[key] = update

        stale_keys = [
            key for key, update in self._latest_planner_paths.items()
            if now_s - float(update.get("timestamp_s", now_s)) > 1.0
        ]
        for key in stale_keys:
            self._latest_planner_paths.pop(key, None)

    def _sync_field_geometry(self):
        try:
            field = self._wm.get_field_size()
        except Exception:
            return
        if field is None:
            return
        key = (
            getattr(field, "field_length", None),
            getattr(field, "field_width", None),
            getattr(field, "goal_width", None),
            getattr(field, "goal_depth", None),
            getattr(field, "boundary_width", None),
            getattr(field, "penalty_area_depth", None),
            getattr(field, "penalty_area_width", None),
        )
        if key == self._last_field_geometry_key:
            return
        self._last_field_geometry_key = key
        self.field_geometry_ready.emit(field)

    def _drain_dispatch_info(self):
        if self._dispatch_info_q is None:
            return
        latest = None
        try:
            while True:
                latest = self._dispatch_info_q.get_nowait()
        except Exception:
            pass
        if latest is not None:
            self._mark_channel_seen("send_grsim")
            self.dispatch_info.emit(latest)

    def _record_world_snapshot(self):
        try:
            snap = self._wm.snapshot()
            if not self._snapshot_recorder.write(snap):
                self.log_message.emit("[record] Snapshot queue full; dropping")
            else:
                self._mark_channel_seen("record_wm")
        except Exception as exc:
            self.log_message.emit(f"[record] snapshot failed: {exc}")

    def _sync_onboard_from_wm(self):
        """Mirror new onboard observations from the shared WM into the
        UI-local store and emit signals. The recv_q is drained by WMWorker
        (the sole consumer); the WM is the source of truth."""
        if not self._wm:
            return
        try:
            snapshot = self._wm.onboard_snapshot()
        except Exception:
            return
        for key, obs in snapshot.items():
            ts = getattr(obs, "recv_ts", 0.0)
            if ts <= self._onboard_last_ts.get(key, 0.0):
                continue
            self._onboard_last_ts[key] = ts
            self._mark_channel_seen("robot_recv")
            self._onboard_store.put(obs)
            self.onboard_packet.emit(obs, None)

    def _mark_channel_seen(self, name: str):
        now = time.time()
        previous = self._channel_last_seen.get(name)
        self._channel_last_seen[name] = now
        if previous is None:
            self._channel_display_ms[name] = 0
        else:
            self._channel_display_ms[name] = min(int((now - previous) * 1000.0), 99)

    def _emit_channel_status(self):
        now = time.time()
        stale_after_ms = {
            "vision": 150,
            "use_grsim": 150,
            "gc": 1000,
            "robot_recv": 1000,
            "send_grsim": 1000,
            "record_wm": 1500,
        }
        status = {}
        for name, enabled in self._channel_options.items():
            last_name = "vision" if name == "use_grsim" else name
            last = self._channel_last_seen.get(last_name)
            age_ms = None if last is None else int((now - last) * 1000.0)
            stale_limit = stale_after_ms.get(name, 1000)
            stale = age_ms is None or age_ms > stale_limit
            display_ms = self._channel_display_ms.get(last_name)
            status[name] = {
                "enabled": bool(enabled),
                "latency_ms": None if stale else display_ms,
                "age_ms": age_ms,
                "stale": stale,
            }
        self.channel_status_ready.emit(status)

    # ── Simulation controls ───────────────────────────────────────

    def place_ball(self, x_mm, y_mm, vx=0.0, vy=0.0):
        blocked = self.dashboard_action_block_reason()
        if blocked:
            self.log_message.emit(f"[dashboard] Ball placement blocked: {blocked}")
            return False
        if not self._grsim_sender:
            self.log_message.emit(
                "[dashboard] Ball placement blocked: grSim sender unavailable"
            )
            return False
        try:
            pkt = grSimPacketFactory.ball_replacement_command(
                x=x_mm / 1000.0, y=y_mm / 1000.0,
                vx=vx / 1000.0, vy=vy / 1000.0)
            self._grsim_sender.send_packet(pkt)
            self.log_message.emit(f"[sim] Ball placed at ({x_mm:.0f}, {y_mm:.0f})")
            return True
        except Exception as e:
            self.log_message.emit(f"[sim] Ball placement failed: {e}")
            return False

    def place_robot(self, robot_id, is_yellow, x_mm, y_mm, orientation=0.0):
        blocked = self.dashboard_action_block_reason()
        if blocked:
            self.log_message.emit(f"[dashboard] Robot placement blocked: {blocked}")
            return False
        if not self._grsim_sender:
            self.log_message.emit(
                "[dashboard] Robot placement blocked: grSim sender unavailable"
            )
            return False
        try:
            pkt = grSimPacketFactory.robot_replacement_command(
                x=x_mm / 1000.0, y=y_mm / 1000.0,
                orientation=orientation,
                robot_id=robot_id, isYellow=is_yellow)
            self._grsim_sender.send_packet(pkt)
            team = "Yellow" if is_yellow else "Blue"
            self.log_message.emit(
                f"[sim] {team} #{robot_id} placed at ({x_mm:.0f}, {y_mm:.0f})")
            return True
        except Exception as e:
            self.log_message.emit(f"[sim] Robot placement failed: {e}")
            return False


def _positive_float(value):
    if value is None:
        return None
    value = float(value)
    if value <= 0.0:
        return None
    return value
