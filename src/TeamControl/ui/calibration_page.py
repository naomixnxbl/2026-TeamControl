"""
Calibration Page — learn and apply drive calibration values.

Auto-calibrate drives the robot back and forth across the field,
measures actual speed vs commanded speed and lateral drift,
then saves correction factors to calibration.json.

Layout:
  +------------------------------+-----------------------+
  |                              |  Speed / Passes       |
  |        Field Canvas          |  Auto Calibrate       |
  |        (own instance)        |  Learned values       |
  |                              |  Log                  |
  +------------------------------+-----------------------+
"""

import json
import math
import os
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QFrame, QDoubleSpinBox, QSpinBox,
    QPlainTextEdit, QProgressBar, QScrollArea, QGridLayout,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from TeamControl.ui.theme import (
    ACCENT, TEXT_DIM, TEXT, SUCCESS, WARNING, DANGER,
    BORDER, BG_DARK, BG_CARD,
)
from TeamControl.ui.field_canvas import FieldCanvas
from TeamControl.network.robot_command import RobotCommand
from TeamControl.network.sender import Sender
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.robot.constants import HALF_LEN, HALF_WID, FIELD_MARGIN, MAX_W
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import clamp, move_toward, wall_brake


# ── Constants ─────────────────────────────────────────────────────
_MARGIN = 400                # mm from field edge for start/end points
_ARRIVE = 100                # mm — close enough to a waypoint
_SETTLE_MS = 1200            # ms wait after stop to measure final pose
_TURN_GAIN = 0.5             # angular velocity gain
_NAV_SPEED = 0.35            # speed for navigating to start position

# Calibration data file
_CAL_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "calibration.json"))


def _card(title_text):
    """Create a styled card frame with title."""
    card = QFrame()
    card.setObjectName("card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(6)
    title = QLabel(title_text)
    title.setStyleSheet(
        f"font-size:13px; font-weight:bold; color:{ACCENT}; padding:0;")
    lay.addWidget(title)
    return card, lay


def _label(text, color=TEXT_DIM, size=11):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color:{color}; font-size:{size}px;")
    return lbl


def _val_label(text="—"):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{TEXT}; font-size:12px; font-family:monospace; padding:2px;")
    return lbl


class CalibrationPage(QWidget):
    """Calibration tab — auto-learn speed scale and drift correction."""

    def __init__(self, parent=None, engine=None, test_panel=None):
        super().__init__(parent)
        self._engine = engine
        self._test_panel = test_panel
        self._our_id_spin = None
        self._sender = Sender()

        # ── Load existing calibration ─────────────────────────
        self._cal = self._load_cal()

        # ── Layout ────────────────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # Left: field canvas (separate widget from Dashboard — must wire signals here)
        self._field = FieldCanvas()
        splitter.addWidget(self._field)
        if self._test_panel is not None:
            self._field.point_picked.connect(self._test_panel.go_to_point)
            self._field.action_requested.connect(self._test_panel.field_action)
        if self._engine is not None:
            self._field.ball_placed.connect(
                lambda x, y: self._engine.place_ball(x, y))
            self._field.robot_placed.connect(
                lambda rid, yl, x, y: self._engine.place_robot(
                    rid, yl, x, y))

        # Right: scrollable sidebar
        sidebar = QWidget()
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(500)
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(0, 0, 0, 0)
        sb_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {BG_DARK}; border: none; }}")
        scroll_inner = QWidget()
        scroll_inner.setStyleSheet(f"background: {BG_DARK};")
        inner = QVBoxLayout(scroll_inner)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(8)

        self._build_settings_card(inner)
        self._build_auto_card(inner)
        self._build_values_card(inner)
        self._build_log_card(inner)

        inner.addStretch()
        scroll.setWidget(scroll_inner)
        sb_lay.addWidget(scroll)
        splitter.addWidget(sidebar)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1100, 400])
        root.addWidget(splitter)

        # ── Timers ────────────────────────────────────────────
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(50)   # 20 Hz
        self._tick_timer.timeout.connect(self._tick)

        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.timeout.connect(self._on_settle)

        # Frame polling — keep field canvas in sync
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(50)
        self._frame_timer.timeout.connect(self._poll_frame)
        self._frame_timer.start()

        # ── State ─────────────────────────────────────────────
        self._phase = None          # "nav" | "running" | "settling" | None
        self._pass_idx = 0          # current pass number
        self._total_passes = 0
        self._direction = 1         # 1 = left→right, -1 = right→left
        self._start_wp = None       # (x, y) start of current run
        self._end_wp = None         # (x, y) end of current run
        self._run_start_time = 0.0
        self._run_start_pose = None
        self._stop_pose = None
        self._test_speed = 0.0
        self._line_y = 0.0          # ideal Y for drift measurement

        self._ideal_heading = 0.0   # heading to hold during runs

        # Collected data for current auto-cal session
        self._run_results = []      # list of dicts per pass

        # Speed sweep state
        self._sweep_mode = False
        self._sweep_speeds = []     # list of speeds to test
        self._sweep_idx = 0         # current speed index
        self._sweep_results = {}    # speed -> list of run results

    # ══════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════

    def set_our_bot_spin(self, spin):
        self._our_id_spin = spin

    # ══════════════════════════════════════════════════════════════
    #  UI Builders
    # ══════════════════════════════════════════════════════════════

    def _build_settings_card(self, parent_lay):
        card, lay = _card("Test Settings")

        # Speed
        row = QHBoxLayout()
        row.addWidget(_label("Speed (m/s):"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.05, 1.0)
        self._speed_spin.setSingleStep(0.05)
        self._speed_spin.setValue(0.30)
        self._speed_spin.setFixedWidth(80)
        row.addWidget(self._speed_spin)
        row.addStretch()
        lay.addLayout(row)

        # Passes
        row2 = QHBoxLayout()
        row2.addWidget(_label("Passes (back & forth):"))
        self._passes_spin = QSpinBox()
        self._passes_spin.setRange(1, 20)
        self._passes_spin.setValue(4)
        self._passes_spin.setFixedWidth(80)
        row2.addWidget(self._passes_spin)
        row2.addStretch()
        lay.addLayout(row2)

        # Status
        self._status_lbl = _label("Ready", TEXT_DIM, 12)
        lay.addWidget(self._status_lbl)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(14)
        self._progress.setTextVisible(False)
        lay.addWidget(self._progress)

        parent_lay.addWidget(card)

    def _build_auto_card(self, parent_lay):
        card, lay = _card("Auto Calibrate")

        lay.addWidget(_label(
            "Drives the robot left\u2194right across the field, "
            "measures actual speed and lateral drift, "
            "then computes correction factors.",
            TEXT_DIM, 11))

        # Buttons
        btn_row = QHBoxLayout()

        self._start_btn = QPushButton("Start Calibration")
        self._start_btn.setStyleSheet(
            f"background:{ACCENT}; color:white; padding:8px 16px; "
            f"border-radius:4px; font-weight:bold;")
        self._start_btn.clicked.connect(self._start_auto)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet(
            f"background:{DANGER}; color:white; padding:8px 16px; "
            f"border-radius:4px; font-weight:bold;")
        self._stop_btn.clicked.connect(self._stop_test)
        btn_row.addWidget(self._stop_btn)

        lay.addLayout(btn_row)

        btn_row2 = QHBoxLayout()

        self._single_btn = QPushButton("Single Pass")
        self._single_btn.setStyleSheet(
            f"background:{BG_CARD}; color:{TEXT}; padding:6px 12px; "
            f"border:1px solid {BORDER}; border-radius:4px;")
        self._single_btn.clicked.connect(self._start_single)
        btn_row2.addWidget(self._single_btn)

        self._sweep_btn = QPushButton("Speed Sweep")
        self._sweep_btn.setToolTip(
            "Test multiple speeds to find the fastest accurate one")
        self._sweep_btn.setStyleSheet(
            f"background:{BG_CARD}; color:{TEXT}; padding:6px 12px; "
            f"border:1px solid {BORDER}; border-radius:4px;")
        self._sweep_btn.clicked.connect(self._start_sweep)
        btn_row2.addWidget(self._sweep_btn)

        lay.addLayout(btn_row2)

        parent_lay.addWidget(card)

    def _build_values_card(self, parent_lay):
        card, lay = _card("Calibration Values")

        lay.addWidget(_label(
            "Edit values manually or auto-calibrate to learn them.",
            TEXT_DIM, 11))

        grid = QGridLayout()
        grid.setSpacing(6)

        # Speed scale
        grid.addWidget(_label("Speed scale:"), 0, 0)
        self._val_speed_spin = QDoubleSpinBox()
        self._val_speed_spin.setRange(0.01, 5.0)
        self._val_speed_spin.setDecimals(4)
        self._val_speed_spin.setSingleStep(0.01)
        self._val_speed_spin.setFixedWidth(100)
        grid.addWidget(self._val_speed_spin, 0, 1)
        grid.addWidget(_label("actual / commanded", TEXT_DIM, 10), 0, 2)

        # Lateral drift
        grid.addWidget(_label("Lateral drift/m:"), 1, 0)
        self._val_drift_spin = QDoubleSpinBox()
        self._val_drift_spin.setRange(-50.0, 50.0)
        self._val_drift_spin.setDecimals(2)
        self._val_drift_spin.setSingleStep(0.5)
        self._val_drift_spin.setSuffix(" mm/m")
        self._val_drift_spin.setFixedWidth(120)
        grid.addWidget(self._val_drift_spin, 1, 1)

        # Stop overshoot
        grid.addWidget(_label("Stop overshoot:"), 2, 0)
        self._val_stop_spin = QDoubleSpinBox()
        self._val_stop_spin.setRange(-200.0, 200.0)
        self._val_stop_spin.setDecimals(1)
        self._val_stop_spin.setSingleStep(5.0)
        self._val_stop_spin.setSuffix(" mm")
        self._val_stop_spin.setFixedWidth(120)
        grid.addWidget(self._val_stop_spin, 2, 1)

        # Runs count (read-only info)
        grid.addWidget(_label("Total runs:"), 3, 0)
        self._val_runs = _val_label()
        grid.addWidget(self._val_runs, 3, 1)

        lay.addLayout(grid)
        self._refresh_values()

        # Buttons
        btn_row = QHBoxLayout()

        self._apply_btn = QPushButton("Apply && Save")
        self._apply_btn.setStyleSheet(
            f"background:{SUCCESS}; color:white; padding:6px 14px; "
            f"border-radius:4px; font-weight:bold;")
        self._apply_btn.clicked.connect(self._apply_cal)
        btn_row.addWidget(self._apply_btn)

        self._reset_btn = QPushButton("Reset to Defaults")
        self._reset_btn.setStyleSheet(
            f"background:{BG_CARD}; color:{WARNING}; padding:6px 14px; "
            f"border:1px solid {BORDER}; border-radius:4px;")
        self._reset_btn.clicked.connect(self._reset_cal)
        btn_row.addWidget(self._reset_btn)

        lay.addLayout(btn_row)
        parent_lay.addWidget(card)

    def _build_log_card(self, parent_lay):
        card, lay = _card("Results Log")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet(
            f"background:{BG_DARK}; color:{TEXT_DIM}; "
            f"border:1px solid {BORDER}; font-family:monospace; font-size:11px;")
        lay.addWidget(self._log)

        clr_btn = QPushButton("Clear Log")
        clr_btn.setStyleSheet(
            f"background:{BG_CARD}; color:{TEXT_DIM}; padding:4px 10px; "
            f"border:1px solid {BORDER}; border-radius:3px;")
        clr_btn.clicked.connect(self._log.clear)
        lay.addWidget(clr_btn)

        parent_lay.addWidget(card)

    # ══════════════════════════════════════════════════════════════
    #  Frame polling
    # ══════════════════════════════════════════════════════════════

    def _poll_frame(self):
        if not self._engine or not self._engine._wm:
            return
        try:
            frame = self._engine._wm.get_latest_frame()
            if frame is None:
                return
            snap = self._engine._extract_snapshot(frame)
            self._field.set_frame(snap)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  Robot helpers
    # ══════════════════════════════════════════════════════════════

    def _get_rid_and_yellow(self):
        rid = self._our_id_spin.value() if self._our_id_spin else 0
        cfg = self._engine.config if self._engine else None
        is_yellow = cfg.us_yellow if cfg else True
        return rid, is_yellow

    def _get_robot_pose(self):
        if not self._engine or not self._engine._wm:
            return None
        frame = self._engine._wm.get_latest_frame()
        if frame is None:
            return None
        rid, is_yellow = self._get_rid_and_yellow()
        team = frame.robots_yellow if is_yellow else frame.robots_blue
        try:
            robot = team[rid]
        except (IndexError, TypeError):
            return None
        from TeamControl.SSL.vision.robots import Robot
        if not isinstance(robot, Robot):
            return None
        return (float(robot.x), float(robot.y), float(robot.o))

    def _send_cmd(self, vx=0.0, vy=0.0, w=0.0, kick=0, dribble=0):
        rid, is_yellow = self._get_rid_and_yellow()
        cmd = RobotCommand(
            robot_id=rid, vx=vx, vy=vy, w=w,
            kick=kick, dribble=dribble, isYellow=is_yellow)
        if self._test_panel:
            self._test_panel._do_send(cmd)

    def _send_stop(self):
        self._send_cmd()

    # ══════════════════════════════════════════════════════════════
    #  Calibration data persistence
    # ══════════════════════════════════════════════════════════════

    def _default_cal(self):
        return {
            "speed_scale": 1.0,
            "lateral_drift_per_m": 0.0,
            "stop_overshoot_mm": 0.0,
            "runs": [],
        }

    def _load_cal(self):
        try:
            with open(_CAL_PATH, "r") as f:
                data = json.load(f)
            cal = self._default_cal()
            for k in cal:
                if k in data:
                    cal[k] = data[k]
            return cal
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return self._default_cal()

    def _save_cal(self):
        try:
            with open(_CAL_PATH, "w") as f:
                json.dump(self._cal, f, indent=2)
        except Exception as e:
            self._set_status(f"Save failed: {e}", DANGER)

    def _refresh_values(self):
        """Update the displayed calibration values from internal state."""
        c = self._cal
        self._val_speed_spin.setValue(c['speed_scale'])
        self._val_drift_spin.setValue(c['lateral_drift_per_m'])
        self._val_stop_spin.setValue(c['stop_overshoot_mm'])
        n = len(c.get("runs", []))
        self._val_runs.setText(str(n))

    # ══════════════════════════════════════════════════════════════
    #  Test lifecycle
    # ══════════════════════════════════════════════════════════════

    def _start_auto(self):
        """Start full auto-calibration: multiple back-and-forth passes."""
        pose = self._get_robot_pose()
        if pose is None:
            self._set_status("No robot — is engine running?", DANGER)
            return

        self._sweep_mode = False
        self._test_speed = self._speed_spin.value()
        self._total_passes = self._passes_spin.value()
        self._pass_idx = 0
        self._direction = 1   # start going left → right
        self._run_results = []
        self._line_y = pose[1]  # hold this Y throughout

        self._set_status("Auto calibrating...", ACCENT)
        self._begin_pass()

    def _start_sweep(self):
        """Test multiple speeds to find the fastest accurate one."""
        pose = self._get_robot_pose()
        if pose is None:
            self._set_status("No robot — is engine running?", DANGER)
            return

        self._sweep_mode = True
        self._sweep_speeds = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
        self._sweep_idx = 0
        self._sweep_results = {}
        self._line_y = pose[1]
        self._direction = 1

        self._log.appendPlainText(
            f"Speed sweep: testing {len(self._sweep_speeds)} speeds "
            f"({self._sweep_speeds[0]:.2f} - {self._sweep_speeds[-1]:.2f} m/s)")

        self._start_sweep_speed()

    def _start_sweep_speed(self):
        """Begin passes at the current sweep speed."""
        speed = self._sweep_speeds[self._sweep_idx]
        self._test_speed = speed
        self._total_passes = 2  # 2 passes per speed (1 back + forth)
        self._pass_idx = 0
        self._run_results = []

        self._set_status(
            f"Sweep: testing {speed:.2f} m/s "
            f"({self._sweep_idx + 1}/{len(self._sweep_speeds)})", ACCENT)
        self._begin_pass()

    def _start_single(self):
        """Run a single pass for quick testing."""
        pose = self._get_robot_pose()
        if pose is None:
            self._set_status("No robot — is engine running?", DANGER)
            return

        self._sweep_mode = False
        self._test_speed = self._speed_spin.value()
        self._total_passes = 1
        self._pass_idx = 0
        self._direction = 1
        self._run_results = []
        self._line_y = pose[1]

        self._set_status("Single pass...", ACCENT)
        self._begin_pass()

    def _begin_pass(self):
        """Set up waypoints for one pass and start."""
        left_x = -HALF_LEN + _MARGIN
        right_x = HALF_LEN - _MARGIN
        y = self._line_y

        if self._direction == 1:
            self._start_wp = (left_x, y)
            self._end_wp = (right_x, y)
            self._ideal_heading = 0.0      # face right (+X)
        else:
            self._start_wp = (right_x, y)
            self._end_wp = (left_x, y)
            self._ideal_heading = math.pi  # face left (-X)

        # Show waypoints on field
        self._field.set_targets([self._start_wp, self._end_wp])
        self._field.set_paths([[self._start_wp, self._end_wp]])

        pct = int(self._pass_idx / max(self._total_passes, 1) * 100)
        self._progress.setValue(pct)

        if self._pass_idx == 0:
            # First pass: navigate to start position
            self._phase = "nav"
            self._tick_timer.start()
            self._set_status(
                f"Pass {self._pass_idx + 1}/{self._total_passes} — "
                f"navigating to start", ACCENT)
        else:
            # Return trip: robot is already at the end of the previous pass
            # which is the start of this pass. Skip nav, run immediately.
            pose = self._get_robot_pose()
            self._run_start_time = time.time()
            self._run_start_pose = pose
            self._phase = "running"
            self._tick_timer.start()
            self._set_status(
                f"Pass {self._pass_idx + 1}/{self._total_passes} — "
                f"running at {self._test_speed:.2f} m/s", SUCCESS)

    def _stop_test(self):
        self._tick_timer.stop()
        self._settle_timer.stop()
        self._send_stop()
        self._phase = None
        self._field.set_targets([])
        self._field.set_paths([])
        self._progress.setValue(0)
        self._set_status("Stopped", TEXT_DIM)

    def _set_status(self, text, color=TEXT_DIM):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:12px;")

    # ══════════════════════════════════════════════════════════════
    #  Tick — 20 Hz control loop
    # ══════════════════════════════════════════════════════════════

    def _tick(self):
        pose = self._get_robot_pose()
        if pose is None:
            return

        if self._phase == "nav":
            self._tick_nav(pose)
        elif self._phase == "running":
            self._tick_running(pose)

    def _tick_nav(self, pose):
        """Navigate to the start waypoint before beginning the timed run."""
        rel = world2robot(pose, self._start_wp)
        dist = math.hypot(rel[0], rel[1])
        angle = math.atan2(rel[1], rel[0])

        if dist < _ARRIVE:
            # Arrived at start — begin the timed run
            self._send_stop()
            self._run_start_time = time.time()
            self._run_start_pose = pose
            self._phase = "running"
            self._set_status(
                f"Pass {self._pass_idx + 1}/{self._total_passes} — "
                f"running at {self._test_speed:.2f} m/s",
                SUCCESS)
            return

        # Move to start using ball_nav functions
        vx, vy = move_toward(rel, _NAV_SPEED, ramp_dist=300, stop_dist=50)
        w = clamp(angle * _TURN_GAIN, -MAX_W, MAX_W)
        vx, vy = wall_brake(pose[0], pose[1], vx, vy)
        self._send_cmd(vx=vx, vy=vy, w=w)

    def _tick_running(self, pose):
        """Drive toward end waypoint while rotating, staying perfectly on-line.

        The robot can turn freely without curving off the path because:
          1. Velocity is computed in WORLD frame (toward waypoint + lateral fix)
          2. Transformed to ROBOT frame using the current heading
          3. Rotation compensation pre-rotates the robot-frame velocity to
             cancel the drift caused by rotating during the timestep
        """
        dx = self._end_wp[0] - pose[0]
        dy = self._end_wp[1] - pose[1]
        dist = math.hypot(dx, dy)

        if dist < _ARRIVE:
            self._send_stop()
            self._phase = "settling"
            self._tick_timer.stop()
            self._settle_timer.start(_SETTLE_MS)
            self._set_status(
                f"Pass {self._pass_idx + 1}/{self._total_passes} — "
                f"settling...", WARNING)
            return

        # ── 1. World-frame velocity toward target ───────────────
        speed = self._test_speed
        if dist < 400:
            speed = max(speed * (dist / 400.0), 0.08)

        ux, uy = dx / dist, dy / dist          # unit direction to target
        vx_w = ux * speed
        vy_w = uy * speed

        # ── 2. Lateral correction — stay on ideal Y line ────────
        #    Proportional push back toward the line, clamped so it
        #    never overpower the forward drive.
        y_err = pose[1] - self._line_y              # +  means above line
        lat_fix = clamp(y_err * 0.001, -0.12, 0.12)
        vy_w -= lat_fix                             # push toward line

        # ── 3. Heading controller — face direction of travel ────
        h_err = self._ideal_heading - pose[2]
        h_err = math.atan2(math.sin(h_err), math.cos(h_err))
        w = clamp(h_err * 1.5, -MAX_W, MAX_W)

        # ── 4. World → Robot frame transform ────────────────────
        #    The robot is omnidirectional so we just rotate the
        #    world-frame velocity vector by -θ (robot heading).
        theta = pose[2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vx_r =  vx_w * cos_t + vy_w * sin_t
        vy_r = -vx_w * sin_t + vy_w * cos_t

        # ── 5. Rotation compensation ────────────────────────────
        #    While the robot rotates by w*dt during this tick, the
        #    robot frame rotates with it, skewing the velocity direction.
        #    Pre-rotate by -w*dt/2 (midpoint method) to cancel this.
        dt = 0.05                                   # 20 Hz tick
        if abs(w) > 0.01:
            half = -w * dt * 0.5
            c, s = math.cos(half), math.sin(half)
            vx_r, vy_r = vx_r * c - vy_r * s, vx_r * s + vy_r * c

        # ── 6. Wall brake ───────────────────────────────────────
        vx_r, vy_r = wall_brake(pose[0], pose[1], vx_r, vy_r)

        self._send_cmd(vx=vx_r, vy=vy_r, w=w)

    def _on_settle(self):
        """Called after settle timer — measure final position and record."""
        pose = self._get_robot_pose()
        if pose is None:
            self._set_status("Lost robot pose during settle", DANGER)
            self._stop_test()
            return

        self._stop_pose = pose
        self._record_run()

        # Next pass or finish
        self._pass_idx += 1
        if self._pass_idx >= self._total_passes:
            self._finish_auto()
        else:
            self._direction *= -1   # alternate direction
            self._begin_pass()

    # ══════════════════════════════════════════════════════════════
    #  Measurement & recording
    # ══════════════════════════════════════════════════════════════

    def _record_run(self):
        """Compute metrics for the run that just finished."""
        sp = self._run_start_pose
        ep = self._stop_pose
        dt = time.time() - self._run_start_time

        if sp is None or ep is None or dt < 0.1:
            return

        # Distance traveled (world frame)
        dx = ep[0] - sp[0]
        dy = ep[1] - sp[1]
        dist_traveled = math.hypot(dx, dy)

        # Expected distance
        expected_dist = math.hypot(
            self._end_wp[0] - self._start_wp[0],
            self._end_wp[1] - self._start_wp[1])

        # Actual average speed (mm/s → m/s)
        actual_speed = (dist_traveled / dt) / 1000.0 if dt > 0 else 0
        commanded_speed = self._test_speed
        speed_ratio = actual_speed / commanded_speed if commanded_speed > 0 else 1.0

        # Lateral drift (deviation from ideal Y line)
        lateral_drift = abs(ep[1] - self._line_y)

        # Drift per meter traveled
        drift_per_m = (lateral_drift / (dist_traveled / 1000.0)
                       if dist_traveled > 100 else 0)

        # Overshoot — how far past the end waypoint
        overshoot = dist_traveled - expected_dist

        result = {
            "speed_cmd": commanded_speed,
            "speed_actual": round(actual_speed, 4),
            "speed_ratio": round(speed_ratio, 4),
            "dist_mm": round(dist_traveled, 1),
            "time_s": round(dt, 3),
            "lateral_drift_mm": round(lateral_drift, 1),
            "drift_per_m": round(drift_per_m, 2),
            "overshoot_mm": round(overshoot, 1),
            "direction": self._direction,
        }
        self._run_results.append(result)

        # Log it
        self._log.appendPlainText(
            f"Pass {self._pass_idx + 1}: "
            f"cmd={commanded_speed:.2f} actual={actual_speed:.3f} m/s "
            f"(ratio={speed_ratio:.3f}) "
            f"drift={lateral_drift:.0f}mm "
            f"overshoot={overshoot:.0f}mm")

    def _finish_auto(self):
        """All passes done — compute calibration values from results."""
        if not self._run_results:
            self._set_status("No results collected", WARNING)
            return

        # If in sweep mode, save results for this speed and move to next
        if self._sweep_mode:
            speed = self._sweep_speeds[self._sweep_idx]
            self._sweep_results[speed] = list(self._run_results)

            avg_ratio = sum(r["speed_ratio"] for r in self._run_results) / len(self._run_results)
            avg_drift = sum(r["drift_per_m"] for r in self._run_results) / len(self._run_results)
            self._log.appendPlainText(
                f"  {speed:.2f} m/s: ratio={avg_ratio:.3f} drift={avg_drift:.1f} mm/m")

            self._sweep_idx += 1
            if self._sweep_idx < len(self._sweep_speeds):
                pct = int(self._sweep_idx / len(self._sweep_speeds) * 100)
                self._progress.setValue(pct)
                self._start_sweep_speed()
                return

            # All speeds tested — pick the best
            self._finish_sweep()
            return

        # Normal (non-sweep) finish
        self._tick_timer.stop()
        self._field.set_targets([])
        self._field.set_paths([])
        self._progress.setValue(100)

        n = len(self._run_results)
        avg_ratio = sum(r["speed_ratio"] for r in self._run_results) / n
        avg_drift = sum(r["drift_per_m"] for r in self._run_results) / n
        avg_overshoot = sum(r["overshoot_mm"] for r in self._run_results) / n

        self._cal["speed_scale"] = round(avg_ratio, 4)
        self._cal["lateral_drift_per_m"] = round(avg_drift, 2)
        self._cal["stop_overshoot_mm"] = round(avg_overshoot, 1)

        if "runs" not in self._cal:
            self._cal["runs"] = []
        self._cal["runs"].extend(self._run_results)

        self._save_cal()
        self._refresh_values()

        self._log.appendPlainText(
            f"--- Calibration complete ({n} passes) ---\n"
            f"  Speed scale:    {avg_ratio:.4f}\n"
            f"  Drift per m:    {avg_drift:.2f} mm/m\n"
            f"  Stop overshoot: {avg_overshoot:.1f} mm\n"
            f"  Saved to calibration.json")

        self._set_status(
            f"Done! Speed scale = {avg_ratio:.4f}", SUCCESS)

    def _finish_sweep(self):
        """All sweep speeds tested — pick the fastest accurate speed."""
        self._tick_timer.stop()
        self._field.set_targets([])
        self._field.set_paths([])
        self._progress.setValue(100)

        # Find the fastest speed with good accuracy (ratio close to 1.0)
        # and low drift
        best_speed = None
        best_ratio = 1.0
        all_runs = []

        self._log.appendPlainText("--- Speed Sweep Results ---")
        for speed in self._sweep_speeds:
            results = self._sweep_results.get(speed, [])
            if not results:
                continue
            all_runs.extend(results)
            avg_ratio = sum(r["speed_ratio"] for r in results) / len(results)
            avg_drift = sum(r["drift_per_m"] for r in results) / len(results)

            # Acceptable if ratio is within 20% and drift is under 30 mm/m
            accurate = 0.80 < avg_ratio < 1.20 and avg_drift < 30
            marker = " <-- BEST" if accurate else ""
            self._log.appendPlainText(
                f"  {speed:.2f} m/s: ratio={avg_ratio:.3f} "
                f"drift={avg_drift:.1f} mm/m"
                f"{'  OK' if accurate else '  POOR'}{marker}")

            if accurate:
                best_speed = speed
                best_ratio = avg_ratio

        if best_speed is not None:
            self._cal["speed_scale"] = round(best_ratio, 4)
            self._cal["optimal_speed"] = best_speed
            self._speed_spin.setValue(best_speed)
            self._log.appendPlainText(
                f"\nOptimal speed: {best_speed:.2f} m/s "
                f"(scale={best_ratio:.4f})")
            self._set_status(
                f"Sweep done! Best: {best_speed:.2f} m/s", SUCCESS)
        else:
            self._log.appendPlainText("\nNo speed passed accuracy threshold")
            self._set_status("Sweep done — no accurate speed found", WARNING)

        # Compute averages from all runs
        if all_runs:
            n = len(all_runs)
            self._cal["lateral_drift_per_m"] = round(
                sum(r["drift_per_m"] for r in all_runs) / n, 2)
            self._cal["stop_overshoot_mm"] = round(
                sum(r["overshoot_mm"] for r in all_runs) / n, 1)
            if "runs" not in self._cal:
                self._cal["runs"] = []
            self._cal["runs"].extend(all_runs)

        self._save_cal()
        self._refresh_values()

    # ══════════════════════════════════════════════════════════════
    #  Apply / Reset
    # ══════════════════════════════════════════════════════════════

    def _apply_cal(self):
        """Read values from spinboxes, save to disk, and reload."""
        # Read manual edits from spinboxes
        self._cal['speed_scale'] = round(self._val_speed_spin.value(), 4)
        self._cal['lateral_drift_per_m'] = round(self._val_drift_spin.value(), 2)
        self._cal['stop_overshoot_mm'] = round(self._val_stop_spin.value(), 1)

        self._save_cal()

        # Reload into ball_nav if possible
        try:
            from TeamControl.robot import ball_nav
            ball_nav._reload_calibration()
        except Exception:
            pass

        self._set_status("Calibration saved and applied", SUCCESS)
        self._log.appendPlainText(
            f"Applied: speed_scale={self._cal['speed_scale']:.4f}  "
            f"drift={self._cal['lateral_drift_per_m']:.2f} mm/m")

    def _reset_cal(self):
        """Reset calibration to defaults (no correction)."""
        self._cal = self._default_cal()
        self._save_cal()
        self._refresh_values()

        try:
            from TeamControl.robot import ball_nav
            ball_nav._reload_calibration()
        except Exception:
            pass

        self._set_status("Calibration reset to defaults", WARNING)
        self._log.appendPlainText("Calibration reset to defaults (scale=1.0)")
