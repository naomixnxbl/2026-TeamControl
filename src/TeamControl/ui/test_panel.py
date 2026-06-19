"""
Hardware Test Console — direct robot connectivity testing & manual control.

Tabbed layout:
  Connect & Drive  — pick a robot, set exact velocities, send commands
  Quick Tests      — one-click preset movements
  Robots           — full ipconfig.yaml table
  Packet Log       — raw packet inspector + scrolling log
"""

import time
import math
import socket

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QSpinBox,
    QDoubleSpinBox, QSlider, QPlainTextEdit,
    QTabWidget, QFrame, QSizePolicy, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, QTime
from PySide6.QtGui import QFont

from TeamControl.ui.theme import (
    ACCENT, TEXT_DIM, SUCCESS, DANGER, WARNING,
    BG_DARK, BORDER,
)

from TeamControl.network.robot_command import RobotCommand
from TeamControl.network.sender import Sender
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.grSimPacketFactory import grSimPacketFactory
from TeamControl.utils.yaml_config import Config
from TeamControl.robot.constants import (
    MAX_W, MANUAL_MAX_SPEED, MANUAL_MAX_W,
    FIELD_MARGIN,
)
from TeamControl.world.transform_cords import world2robot
from TeamControl.robot.ball_nav import clamp, move_toward, sanitize_field_target

def _ts():
    return QTime.currentTime().toString("HH:mm:ss.zzz")


class _LogView(QPlainTextEdit):
    MAX_LINES = 2000

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setFont(QFont("Cascadia Code", 10))

    def ok(self, msg):
        self.appendHtml(
            f'<span style="color:{TEXT_DIM}">{_ts()}</span> '
            f'<span style="color:{SUCCESS}">{msg}</span>')

    def err(self, msg):
        self.appendHtml(
            f'<span style="color:{TEXT_DIM}">{_ts()}</span> '
            f'<span style="color:{DANGER}">{msg}</span>')

    def info(self, msg):
        self.appendHtml(
            f'<span style="color:{TEXT_DIM}">{_ts()}</span> '
            f'<span style="color:#eaeaea">{msg}</span>')


class _RobotRow:
    __slots__ = ("label", "team", "shell_id", "grsim_id", "ip", "port")

    def __init__(self, label, team, shell_id, grsim_id, ip, port):
        self.label = label
        self.team = team
        self.shell_id = shell_id
        self.grsim_id = grsim_id
        self.ip = ip
        self.port = port


def _heading(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-size:13px; font-weight:bold; color:{ACCENT}; padding:2px 0;")
    return lbl


def _sep():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color:{BORDER};")
    return line


class TestPanel(QWidget):
    """Hardware testing & manual robot control console."""

    def __init__(self, engine=None, field=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._field = field
        self._sender = Sender(device_ip="192.168.1.2")
        self._continuous_timer = QTimer(self)
        self._continuous_timer.setInterval(50)
        self._continuous_timer.timeout.connect(self._send_continuous)
        self._robots: list[_RobotRow] = []
        self._n_sent = 0
        self._n_err = 0

        # Action test state
        self._action_timer = QTimer(self)
        self._action_timer.setInterval(50)  # 20 Hz
        self._action_timer.timeout.connect(self._action_tick)
        self._action_mode = None  # "go_to_ball", "go_to_ball_kick", "draw_square", "go_to_point"
        self._last_ball_dist = None  # track last known ball distance for kick-on-occlude
        self._square_step = 0
        self._square_step_ticks = 0
        self._goto_target = None  # (x_mm, y_mm) for go_to_point
        self._our_id_spin = None  # set by MainWindow — toolbar "Our Bot" spinner
        self._dashboard_rid = None   # set by dashboard robot selector (overrides toolbar)
        self._dashboard_yellow = True

        self._build_ui()
        self._load_robots()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_connection_status)
        self._status_timer.start()

    # ══════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_drive_tab(), "Connect && Drive")
        tabs.addTab(self._build_log_tab(), "Packet Log")
        root.addWidget(tabs)

    # ── Tab 1: Connect & Drive ────────────────────────────────────

    def _build_drive_tab(self):
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(16)

        # ── Left column: target ───────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(_heading("Robot Target"))

        tg = QGridLayout()
        tg.setSpacing(8)
        tg.setColumnStretch(1, 1)

        tg.addWidget(QLabel("Preset:"), 0, 0)
        self._robot_combo = QComboBox()
        self._robot_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._robot_combo.currentIndexChanged.connect(self._on_robot_selected)
        tg.addWidget(self._robot_combo, 0, 1, 1, 3)

        tg.addWidget(QLabel("IP Address:"), 1, 0)
        self._ip_edit = QLineEdit("127.0.0.1")
        tg.addWidget(self._ip_edit, 1, 1, 1, 3)

        tg.addWidget(QLabel("Port:"), 2, 0)
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1025, 65534)
        self._port_spin.setValue(50514)
        tg.addWidget(self._port_spin, 2, 1)

        tg.addWidget(QLabel("Shell ID:"), 3, 0)
        self._id_spin = QSpinBox()
        self._id_spin.setRange(0, 15)
        tg.addWidget(self._id_spin, 3, 1)

        tg.addWidget(QLabel("Team:"), 4, 0)
        self._team_combo = QComboBox()
        self._team_combo.addItems(["Yellow", "Blue"])
        tg.addWidget(self._team_combo, 4, 1)

        left.addLayout(tg)

        ping_btn = QPushButton("Test Connection")
        ping_btn.setMinimumHeight(36)
        ping_btn.setStyleSheet(f"font-weight:bold; color:{ACCENT};")
        ping_btn.clicked.connect(self._test_connection)
        left.addWidget(ping_btn)

        grsim_test_btn = QPushButton("Test grSim Sender")
        grsim_test_btn.setMinimumHeight(36)
        grsim_test_btn.setToolTip("Send a direct grSim packet and report dispatcher grSim status")
        grsim_test_btn.clicked.connect(self._test_grsim_sender)
        left.addWidget(grsim_test_btn)

        left.addWidget(_sep())
        left.addWidget(_heading("Connection Status"))
        self._hw_status = QLabel("Hardware: —")
        self._hw_status.setWordWrap(True)
        self._hw_status.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:12px; padding:2px 4px;")
        left.addWidget(self._hw_status)
        self._grsim_status_check = QCheckBox("Send to Simulation (grSim)")
        self._grsim_status_check.setEnabled(False)
        self._grsim_status_check.setToolTip(
            "Read-only: ON when the engine is configured to also send commands to grSim")
        left.addWidget(self._grsim_status_check)
        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(320)
        outer.addWidget(left_w)

        # Vertical divider
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet(f"color:{BORDER};")
        outer.addWidget(div)

        # ── Right column: velocity + send ─────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        right.addWidget(_heading("Velocity Command"))

        vel_grid = QGridLayout()
        vel_grid.setSpacing(10)
        vel_grid.setColumnStretch(1, 1)

        self._vx_spin, self._vx_slider = self._make_vel_row(
            "VX  (tangent)", -MANUAL_MAX_SPEED, MANUAL_MAX_SPEED, 0.0, "m/s", vel_grid, 0)
        self._vy_spin, self._vy_slider = self._make_vel_row(
            "VY  (normal)", -MANUAL_MAX_SPEED, MANUAL_MAX_SPEED, 0.0, "m/s", vel_grid, 1)
        self._w_spin, self._w_slider = self._make_vel_row(
            "W   (angular)", -MANUAL_MAX_W, MANUAL_MAX_W, 0.0, "rad/s", vel_grid, 2)

        right.addLayout(vel_grid)

        right.addWidget(_sep())
        right.addWidget(_heading("Kick & Dribble"))

        kd_grid = QGridLayout()
        kd_grid.setSpacing(10)

        self._kick_btn = QPushButton("Kick")
        self._kick_btn.setStyleSheet("font-size:14px; font-weight:bold;")
        self._kick_btn.setCheckable(True)
        self._dribble_btn = QPushButton("Dribble / Spinner")
        self._dribble_btn.setStyleSheet("font-size:14px; font-weight:bold;")
        self._dribble_btn.setCheckable(True)

        kd_grid.addWidget(self._kick_btn, 0, 0)
        kd_grid.addWidget(self._dribble_btn, 0, 1)

        self._kick_speed_spin = QDoubleSpinBox()
        self._kick_speed_spin.setRange(0, 20)
        self._kick_speed_spin.setValue(10.0)
        self._kick_speed_spin.setSingleStep(0.5)
        self._kick_speed_spin.setSuffix(" m/s")
        self._kick_speed_spin.setPrefix("Kick speed: ")
        kd_grid.addWidget(self._kick_speed_spin, 1, 0, 1, 2)

        right.addLayout(kd_grid)

        right.addWidget(_sep())
        right.addWidget(_heading("Send"))

        send_grid = QGridLayout()
        send_grid.setSpacing(8)

        send_once = QPushButton("Send Once")
        send_once.setObjectName("startBtn")
        send_once.setMinimumHeight(44)
        send_once.setStyleSheet("font-size:14px;")
        send_once.clicked.connect(self._send_once)
        send_grid.addWidget(send_once, 0, 0)

        self._cont_btn = QPushButton("Start Continuous (20 Hz)")
        self._cont_btn.setMinimumHeight(44)
        self._cont_btn.setStyleSheet("font-size:14px;")
        self._cont_btn.clicked.connect(self._toggle_continuous)
        send_grid.addWidget(self._cont_btn, 0, 1)

        stop_btn = QPushButton("STOP")
        stop_btn.setObjectName("stopBtn")
        stop_btn.setMinimumHeight(44)
        stop_btn.setStyleSheet("font-size:14px;")
        stop_btn.clicked.connect(self._send_stop)
        send_grid.addWidget(stop_btn, 1, 0)

        zero_btn = QPushButton("Zero All Inputs")
        zero_btn.setMinimumHeight(44)
        zero_btn.setStyleSheet("font-size:14px;")
        zero_btn.clicked.connect(self._zero_inputs)
        send_grid.addWidget(zero_btn, 1, 1)

        right.addLayout(send_grid)

        # Raw packet preview (inline)
        right.addWidget(_sep())
        right.addWidget(_heading("Last Packet"))
        self._raw_label = QLabel("No packets sent yet")
        self._raw_label.setFont(QFont("Cascadia Code", 11))
        self._raw_label.setWordWrap(True)
        self._raw_label.setMinimumHeight(60)
        self._raw_label.setStyleSheet(
            f"background:{BG_DARK}; padding:10px; border:1px solid {BORDER}; "
            f"border-radius:6px; color:{ACCENT};")
        self._raw_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self._raw_label)

        # ── Actions (right-click on field triggers these) ────────────
        right.addWidget(_sep())
        right.addWidget(_heading("Actions"))

        act_btn_grid = QGridLayout()
        act_btn_grid.setSpacing(6)

        self._goto_ball_btn = QPushButton("Go to Ball")
        self._goto_ball_btn.setMinimumHeight(36)
        self._goto_ball_btn.setToolTip(
            "Navigate robot toward the ball (requires engine running + vision)")
        self._goto_ball_btn.clicked.connect(lambda: self._start_action("go_to_ball"))
        act_btn_grid.addWidget(self._goto_ball_btn, 0, 0)

        self._goto_ball_kick_btn = QPushButton("Go & Kick")
        self._goto_ball_kick_btn.setMinimumHeight(36)
        self._goto_ball_kick_btn.setToolTip("Navigate to ball and fire the kicker on contact")
        self._goto_ball_kick_btn.clicked.connect(
            lambda: self._start_action("go_to_ball_kick"))
        act_btn_grid.addWidget(self._goto_ball_kick_btn, 0, 1)

        self._goto_point_btn = QPushButton("Go to Point…")
        self._goto_point_btn.setMinimumHeight(36)
        self._goto_point_btn.setToolTip(
            "Click a point on the field canvas to set the navigation target")
        self._goto_point_btn.clicked.connect(self._pick_goto_point)
        act_btn_grid.addWidget(self._goto_point_btn, 1, 0)

        self._draw_square_btn = QPushButton("Draw Square")
        self._draw_square_btn.setMinimumHeight(36)
        self._draw_square_btn.setToolTip("Navigate in a 500 mm square pattern")
        self._draw_square_btn.clicked.connect(lambda: self._start_action("draw_square"))
        act_btn_grid.addWidget(self._draw_square_btn, 1, 1)

        right.addLayout(act_btn_grid)

        act_grid = QGridLayout()
        act_grid.setSpacing(8)

        act_grid.addWidget(QLabel("Go-to velocity:"), 0, 0)
        self._goto_vel_spin = QDoubleSpinBox()
        self._goto_vel_spin.setRange(0.05, MANUAL_MAX_SPEED)
        self._goto_vel_spin.setValue(0.2)
        self._goto_vel_spin.setSingleStep(0.05)
        self._goto_vel_spin.setSuffix("  m/s")
        self._goto_vel_spin.setMinimumWidth(120)
        act_grid.addWidget(self._goto_vel_spin, 0, 1)

        right.addLayout(act_grid)

        self._action_status = QLabel("")
        self._action_status.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px; padding:4px;")
        right.addWidget(self._action_status)

        self._goto_status = QLabel("")
        self._goto_status.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px; padding:4px;")
        right.addWidget(self._goto_status)

        stop_row = QHBoxLayout()
        stop_row.setSpacing(8)

        stop_btn = QPushButton("STOP")
        stop_btn.setObjectName("stopBtn")
        stop_btn.setMinimumHeight(40)
        stop_btn.setStyleSheet("font-size:14px;")
        stop_btn.clicked.connect(self._send_stop)
        stop_row.addWidget(stop_btn)

        stop_all = QPushButton("STOP ALL")
        stop_all.setObjectName("stopBtn")
        stop_all.setMinimumHeight(40)
        stop_all.setStyleSheet("font-size:14px;")
        stop_all.clicked.connect(self._stop_all)
        stop_row.addWidget(stop_all)

        right.addLayout(stop_row)

        right.addStretch()
        outer.addLayout(right, 1)

        return page

    # ── Packet Log ─────────────────────────────────────────────────

    def _build_log_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        lay.addWidget(_heading("Packet Inspector"))

        bar = QHBoxLayout()
        self._send_count = QLabel("0 sent")
        self._send_count.setStyleSheet(f"color:{TEXT_DIM}; font-size:13px;")
        self._err_count = QLabel("0 errors")
        self._err_count.setStyleSheet(f"color:{DANGER}; font-size:13px;")
        clear_btn = QPushButton("Clear Log")
        clear_btn.setFixedWidth(80)
        clear_btn.clicked.connect(lambda: (self._log.clear(),
                                           self._reset_counts()))
        bar.addWidget(self._send_count)
        bar.addWidget(QLabel("  "))
        bar.addWidget(self._err_count)
        bar.addStretch()
        bar.addWidget(clear_btn)
        lay.addLayout(bar)

        self._log = _LogView()
        lay.addWidget(self._log)

        return page

    # ── Velocity row factory ──────────────────────────────────────

    def _make_vel_row(self, label, lo, hi, default, suffix, grid, row):
        lbl = QLabel(label)
        lbl.setStyleSheet("font-weight:bold; font-size:13px;")
        lbl.setMinimumWidth(110)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(lo * 100), int(hi * 100))
        slider.setValue(int(default * 100))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int((hi - lo) * 10))
        slider.setMinimumWidth(200)

        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(default)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setSuffix(f"  {suffix}")
        spin.setMinimumWidth(130)
        spin.setMinimumHeight(30)
        spin.setStyleSheet("font-size:13px;")

        reset = QPushButton("0")
        reset.setFixedSize(30, 30)
        reset.setToolTip("Reset to zero")

        slider.valueChanged.connect(lambda v: spin.setValue(v / 100.0))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * 100)))
        reset.clicked.connect(lambda: (spin.setValue(0), slider.setValue(0)))

        grid.addWidget(lbl, row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(spin, row, 2)
        grid.addWidget(reset, row, 3)

        return spin, slider

    # ── Robot loading ─────────────────────────────────────────────

    def _load_robots(self):
        self._robots.clear()
        self._robot_combo.clear()
        try:
            cfg = Config()
        except Exception as e:
            self._log.err(f"Failed to load config: {e}")
            return

        for team_name, team_data in [("Yellow", cfg.yellow), ("Blue", cfg.blue)]:
            if not team_data:
                continue
            for key, rd in team_data.items():
                r = _RobotRow(
                    label=f"{team_name} {key}",
                    team=team_name,
                    shell_id=rd.get("shellID", 0),
                    grsim_id=rd.get("grSimID", 0),
                    ip=rd.get("ip", "127.0.0.1"),
                    port=rd.get("port", 50514),
                )
                self._robots.append(r)
                self._robot_combo.addItem(
                    f"{r.label}  —  shell {r.shell_id}  →  {r.ip}:{r.port}")

        self._robot_combo.addItem("— Custom —")

        self._log.info(f"Loaded {len(self._robots)} robots from ipconfig.yaml")

    def _on_robot_selected(self, idx):
        if 0 <= idx < len(self._robots):
            r = self._robots[idx]
            self._ip_edit.setText(r.ip)
            self._port_spin.setValue(r.port)
            self._id_spin.setValue(r.shell_id)
            self._team_combo.setCurrentText(r.team)

    # ── Command building ──────────────────────────────────────────

    def set_our_bot_spin(self, spin):
        """Give the test panel access to the toolbar's Our Bot spinner."""
        self._our_id_spin = spin

    def select_robot(self, is_yellow: bool, robot_id: int):
        """Override target robot for action commands (called from dashboard/field click)."""
        self._dashboard_rid = robot_id
        self._dashboard_yellow = bool(is_yellow)
        self._id_spin.setValue(robot_id)
        self._team_combo.setCurrentText("Yellow" if is_yellow else "Blue")

    def start_action(self, action_name: str):
        """Trigger a named action (called from dashboard)."""
        if action_name == "stop":
            self._stop_action()
        else:
            self._start_action(action_name)

    def _get_action_rid_yellow(self):
        """Get robot ID and team for action commands.

        Priority: dashboard selection > toolbar spinner > id_spin/team_combo.
        """
        if self._dashboard_rid is not None:
            return self._dashboard_rid, self._dashboard_yellow
        if self._our_id_spin is not None:
            rid = self._our_id_spin.value()
        else:
            rid = self._id_spin.value()
        cfg = self._engine.config if self._engine else None
        if cfg is not None:
            is_yellow = cfg.us_yellow
        else:
            is_yellow = (self._team_combo.currentText() == "Yellow")
        return rid, is_yellow

    def _build_action_cmd(self, vx=0.0, vy=0.0, w=0.0, kick=0, dribble=0):
        """Build a RobotCommand using the toolbar's robot ID and engine team."""
        rid, is_yellow = self._get_action_rid_yellow()
        return RobotCommand(
            robot_id=rid, vx=vx, vy=vy, w=w,
            kick=kick, dribble=dribble, isYellow=is_yellow)

    def _send_action(self, cmd: RobotCommand):
        """Send a field action command through the unified send path."""
        self._do_send(cmd)

    def _build_cmd(self, vx=None, vy=None, w=None, kick=None, dribble=None):
        return RobotCommand(
            robot_id=self._id_spin.value(),
            vx=vx if vx is not None else self._vx_spin.value(),
            vy=vy if vy is not None else self._vy_spin.value(),
            w=w if w is not None else self._w_spin.value(),
            kick=kick if kick is not None else int(self._kick_btn.isChecked()),
            dribble=dribble if dribble is not None else int(self._dribble_btn.isChecked()),
            isYellow=(self._team_combo.currentText() == "Yellow"),
        )

    # ── Sending ───────────────────────────────────────────────────

    def _do_send(self, cmd: RobotCommand):
        """Send a command. Routes through dispatcher (hw + grSim) when engine is running."""
        raw_text = str(cmd)

        if self._engine and self._engine.is_running:
            self._engine.send_robot_command(cmd, runtime=0.20)
            self._raw_label.setText(
                f"<b>Via:</b> Dispatcher<br>"
                f"<b>Raw:</b> {raw_text}<br>"
                f"<b>Bytes:</b> {cmd.encode()!r}")
            self._log.ok(
                f"→ dispatcher  |  "
                f"id={cmd.robot_id} vx={cmd.vx:.2f} vy={cmd.vy:.2f} "
                f"w={cmd.w:.2f} kick={cmd.kick} drib={cmd.dribble}")
            self._n_sent += 1
            self._update_hw_status("Hardware: via dispatcher", SUCCESS)
            self._update_counts()
            return

        # Direct UDP — engine is not running, no dispatcher available
        ip = self._ip_edit.text().strip()
        port = self._port_spin.value()
        self._raw_label.setText(
            f"<b>To:</b> {ip}:{port}<br>"
            f"<b>Raw:</b> {raw_text}<br>"
            f"<b>Bytes:</b> {cmd.encode()!r}")
        try:
            self._sender.send(cmd, ip, port)
            self._log.ok(
                f"→ {ip}:{port}  |  "
                f"id={cmd.robot_id} vx={cmd.vx:.2f} vy={cmd.vy:.2f} "
                f"w={cmd.w:.2f} kick={cmd.kick} drib={cmd.dribble}")
            self._n_sent += 1
            self._update_hw_status(f"Hardware: OK → {ip}:{port}", SUCCESS)
        except Exception as e:
            self._log.err(f"SEND FAILED to {ip}:{port} — {e}")
            self._n_err += 1
            self._update_hw_status(f"Hardware: FAILED — {e}", DANGER)
        self._update_counts()
    def _send_once(self):
        self._do_send(self._build_cmd())

    def _send_stop(self):
        self._stop_action()
        cmd = self._build_cmd(vx=0, vy=0, w=0, kick=0, dribble=0)
        self._do_send(cmd)
        self._zero_inputs()
        self._log.info("STOP sent")

    def _zero_inputs(self):
        self._vx_spin.setValue(0)
        self._vy_spin.setValue(0)
        self._w_spin.setValue(0)
        self._kick_btn.setChecked(False)
        self._dribble_btn.setChecked(False)

    def _send_test(self, vx, vy, w, kick, dribble):
        cmd = self._build_cmd(vx=vx, vy=vy, w=w, kick=kick, dribble=dribble)
        self._do_send(cmd)

    def _toggle_continuous(self):
        if self._continuous_timer.isActive():
            self._continuous_timer.stop()
            self._cont_btn.setText("Start Continuous (20 Hz)")
            self._cont_btn.setStyleSheet("font-size:14px;")
            self._log.info("Continuous send stopped")
        else:
            self._continuous_timer.start()
            self._cont_btn.setText("STOP Continuous")
            self._cont_btn.setStyleSheet(
                f"background:{DANGER}; color:#fff; font-weight:bold; font-size:14px;")
            self._log.info("Continuous send started at 20 Hz")

    def _send_continuous(self):
        self._do_send(self._build_cmd())

    def _stop_all(self):
        if self._continuous_timer.isActive():
            self._continuous_timer.stop()
            self._cont_btn.setText("Start Continuous (20 Hz)")
            self._cont_btn.setStyleSheet("font-size:14px;")

        for r in self._robots:
            cmd = RobotCommand(
                robot_id=r.shell_id, vx=0, vy=0, w=0,
                kick=0, dribble=0,
                isYellow=(r.team == "Yellow"))
            if self._engine and self._engine.is_running:
                self._engine.send_robot_command(cmd, runtime=0.05)
                self._log.ok(f"STOP → {r.label} (via dispatcher)")
                self._n_sent += 1
            else:
                try:
                    self._sender.send(cmd, r.ip, r.port)
                    self._log.ok(f"STOP → {r.label} ({r.ip}:{r.port})")
                    self._n_sent += 1
                except Exception as e:
                    self._log.err(f"STOP FAILED → {r.label} — {e}")
                    self._n_err += 1

        self._update_counts()
        self._log.info(f"STOP ALL sent to {len(self._robots)} robots")

    # ── Action tests ────────────────────────────────────────────

    def _start_action(self, mode):
        self._stop_action()
        self._set_field_manual_override(True)
        self._action_mode = mode
        self._square_step = 0
        self._square_step_ticks = 0
        self._last_ball_dist = None
        self._action_timer.start()

        labels = {
            "go_to_ball": "Go to Ball",
            "go_to_ball_kick": "Go to Ball & Kick",
            "draw_square": "Draw Square",
        }
        self._action_status.setStyleSheet(
            f"color:{SUCCESS}; font-size:12px; padding:4px;")
        self._action_status.setText(
            f"Running: {labels.get(mode, mode)} — click STOP to cancel")
        self._log.info(f"Action test started: {labels.get(mode, mode)}")
        if self._engine and self._engine.current_mode not in (
            "vision_only",
            "voronoi_test",
        ):
            self._log.info(
                "(Dispatcher paused for this bot so AI does not override field commands.)")

    def _set_field_manual_override(self, enabled: bool):
        """Pause AI dispatcher for our bot so field / quick-test commands work."""
        if not self._engine or not self._engine.is_running:
            return
        rid, iy = self._get_action_rid_yellow()
        self._engine.set_field_manual_control(rid, iy, enabled)

    def _stop_action(self):
        if self._action_timer.isActive():
            self._action_timer.stop()
            self._action_mode = None
            self._action_status.setText("")
            self._set_field_manual_override(False)
            # Send a stop command
            cmd = self._build_action_cmd()
            self._send_action(cmd)

    def _get_ball_and_robot(self):
        """Get ball and robot positions from the engine's world model."""
        if not self._engine:
            return None, None
        wm = self._engine._wm
        if wm is None:
            return None, None
        frame = wm.get_latest_frame()
        if frame is None:
            return None, None

        ball = frame.ball
        if ball is None:
            return None, None
        ball_pos = (float(ball.x), float(ball.y))

        rid, is_yellow = self._get_action_rid_yellow()
        team = frame.robots_yellow if is_yellow else frame.robots_blue
        try:
            robot = team[rid]
        except (IndexError, TypeError):
            return ball_pos, None

        from TeamControl.SSL.vision.robots import Robot
        if not isinstance(robot, Robot):
            return ball_pos, None

        robot_pose = (float(robot.x), float(robot.y), float(robot.o))
        return ball_pos, robot_pose

    def field_action(self, action_name):
        """Called from the field right-click menu."""
        if action_name == "stop":
            # Stop any running action and release manual override
            self._stop_action()
        else:
            self._start_action(action_name)

    def _pick_goto_point(self):
        if self._field is None:
            self._goto_status.setStyleSheet(
                f"color:{DANGER}; font-size:12px; padding:4px;")
            self._goto_status.setText("No field canvas available")
            return
        self._field.set_place_mode("go_to_point")
        self._goto_status.setStyleSheet(
            f"color:{WARNING}; font-size:12px; padding:4px;")
        self._goto_status.setText("Click a point on the Dashboard field view...")

    def go_to_point(self, x_mm, y_mm):
        """Called when the user clicks on the field after picking go-to-point."""
        self._goto_target = sanitize_field_target(
            (x_mm, y_mm),
            margin=FIELD_MARGIN,
        )
        x_mm, y_mm = self._goto_target
        self._stop_action()
        self._set_field_manual_override(True)
        self._action_mode = "go_to_point"
        self._action_timer.start()
        self._goto_status.setStyleSheet(
            f"color:{SUCCESS}; font-size:12px; padding:4px;")
        self._goto_status.setText(
            f"Going to ({x_mm:.0f}, {y_mm:.0f}) at "
            f"{self._goto_vel_spin.value():.2f} m/s — click STOP to cancel")
        self._action_status.setStyleSheet(
            f"color:{SUCCESS}; font-size:12px; padding:4px;")
        self._action_status.setText("Running: Go to Point — click STOP to cancel")
        self._log.info(f"Go to point ({x_mm:.0f}, {y_mm:.0f})")
        if self._engine and self._engine.current_mode not in (
            "vision_only",
            "voronoi_test",
        ):
            self._log.info(
                "(Dispatcher paused for this bot so AI does not override.)")


    def _get_robot_pose(self):
        """Get just the robot pose (no ball needed)."""
        if not self._engine:
            return None
        wm = self._engine._wm
        if wm is None:
            return None
        frame = wm.get_latest_frame()
        if frame is None:
            return None
        rid, is_yellow = self._get_action_rid_yellow()
        team = frame.robots_yellow if is_yellow else frame.robots_blue
        try:
            robot = team[rid]
        except (IndexError, TypeError):
            return None
        from TeamControl.SSL.vision.robots import Robot
        if not isinstance(robot, Robot):
            return None
        return (float(robot.x), float(robot.y), float(robot.o))

    def _action_tick(self):
        if self._action_mode == "draw_square":
            self._tick_draw_square()
            return

        if self._action_mode == "go_to_point":
            self._tick_go_to_point()
            return

        # go_to_ball or go_to_ball_kick
        ball_pos, robot_pose = self._get_ball_and_robot()
        if robot_pose is None:
            return

        # Ball disappeared — if we were close, assume we're on it
        if ball_pos is None:
            if (self._action_mode == "go_to_ball_kick"
                    and self._last_ball_dist is not None
                    and self._last_ball_dist < 200):
                cmd = self._build_action_cmd(vx=0.3, kick=1)
                self._send_action(cmd)
            return

        rel_ball = world2robot(robot_pose, sanitize_field_target(ball_pos))
        dist = math.hypot(rel_ball[0], rel_ball[1])
        angle = math.atan2(rel_ball[1], rel_ball[0])
        self._last_ball_dist = dist

        kick = 0
        dribble = 0

        # Behaviour tuning for go_to_ball vs go_to_ball_kick
        if self._action_mode == "go_to_ball_kick":
            far_dist = 600.0   # mm — cruise
            slow_dist = 250.0  # mm — start slowing + dribbling
            kick_dist = 130.0  # mm — close enough to fire kicker

            if dist > far_dist:
                # Far: move faster to get in the area
                vx, vy = move_toward(rel_ball, 0.6, ramp_dist=800, stop_dist=slow_dist)
                w = clamp(angle * 0.6, -MAX_W, MAX_W)
            elif dist > kick_dist:
                # Mid: slow down and keep dribbler on while aligning
                dribble = 1
                vx, vy = move_toward(rel_ball, 0.25, ramp_dist=400, stop_dist=kick_dist)
                w = clamp(angle * 0.7, -MAX_W, MAX_W)
            else:
                # Very close: line up and fire the real kicker
                dribble = 1
                vx, vy = move_toward(rel_ball, 0.12, ramp_dist=150, stop_dist=40)
                w = clamp(angle * 0.8, -MAX_W, MAX_W)
                if abs(angle) < 0.18:
                    # Small straight push then kick
                    vx = 0.25
                    vy = 0.0
                    kick = 1
                    dribble = 0
        else:
            # Plain go_to_ball: just move toward ball and stop near it
            vx, vy = move_toward(rel_ball, 0.45, ramp_dist=500, stop_dist=80)
            w = clamp(angle * 0.5, -MAX_W, MAX_W)

        cmd = self._build_action_cmd(vx=vx, vy=vy, w=w, kick=kick, dribble=dribble)
        self._send_action(cmd)

    def _tick_go_to_point(self):
        if self._goto_target is None:
            self._stop_action()
            return

        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        rel = world2robot(robot_pose, self._goto_target)
        dist = math.hypot(rel[0], rel[1])
        angle = math.atan2(rel[1], rel[0])

        if dist < 80:
            # Arrived — send explicit stop
            cmd = self._build_action_cmd()
            self._send_action(cmd)
            self._stop_action()
            self._goto_status.setStyleSheet(
                f"color:{SUCCESS}; font-size:12px; padding:4px;")
            self._goto_status.setText("Arrived at target point")
            return

        max_speed = self._goto_vel_spin.value()

        # Omnidirectional movement toward target
        vx, vy = move_toward(rel, max_speed, ramp_dist=400, stop_dist=80)
        w = clamp(angle * 0.5, -MAX_W, MAX_W)

        cmd = self._build_action_cmd(vx=vx, vy=vy, w=w)
        self._send_action(cmd)

    # Square waypoints — small square near center of field (500mm sides)
    _SQUARE_WAYPOINTS = [
        ( 250,  250),   # front-right
        ( 250, -250),   # back-right
        (-250, -250),   # back-left
        (-250,  250),   # front-left
    ]
    _SQUARE_ARRIVE_DIST = 150  # mm — close enough to move to next waypoint

    def _tick_draw_square(self):
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        if self._square_step >= len(self._SQUARE_WAYPOINTS):
            cmd = self._build_action_cmd()
            self._send_action(cmd)
            self._stop_action()
            self._action_status.setStyleSheet(
                f"color:{SUCCESS}; font-size:12px; padding:4px;")
            self._action_status.setText("Draw Square completed")
            return

        # Navigate to current waypoint
        target = sanitize_field_target(self._SQUARE_WAYPOINTS[self._square_step])
        rel = world2robot(robot_pose, target)
        dist = math.hypot(rel[0], rel[1])
        angle = math.atan2(rel[1], rel[0])

        if dist < self._SQUARE_ARRIVE_DIST:
            # Stop at waypoint, then move to next
            cmd = self._build_action_cmd()
            self._send_action(cmd)
            self._square_step += 1
            self._action_status.setText(
                f"Draw Square — waypoint {self._square_step}/{len(self._SQUARE_WAYPOINTS)}")
            return

        # Omnidirectional movement toward waypoint
        vx, vy = move_toward(rel, 0.4, ramp_dist=400, stop_dist=80)
        w = clamp(angle * 0.5, -MAX_W, MAX_W)

        cmd = self._build_action_cmd(vx=vx, vy=vy, w=w)
        self._send_action(cmd)

    def _update_hw_status(self, msg: str, color=None):
        if not hasattr(self, "_hw_status") or self._hw_status is None:
            return
        c = color if color else TEXT_DIM
        self._hw_status.setText(msg)
        self._hw_status.setStyleSheet(f"color:{c}; font-size:12px; padding:2px 4px;")

    def _refresh_connection_status(self):
        if not hasattr(self, "_grsim_status_check") or self._grsim_status_check is None:
            return
        if self._engine and self._engine.is_running:
            opts = getattr(self._engine, "_channel_options", {})
            grsim_on = bool(opts.get("send_grsim", False))
        else:
            grsim_on = False
        self._grsim_status_check.setChecked(grsim_on)

    def _test_connection(self):
        ip = self._ip_edit.text().strip()
        port = self._port_spin.value()
        rid = self._id_spin.value()
        self._log.info(f"Testing connection to {ip}:{port} (robot {rid})…")

        cmd = RobotCommand(robot_id=rid, vx=0, vy=0, w=0, kick=0, dribble=0,
                           isYellow=(self._team_combo.currentText() == "Yellow"))
        try:
            self._sender.send(cmd, ip, port)
            self._log.ok(
                f"Packet sent to {ip}:{port} — "
                f"UDP is fire-and-forget so no receipt confirmation")
            self._n_sent += 1
            self._update_hw_status(f"Hardware: Sent to {ip}:{port}", SUCCESS)
        except socket.error as e:
            self._log.err(f"SOCKET ERROR: {e}")
            self._n_err += 1
            self._update_hw_status(f"Hardware: Socket error — {e}", DANGER)
        except Exception as e:
            self._log.err(f"ERROR: {type(e).__name__}: {e}")
            self._n_err += 1
            self._update_hw_status(f"Hardware: Error — {type(e).__name__}", DANGER)
        self._update_counts()

    def _test_grsim_sender(self):
        try:
            cfg = Config()
            ip, port = cfg.grSim_addr
        except Exception as e:
            self._log.err(f"grSim config load failed: {type(e).__name__}: {e}")
            self._n_err += 1
            self._update_counts()
            return

        idx = self._robot_combo.currentIndex()
        grsim_id = self._id_spin.value()
        if 0 <= idx < len(self._robots):
            grsim_id = self._robots[idx].grsim_id
        is_yellow = self._team_combo.currentText() == "Yellow"
        cmd = RobotCommand(
            robot_id=self._id_spin.value(),
            vx=0.0,
            vy=0.0,
            w=1.0,
            kick=0,
            dribble=0,
            isYellow=is_yellow,
        )
        packet = grSimPacketFactory.robot_command(
            robot_id=grsim_id,
            vx=cmd.vx,
            vy=cmd.vy,
            w=cmd.w,
            kick=cmd.kick,
            dribble=cmd.dribble,
            isYellow=cmd.isYellow,
        )
        data = packet.SerializeToString()
        self._log.info(
            f"Testing grSim sender: config={ip}:{port}, shell={cmd.robot_id}, "
            f"grSimID={grsim_id}, team={'yellow' if is_yellow else 'blue'}, bytes={len(data)}")

        if self._engine and self._engine.is_running:
            opts = getattr(self._engine, "_channel_options", {})
            self._log.info(
                f"Dispatcher grSim option: send_grsim={bool(opts.get('send_grsim', False))}")

        try:
            sender = grSimSender(ip=ip, port=port)
            for _ in range(10):
                sender.send_robot_command(cmd, override_id=grsim_id)
            sender.close()
            self._log.ok(
                f"grSim direct send OK: sent 10 UDP packets to {ip}:{port}. "
                "If the robot does not move, check grSim command port and team color.")
            self._n_sent += 10
            self._update_hw_status(f"grSim: sent to {ip}:{port}", SUCCESS)
        except socket.error as e:
            self._log.err(f"grSim SOCKET ERROR: {e}")
            self._n_err += 1
            self._update_hw_status(f"grSim: socket error - {e}", DANGER)
        except Exception as e:
            self._log.err(f"grSim ERROR: {type(e).__name__}: {e}")
            self._n_err += 1
            self._update_hw_status(f"grSim: error - {type(e).__name__}", DANGER)
        self._update_counts()

    def _update_counts(self):
        self._send_count.setText(f"{self._n_sent} sent")
        self._err_count.setText(f"{self._n_err} errors")

    def _reset_counts(self):
        self._n_sent = 0
        self._n_err = 0
        self._update_counts()
