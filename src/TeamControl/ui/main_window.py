"""
Main Window — clean tabbed layout, no docks.

Tabs:
  Dashboard       Field + robots + game state + network sidebar
  Behavior Tree   Interactive BT visualizer / editor
  Hardware Test   Manual robot control & testing console
  Settings        Simulation controls + Config editor + Network
  Console         Scrolling log viewer
"""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)
from PySide6.QtGui import QAction, QFont, QIcon

from TeamControl.ui.calibration_page import CalibrationPage
from TeamControl.ui.dashboard_page import DashboardPage
from TeamControl.ui.dispatcher_panel import DispatcherPanel
from TeamControl.ui.engine import SimEngine
from TeamControl.ui.log_panel import LogPanel
from TeamControl.ui.map_canvas import MapCanvas, MapDebugWidget
from TeamControl.ui.onboard_possession_panel import OnboardPossessionPanel
from TeamControl.ui.settings_page import SettingsPage
from TeamControl.ui.skill_lab_page import SkillLabPage
from TeamControl.ui.test_panel import TestPanel
from TeamControl.ui.theme import ACCENT, DANGER, QSS, SUCCESS, TEXT, TEXT_DIM, WARNING


class MainWindow(QMainWindow):
    """TurtleRabbit SSL Command Center."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TurtleRabbit — SSL Command Center")
        self.setMinimumSize(900, 620)
        self.resize(1800, 1050)
        self.setStyleSheet(QSS)

        # ── Engine ────────────────────────────────────────────────
        self._engine = SimEngine(self)

        # ── Shared field canvas ──────────────────────────────────
        self._field = MapCanvas()

        # ── Pages ─────────────────────────────────────────────────
        self._test_panel = TestPanel(engine=self._engine, field=self._field)
        self._dispatch_panel = DispatcherPanel(engine=self._engine)
        self._calibration = CalibrationPage(
            engine=self._engine,
            test_panel=self._test_panel,
        )
        self._dashboard = DashboardPage(
            self._field, engine=self._engine, test_panel=self._test_panel,
        )
        self._settings = SettingsPage()
        self._settings.set_channel_defaults(self._engine.reload_config())
        self._log_panel = LogPanel()
        self._map_debug = MapDebugWidget()
        self._onboard_panel = OnboardPossessionPanel(engine=self._engine)
        self._skill_lab = SkillLabPage(engine=self._engine, field=self._field)

        # ── Central tabs ──────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setObjectName("mainTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setElideMode(Qt.ElideRight)
        self._tabs.tabBar().setUsesScrollButtons(True)

        # Dashboard is the top-level Home tab; Calibration has its own main tab.
        self._tabs.addTab(self._dashboard, "  Home  ")
        self._tabs.addTab(self._map_debug, "  World Map  ")
        self._tabs.addTab(self._calibration, "  Calibration  ")
        self._tabs.addTab(self._settings, "  Settings  ")
        self._tabs.addTab(self._log_panel, "  Console  ")
        self._tabs.addTab(self._test_panel, "  Hardware Test  ")
        self._tabs.addTab(self._skill_lab, "  Skill Lab  ")
        self._tabs.addTab(self._dispatch_panel, "  Dispatcher  ")
        self._tabs.addTab(self._onboard_panel, "  Onboard Possession  ")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self._tabs)

        # ── Toolbar ──────────────────────────────────────────────
        self._build_toolbar()

        # ── Menu ─────────────────────────────────────────────────
        self._build_menu()

        # ── Status bar ───────────────────────────────────────────
        self._status_mode = QLabel("Mode: —")
        self._status_mode.setStyleSheet(f"color:{TEXT_DIM}; padding:0 12px;")
        self._status_coords = QLabel("(—, —)")
        self._status_coords.setStyleSheet(f"color:{TEXT}; padding:0 12px;")
        self._status_fps = QLabel("0 fps")
        self._status_fps.setStyleSheet(f"color:{TEXT_DIM}; padding:0 12px;")

        sb = QStatusBar()
        sb.addWidget(self._status_mode)
        sb.addWidget(self._status_coords)
        sb.addPermanentWidget(self._status_fps)
        self.setStatusBar(sb)

        # Give calibration and test panel access to the "Our Bot" spinner
        self._calibration.set_our_bot_spin(self._our_id_spin)
        self._test_panel.set_our_bot_spin(self._our_id_spin)

        # ── Wire signals ─────────────────────────────────────────
        self._wire_signals()

        # Show cal card for the default selected mode right away
        self._on_mode_combo_changed(self._mode_combo.currentText())

        # ── Boot log ─────────────────────────────────────────────
        self._log_panel.append("[engine] TurtleRabbit Command Center ready")
        self._log_panel.append("[engine] Select a mode and click Start")

    # ── Toolbar ──────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Control")
        tb.setMovable(False)
        self.addToolBar(tb)

        logo = QLabel("  TurtleRabbit  ")
        logo.setFont(QFont("Segoe UI", 14, QFont.Bold))
        logo.setStyleSheet(f"color:{ACCENT};")
        tb.addWidget(logo)
        tb.addSeparator()

        tb.addWidget(QLabel("  Mode: "))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(SimEngine.MODES)
        self._mode_combo.setMinimumWidth(130)
        self._mode_combo.currentTextChanged.connect(self._on_mode_combo_changed)
        tb.addWidget(self._mode_combo)
        tb.addSeparator()

        tb.addWidget(QLabel("  Our ID: "))
        self._our_id_spin = QSpinBox()
        self._our_id_spin.setRange(0, 15)
        self._our_id_spin.setValue(0)
        self._our_id_spin.setToolTip("Shell ID of our robot (from ipconfig.yaml)")
        self._our_id_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._our_id_spin.setAlignment(Qt.AlignCenter)
        self._our_id_spin.setFixedWidth(44)
        tb.addWidget(self._our_id_spin)

        tb.addWidget(QLabel("  Enemy ID: "))
        self._enemy_id_spin = QSpinBox()
        self._enemy_id_spin.setRange(0, 15)
        self._enemy_id_spin.setValue(0)
        self._enemy_id_spin.setToolTip("Shell ID of enemy robot")
        self._enemy_id_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._enemy_id_spin.setAlignment(Qt.AlignCenter)
        self._enemy_id_spin.setFixedWidth(44)
        tb.addWidget(self._enemy_id_spin)
        tb.addSeparator()

        self._start_btn = QPushButton("  Start  ")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.clicked.connect(self._on_start)
        tb.addWidget(self._start_btn)

        self._stop_btn = QPushButton("  Stop  ")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        tb.addWidget(self._stop_btn)
        tb.addSeparator()

        self._state_label = QLabel("  IDLE  ")
        self._state_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self._state_label.setStyleSheet(f"color:{TEXT_DIM};")
        tb.addWidget(self._state_label)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

    # ── Menu ─────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        file_menu.addAction("Reload Config", self._settings.config_panel.load)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        view_menu = mb.addMenu("View")
        view_menu.addAction("Reset Field View", self._field._zoom_fit)
        view_menu.addSeparator()
        view_menu.addAction(
            "Calibration",
            lambda checked=False: self._tabs.setCurrentWidget(self._calibration),
        )

        sim_menu = mb.addMenu("Simulation")
        sim_menu.addAction("Center Ball", lambda: self._place_ball_from_dashboard(0, 0))
        sim_menu.addAction(
            "Kickoff Formation", self._settings.sim_panel._kickoff_formation
        )

        mode_menu = mb.addMenu("Mode")
        for m in SimEngine.MODES:
            mode_menu.addAction(
                m.capitalize(), lambda checked=False, mode=m: self._switch_mode(mode)
            )

        help_menu = mb.addMenu("Help")
        help_menu.addAction("About", self._show_about)

    def _show_about(self):
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "TurtleRabbit",
            "WSU TurtleRabbit SSL Command Center\n\n"
            "RoboCup Small Size League\n"
            "Team Control Dashboard v2.0",
        )

    # ── Signal wiring ────────────────────────────────────────────

    def _wire_signals(self):
        eng = self._engine

        eng.frame_ready.connect(self._on_frame)
        eng.map_render_ready.connect(self._map_debug.set_render_data)
        eng.map_render_ready.connect(self._dashboard.set_render_data)
        eng.field_geometry_ready.connect(self._field.set_field_geometry)
        eng.field_geometry_ready.connect(self._map_debug.set_field_geometry)
        eng.field_geometry_ready.connect(self._calibration.set_field_geometry)
        eng.game_state_ready.connect(self._dashboard.update_game_state)
        eng.bt_state_ready.connect(self._dashboard.update_bt_state)
        eng.dispatch_info.connect(self._dispatch_panel.update_info)
        eng.channel_status_ready.connect(self._dashboard.update_channel_status)
        eng.engine_started.connect(self._on_engine_started)
        eng.engine_stopped.connect(self._on_engine_stopped)
        eng.log_message.connect(self._log_panel.append)

        self._dashboard.coordinate_hover.connect(self._on_coord_hover)

        sp = self._settings.sim_panel
        sp.place_ball_requested.connect(self._place_ball_from_dashboard)
        sp.place_robot_requested.connect(self._place_robot_from_dashboard)
        sp.field_place_ball.connect(self._begin_field_ball_placement)
        sp.field_place_robot.connect(
            self._begin_field_robot_placement
        )

        self._field.ball_placed.connect(self._place_ball_from_field)
        self._field.robot_placed.connect(self._place_robot_from_field)
        self._field.point_picked.connect(self._go_to_point_from_field)
        self._field.action_requested.connect(self._field_action_requested)
        self._field.robot_selected.connect(self._select_robot_from_field)

        self._settings.config_panel.config_changed.connect(
            lambda: self._log_panel.append("[config] Configuration saved")
        )

    # ── Handlers ─────────────────────────────────────────────────

    def _on_tab_changed(self, index):
        active = self._tabs.widget(index)
        self._engine.set_map_render_enabled(
            active is self._dashboard or active is self._map_debug
        )

    def _dashboard_block_reason(self, action_name: str) -> str | None:
        reason = self._engine.dashboard_action_block_reason()
        if reason:
            self._log_panel.append(f"[dashboard] {action_name} blocked: {reason}")
        return reason

    def _begin_field_ball_placement(self):
        if self._dashboard_block_reason("Place ball"):
            return
        self._field.set_place_mode("ball")
        self._log_panel.append("[dashboard] Click the field to place the ball")

    def _begin_field_robot_placement(self, robot_id: int, is_yellow: bool):
        if self._dashboard_block_reason("Place robot"):
            return
        self._field.set_place_mode(("robot", robot_id, is_yellow))
        team = "Yellow" if is_yellow else "Blue"
        self._log_panel.append(
            f"[dashboard] Click the field to place {team} #{robot_id}"
        )

    def _place_ball_from_dashboard(self, x_mm, y_mm, vx=0.0, vy=0.0):
        if self._engine.place_ball(x_mm, y_mm, vx, vy):
            self._field.set_ball_place_marker(x_mm, y_mm)

    def _place_ball_from_field(self, x_mm, y_mm):
        self._place_ball_from_dashboard(x_mm, y_mm)

    def _place_robot_from_dashboard(self, robot_id, is_yellow, x_mm, y_mm, orientation):
        self._engine.place_robot(robot_id, is_yellow, x_mm, y_mm, orientation)

    def _place_robot_from_field(self, robot_id, is_yellow, x_mm, y_mm):
        self._place_robot_from_dashboard(robot_id, is_yellow, x_mm, y_mm, 0.0)

    def _go_to_point_from_field(self, x_mm, y_mm):
        if self._tabs.currentWidget() is self._skill_lab:
            self._skill_lab.set_target_point(x_mm, y_mm)
            return
        if self._dashboard_block_reason("Go to point"):
            return
        self._test_panel.go_to_point(x_mm, y_mm)

    def _field_action_requested(self, action_name: str):
        if action_name != "stop" and self._dashboard_block_reason(action_name):
            return
        self._test_panel.field_action(action_name)

    def _select_robot_from_field(self, is_yellow: bool, robot_id: int):
        if self._tabs.currentWidget() is self._skill_lab:
            self._skill_lab.select_robot(is_yellow, robot_id)
            return
        if self._dashboard_block_reason("Robot control selection"):
            return
        self._test_panel.select_robot(is_yellow, robot_id)
        team = "Yellow" if is_yellow else "Blue"
        self._log_panel.append(
            f"[dashboard] Field control target set to {team} #{robot_id}"
        )

    def _on_mode_combo_changed(self, mode):
        self._dashboard.set_mode(mode)

    def _on_start(self):
        mode = self._mode_combo.currentText()
        our_id = self._our_id_spin.value()
        enemy_id = self._enemy_id_spin.value()
        self._log_panel.append(
            f"[engine] Starting {mode} — our bot #{our_id}, enemy bot #{enemy_id}"
        )
        try:
            self._engine.start(
                mode,
                our_id=our_id,
                enemy_id=enemy_id,
                channel_options=self._settings.channel_options(),
            )
        except Exception as e:
            self._log_panel.append(f"[error] Failed to start: {e}")

    def _on_stop(self):
        self._log_panel.append("[engine] Stopping…")
        self._engine.stop()

    def _switch_mode(self, mode):
        idx = SimEngine.MODES.index(mode)
        self._mode_combo.setCurrentIndex(idx)
        if self._engine.is_running:
            self._engine.stop()
        self._engine.start(
            mode,
            our_id=self._our_id_spin.value(),
            enemy_id=self._enemy_id_spin.value(),
            channel_options=self._settings.channel_options(),
        )

    def _on_engine_started(self, mode):
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._mode_combo.setEnabled(False)
        self._our_id_spin.setEnabled(False)
        self._enemy_id_spin.setEnabled(False)
        self._state_label.setText(f"  RUNNING — {mode.upper()}  ")
        self._state_label.setStyleSheet(f"color:{SUCCESS}; font-weight:bold;")
        self._status_mode.setText(f"Mode: {mode}")
        self._dashboard.set_mode(mode)
        self._dashboard.set_engine_running(True)
        self._settings.set_engine_running(True)
        self._dispatch_panel.set_running(True)
        self._on_tab_changed(self._tabs.currentIndex())

    def _on_engine_stopped(self):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._mode_combo.setEnabled(True)
        self._our_id_spin.setEnabled(True)
        self._enemy_id_spin.setEnabled(True)
        self._state_label.setText("  IDLE  ")
        self._state_label.setStyleSheet(f"color:{TEXT_DIM};")
        self._status_mode.setText("Mode: —")
        self._dashboard.set_engine_running(False)
        self._settings.set_engine_running(False)
        self._dispatch_panel.set_running(False)
        self._skill_lab.stop_skill()

    def _on_frame(self, snap):
        self._dashboard.update_frame(snap)
        fps = self._dashboard.get_fps()
        self._status_fps.setText(f"{fps} fps")

    def _on_coord_hover(self, x, y):
        self._status_coords.setText(f"({x:.0f}, {y:.0f}) mm")

    # ── Cleanup ──────────────────────────────────────────────────

    def closeEvent(self, event):
        self._skill_lab.stop_skill()
        if self._engine.is_running:
            self._engine.stop()
        event.accept()
