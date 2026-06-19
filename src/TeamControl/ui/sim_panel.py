"""
Simulation controls for grSim ball/robot placement and runtime channels.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout,
    QPushButton, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QSizePolicy,
)
from PySide6.QtCore import Signal

from TeamControl.ui.theme import ACCENT


class SimPanel(QWidget):
    """Simulation control panel for grSim ball/robot placement."""

    place_ball_requested = Signal(float, float, float, float)
    place_robot_requested = Signal(int, bool, float, float, float)
    field_place_ball = Signal()
    field_place_robot = Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 560)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        title = QLabel("Simulation Controls")
        title.setStyleSheet(f"font-size:15px; font-weight:bold; color:{ACCENT};")
        lay.addWidget(title)

        ball_box = QGroupBox("Ball Placement")
        ball_box.setMinimumHeight(126)
        bg = QGridLayout(ball_box)
        self._configure_form_grid(bg)
        self._ball_x = self._make_spin(-5000, 5000, 0)
        self._ball_y = self._make_spin(-3000, 3000, 0)
        self._ball_vx = self._make_spin(-10000, 10000, 0)
        self._ball_vy = self._make_spin(-10000, 10000, 0)
        bg.addWidget(QLabel("X (mm):"), 0, 0)
        bg.addWidget(self._ball_x, 0, 1)
        bg.addWidget(QLabel("Y (mm):"), 0, 2)
        bg.addWidget(self._ball_y, 0, 3)
        bg.addWidget(QLabel("VX:"), 1, 0)
        bg.addWidget(self._ball_vx, 1, 1)
        bg.addWidget(QLabel("VY:"), 1, 2)
        bg.addWidget(self._ball_vy, 1, 3)
        btn_row = QHBoxLayout()
        place_ball = QPushButton("Place Ball")
        place_ball.setMinimumHeight(28)
        place_ball.clicked.connect(self._on_place_ball)
        click_ball = QPushButton("Click on Field")
        click_ball.setMinimumHeight(28)
        click_ball.setStyleSheet(f"color:{ACCENT}; font-weight:bold;")
        click_ball.clicked.connect(self.field_place_ball.emit)
        center_ball = QPushButton("Center")
        center_ball.setMinimumHeight(28)
        center_ball.clicked.connect(self._center_ball)
        btn_row.addWidget(place_ball)
        btn_row.addWidget(click_ball)
        btn_row.addWidget(center_ball)
        bg.addLayout(btn_row, 2, 0, 1, 4)
        lay.addWidget(ball_box)

        robot_box = QGroupBox("Robot Placement")
        robot_box.setMinimumHeight(146)
        rg = QGridLayout(robot_box)
        self._configure_form_grid(rg)
        self._robot_team = QComboBox()
        self._robot_team.setMinimumWidth(96)
        self._robot_team.addItems(["Yellow", "Blue"])
        self._robot_id = QSpinBox()
        self._robot_id.setRange(0, 15)
        self._robot_id.setMinimumWidth(96)
        self._robot_x = self._make_spin(-5000, 5000, 0)
        self._robot_y = self._make_spin(-3000, 3000, 0)
        self._robot_o = self._make_spin(-180, 180, 0, suffix=" deg")
        rg.addWidget(QLabel("Team:"), 0, 0)
        rg.addWidget(self._robot_team, 0, 1)
        rg.addWidget(QLabel("ID:"), 0, 2)
        rg.addWidget(self._robot_id, 0, 3)
        rg.addWidget(QLabel("X (mm):"), 1, 0)
        rg.addWidget(self._robot_x, 1, 1)
        rg.addWidget(QLabel("Y (mm):"), 1, 2)
        rg.addWidget(self._robot_y, 1, 3)
        rg.addWidget(QLabel("Theta:"), 2, 0)
        rg.addWidget(self._robot_o, 2, 1)
        rbtn = QHBoxLayout()
        place_robot = QPushButton("Place Robot")
        place_robot.setMinimumHeight(28)
        place_robot.clicked.connect(self._on_place_robot)
        click_robot = QPushButton("Click on Field")
        click_robot.setMinimumHeight(28)
        click_robot.setStyleSheet(f"color:{ACCENT}; font-weight:bold;")
        click_robot.clicked.connect(self._on_click_place_robot)
        rbtn.addWidget(place_robot)
        rbtn.addWidget(click_robot)
        rg.addLayout(rbtn, 3, 0, 1, 4)
        lay.addWidget(robot_box)

        qa = QGroupBox("Quick Actions")
        qa.setMinimumHeight(116)
        ql = QVBoxLayout(qa)
        reset_btn = QPushButton("Reset Ball to Center")
        reset_btn.setMinimumHeight(28)
        reset_btn.clicked.connect(self._center_ball)
        kickoff_btn = QPushButton("Kickoff Formation")
        kickoff_btn.setMinimumHeight(28)
        kickoff_btn.setToolTip("Place all robots in kickoff positions")
        kickoff_btn.clicked.connect(self._kickoff_formation)
        ql.addWidget(reset_btn)
        ql.addWidget(kickoff_btn)
        lay.addWidget(qa)

        chan_box = QGroupBox("Runtime Channels")
        chan_box.setMinimumHeight(188)
        cg = QVBoxLayout(chan_box)
        self._channel_checks = {
            "vision": QCheckBox("Vision"),
            "gc": QCheckBox("Game Controller"),
            "robot_recv": QCheckBox("Robot Telemetry Recv"),
            "use_grsim": QCheckBox("Use grSim Vision"),
            "send_grsim": QCheckBox("Send Commands to grSim"),
            "record_wm": QCheckBox("Record World Model"),
        }
        for key, cb in self._channel_checks.items():
            cb.setChecked(key != "record_wm")
            cb.setMinimumHeight(24)
            cg.addWidget(cb)
        lay.addWidget(chan_box)
        lay.addStretch()

    def channel_options(self) -> dict:
        return {key: cb.isChecked() for key, cb in self._channel_checks.items()}

    def set_channel_defaults(self, config):
        if config is None:
            return
        defaults = {
            "vision": True,
            "gc": True,
            "robot_recv": False,
            "use_grsim": bool(getattr(config, "use_grSim_vision", False)),
            "send_grsim": bool(getattr(config, "send_to_grSim", False)),
            "record_wm": bool(getattr(config, "record_world_snapshots", False)),
        }
        for key, val in defaults.items():
            if key in self._channel_checks:
                self._channel_checks[key].setChecked(val)

    def set_engine_running(self, running: bool):
        editable = not running
        for cb in self._channel_checks.values():
            cb.setEnabled(editable)

    def set_channel_controls_enabled(self, enabled: bool):
        self.set_engine_running(not enabled)

    @staticmethod
    def _make_spin(lo, hi, val, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setDecimals(1)
        spin.setSingleStep(100)
        spin.setMinimumWidth(96)
        spin.setMinimumHeight(24)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _configure_form_grid(grid: QGridLayout):
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setColumnMinimumWidth(0, 58)
        grid.setColumnMinimumWidth(1, 96)
        grid.setColumnMinimumWidth(2, 52)
        grid.setColumnMinimumWidth(3, 96)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

    def _on_place_ball(self):
        self.place_ball_requested.emit(
            self._ball_x.value(), self._ball_y.value(),
            self._ball_vx.value(), self._ball_vy.value())

    def _center_ball(self):
        self._ball_x.setValue(0)
        self._ball_y.setValue(0)
        self._ball_vx.setValue(0)
        self._ball_vy.setValue(0)
        self.place_ball_requested.emit(0, 0, 0, 0)

    def _on_place_robot(self):
        import math
        self.place_robot_requested.emit(
            self._robot_id.value(),
            self._robot_team.currentText() == "Yellow",
            self._robot_x.value(),
            self._robot_y.value(),
            math.radians(self._robot_o.value()))

    def _on_click_place_robot(self):
        self.field_place_robot.emit(
            self._robot_id.value(),
            self._robot_team.currentText() == "Yellow")

    def _kickoff_formation(self):
        import math
        positions_yellow = [
            (0, -2200, 0, 0),
            (1, -800, 600, 0),
            (2, -800, -600, 0),
            (3, -200, 0, 0),
            (4, -400, 1200, 0),
            (5, -400, -1200, 0),
        ]
        positions_blue = [
            (0, 2200, 0, math.pi),
            (1, 800, 600, math.pi),
            (2, 800, -600, math.pi),
            (3, 200, 0, math.pi),
            (4, 400, 1200, math.pi),
            (5, 400, -1200, math.pi),
        ]
        for rid, x, y, o in positions_yellow:
            self.place_robot_requested.emit(rid, True, x, y, o)
        for rid, x, y, o in positions_blue:
            self.place_robot_requested.emit(rid, False, x, y, o)
        self.place_ball_requested.emit(0, 0, 0, 0)
