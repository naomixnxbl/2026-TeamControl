"""
Simulation Controls — ball placement, robot teleportation, reset.

Click-to-place integration with the FieldCanvas.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QGroupBox, QGridLayout, QPushButton,
                                QDoubleSpinBox, QSpinBox, QComboBox,
                                QCheckBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from TeamControl.ui.theme import (ACCENT, TEXT_DIM, SUCCESS, DANGER,
                                   YELLOW_TEAM, BLUE_TEAM)


class SimPanel(QWidget):
    """Simulation control panel for grSim ball/robot placement."""

    place_ball_requested = Signal(float, float, float, float)
    place_robot_requested = Signal(int, bool, float, float, float)
    field_place_ball = Signal()         # tell field canvas to enter ball-place mode
    field_place_robot = Signal(int, bool)  # robot-place mode

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        title = QLabel("Simulation Controls")
        title.setStyleSheet(f"font-size:15px; font-weight:bold; color:{ACCENT};")
        lay.addWidget(title)

        # ── Ball placement ────────────────────────────────────────
        ball_box = QGroupBox("Ball Placement")
        bg = QGridLayout(ball_box)

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
        place_ball.clicked.connect(self._on_place_ball)
        click_ball = QPushButton("Click on Field")
        click_ball.setStyleSheet(f"color:{ACCENT}; font-weight:bold;")
        click_ball.clicked.connect(lambda: self.field_place_ball.emit())
        center_ball = QPushButton("Center")
        center_ball.clicked.connect(self._center_ball)
        btn_row.addWidget(place_ball)
        btn_row.addWidget(click_ball)
        btn_row.addWidget(center_ball)
        bg.addLayout(btn_row, 2, 0, 1, 4)
        lay.addWidget(ball_box)

        # ── Robot placement ───────────────────────────────────────
        robot_box = QGroupBox("Robot Placement")
        rg = QGridLayout(robot_box)

        self._robot_team = QComboBox()
        self._robot_team.addItems(["Yellow", "Blue"])
        self._robot_id = QSpinBox()
        self._robot_id.setRange(0, 15)
        self._robot_x = self._make_spin(-5000, 5000, 0)
        self._robot_y = self._make_spin(-3000, 3000, 0)
        self._robot_o = self._make_spin(-180, 180, 0, suffix="°")

        rg.addWidget(QLabel("Team:"), 0, 0)
        rg.addWidget(self._robot_team, 0, 1)
        rg.addWidget(QLabel("ID:"), 0, 2)
        rg.addWidget(self._robot_id, 0, 3)
        rg.addWidget(QLabel("X (mm):"), 1, 0)
        rg.addWidget(self._robot_x, 1, 1)
        rg.addWidget(QLabel("Y (mm):"), 1, 2)
        rg.addWidget(self._robot_y, 1, 3)
        rg.addWidget(QLabel("θ:"), 2, 0)
        rg.addWidget(self._robot_o, 2, 1)

        rbtn = QHBoxLayout()
        place_robot = QPushButton("Place Robot")
        place_robot.clicked.connect(self._on_place_robot)
        click_robot = QPushButton("Click on Field")
        click_robot.setStyleSheet(f"color:{ACCENT}; font-weight:bold;")
        click_robot.clicked.connect(self._on_click_place_robot)
        rbtn.addWidget(place_robot)
        rbtn.addWidget(click_robot)
        rg.addLayout(rbtn, 3, 0, 1, 4)
        lay.addWidget(robot_box)

        # ── Quick actions ─────────────────────────────────────────
        qa = QGroupBox("Quick Actions")
        ql = QVBoxLayout(qa)

        reset_btn = QPushButton("Reset Ball to Center")
        reset_btn.clicked.connect(self._center_ball)

        kickoff_btn = QPushButton("Kickoff Formation")
        kickoff_btn.setToolTip("Place all robots in kickoff positions")
        kickoff_btn.clicked.connect(self._kickoff_formation)

        ql.addWidget(reset_btn)
        ql.addWidget(kickoff_btn)
        lay.addWidget(qa)

        lay.addStretch()

    @staticmethod
    def _make_spin(lo, hi, val, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setDecimals(1)
        spin.setSingleStep(100)
        if suffix:
            spin.setSuffix(suffix)
        return spin

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
            (0, -2200, 0, 0),   # goalie
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
