"""
Onboard Possession test tab.

Shows, per robot:
  - Whether we're receiving fresh onboard-camera packets (freshness, counts)
  - What the robot's camera is reporting (found / pixel / bearing / radius / conf)
  - Whether the robot *claims* possession (camera-side rule)
  - Whether the robot *actually* has the ball (world-model rule: close to the
    SSL-Vision ball position, within POSSESS_DIST)
  - Whether the two agree

Useful for validating the onboard → possession pipeline without running
full team logic.
"""

import math
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPushButton, QDoubleSpinBox, QComboBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont

from TeamControl.ui.theme import (
    ACCENT, TEXT, TEXT_DIM, SUCCESS, DANGER, WARNING, BORDER,
    YELLOW_TEAM, BLUE_TEAM,
)
POSSESS_DIST_DEFAULT = 500  # mm — mirrors TeamControl.robot.team.POSSESS_DIST


def _card(title_text):
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


def _value(text="—", size=12, bold=True):
    lbl = QLabel(text)
    weight = QFont.Bold if bold else QFont.Normal
    lbl.setFont(QFont("Segoe UI", size, weight))
    return lbl


class OnboardPossessionPanel(QWidget):
    """Live view of onboard-camera packets + possession agreement test."""

    COLS = [
        "Team", "ID", "Fresh", "Found", "Conf", "Radius", "Bearing°",
        "Px", "Py", "Age (s)", "Cam says ball?", "Dist to vision ball",
        "World says ball?", "Agree?",
    ]

    def __init__(self, parent=None, engine=None):
        super().__init__(parent)
        self._engine = engine
        self._last_frame = None
        self._packet_count = 0
        self._last_packet_ts = 0.0
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        if engine is not None:
            engine.frame_ready.connect(self._on_frame)
            engine.onboard_packet.connect(self._on_onboard_packet)

    # ── Build ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Header
        hdr = QLabel("Onboard Possession Tester")
        hdr.setStyleSheet(
            f"font-size:16px; font-weight:bold; color:{ACCENT};")
        root.addWidget(hdr)

        # ── Status + thresholds ──────────────────────────────────
        status_card, sl = _card("Status")
        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Packets received:"), 0, 0)
        self._pkt_lbl = _value("0")
        grid.addWidget(self._pkt_lbl, 0, 1)

        grid.addWidget(QLabel("Last packet:"), 0, 2)
        self._last_pkt_lbl = _value("—")
        grid.addWidget(self._last_pkt_lbl, 0, 3)

        grid.addWidget(QLabel("Known robots in store:"), 1, 0)
        self._store_size_lbl = _value("0")
        grid.addWidget(self._store_size_lbl, 1, 1)

        grid.addWidget(QLabel("IP→robot map entries:"), 1, 2)
        self._map_lbl = _value("0")
        grid.addWidget(self._map_lbl, 1, 3)

        sl.addLayout(grid)

        # Thresholds row
        thr = QHBoxLayout()
        thr.setSpacing(8)
        thr.addWidget(QLabel("Max age (s):"))
        self._max_age = QDoubleSpinBox()
        self._max_age.setRange(0.1, 10.0)
        self._max_age.setSingleStep(0.1)
        self._max_age.setValue(1.0)
        thr.addWidget(self._max_age)

        thr.addWidget(QLabel("Min conf:"))
        self._min_conf = QDoubleSpinBox()
        self._min_conf.setRange(0.0, 1.0)
        self._min_conf.setSingleStep(0.05)
        self._min_conf.setValue(0.3)
        thr.addWidget(self._min_conf)

        thr.addWidget(QLabel("Min radius (px):"))
        self._min_radius = QDoubleSpinBox()
        self._min_radius.setRange(0.0, 500.0)
        self._min_radius.setSingleStep(1.0)
        self._min_radius.setValue(15.0)
        self._min_radius.setToolTip(
            "Pixel radius threshold above which the camera 'has' the ball "
            "(bigger blob = closer).")
        thr.addWidget(self._min_radius)

        thr.addWidget(QLabel("Max |bearing| (°):"))
        self._max_bearing = QDoubleSpinBox()
        self._max_bearing.setRange(1.0, 180.0)
        self._max_bearing.setSingleStep(1.0)
        self._max_bearing.setValue(25.0)
        self._max_bearing.setToolTip(
            "Ball must be roughly in front of the camera to count as possession.")
        thr.addWidget(self._max_bearing)

        thr.addWidget(QLabel("World possess dist (mm):"))
        self._poss_dist = QDoubleSpinBox()
        self._poss_dist.setRange(50.0, 2000.0)
        self._poss_dist.setSingleStep(10.0)
        self._poss_dist.setValue(float(POSSESS_DIST_DEFAULT))
        thr.addWidget(self._poss_dist)

        thr.addStretch()
        clr = QPushButton("Reset counters")
        clr.clicked.connect(self._reset_counters)
        thr.addWidget(clr)

        sl.addLayout(thr)
        root.addWidget(status_card)

        # ── Verdict card ─────────────────────────────────────────
        verdict_card, vl = _card("Verdict")
        vg = QGridLayout()
        vg.setSpacing(8)

        vg.addWidget(QLabel("Camera claims possession:"), 0, 0)
        self._cam_verdict = _value("—", size=14)
        vg.addWidget(self._cam_verdict, 0, 1)

        vg.addWidget(QLabel("World model says possession:"), 1, 0)
        self._world_verdict = _value("—", size=14)
        vg.addWidget(self._world_verdict, 1, 1)

        vg.addWidget(QLabel("Agree?"), 2, 0)
        self._agree_lbl = _value("—", size=14)
        vg.addWidget(self._agree_lbl, 2, 1)

        vl.addLayout(vg)
        root.addWidget(verdict_card)

        # ── Per-robot table ─────────────────────────────────────
        tbl_card, tl = _card("Per-Robot Readings")
        self._table = QTableWidget(0, len(self.COLS))
        self._table.setHorizontalHeaderLabels(self.COLS)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setMinimumSectionSize(60)
        tl.addWidget(self._table)
        root.addWidget(tbl_card, 1)

    # ── Slots ────────────────────────────────────────────────────

    def _on_frame(self, snap):
        self._last_frame = snap

    def _on_onboard_packet(self, obs, addr):
        self._packet_count += 1
        self._last_packet_ts = time.time()

    def _reset_counters(self):
        self._packet_count = 0
        self._last_packet_ts = 0.0
        if self._engine is not None:
            self._engine.onboard_store.clear()

    # ── Refresh loop ────────────────────────────────────────────

    def _refresh(self):
        if self._engine is None:
            return

        store = self._engine.onboard_store
        snapshot = store.snapshot()
        ball_xy = self._ball_xy(self._last_frame)
        robot_poses = self._robot_poses(self._last_frame)

        self._pkt_lbl.setText(str(self._packet_count))
        if self._last_packet_ts > 0:
            age = time.time() - self._last_packet_ts
            self._last_pkt_lbl.setText(f"{age:.2f}s ago")
            self._last_pkt_lbl.setStyleSheet(
                f"color:{SUCCESS if age < 1 else (WARNING if age < 3 else DANGER)};")
        else:
            self._last_pkt_lbl.setText("— (no packets)")
            self._last_pkt_lbl.setStyleSheet(f"color:{TEXT_DIM};")

        self._store_size_lbl.setText(str(len(snapshot)))
        self._map_lbl.setText(str(len(self._engine.ip_to_robot)))

        # Thresholds
        max_age = self._max_age.value()
        min_conf = self._min_conf.value()
        min_radius = self._min_radius.value()
        max_bearing = math.radians(self._max_bearing.value())
        poss_dist = self._poss_dist.value()

        # Build table rows sorted by (team, id) for stable order
        rows = sorted(snapshot.items(), key=lambda kv: (not kv[0][0], kv[0][1]))
        self._table.setRowCount(len(rows))

        cam_claims = []  # (team_name, id)
        now = time.time()

        for i, ((is_yellow, rid), obs) in enumerate(rows):
            age = now - obs.recv_ts
            fresh = age <= max_age
            bearing_deg = math.degrees(obs.bearing)

            cam_has = (fresh and obs.found and
                       obs.confidence >= min_conf and
                       obs.radius >= min_radius and
                       abs(obs.bearing) <= max_bearing)

            pose = robot_poses.get((is_yellow, rid))
            if pose is not None and ball_xy is not None:
                d_vision = math.hypot(pose[0] - ball_xy[0], pose[1] - ball_xy[1])
                world_has = d_vision < poss_dist
                dist_txt = f"{d_vision:.0f} mm"
            else:
                d_vision = None
                world_has = False
                dist_txt = "—"

            agree = cam_has == world_has

            if cam_has:
                cam_claims.append((is_yellow, rid))

            team_txt = "Yellow" if is_yellow else "Blue"
            team_color = QColor(YELLOW_TEAM if is_yellow else BLUE_TEAM)

            cells = [
                (team_txt, team_color),
                (str(rid), None),
                ("YES" if fresh else "stale",
                 QColor(SUCCESS) if fresh else QColor(DANGER)),
                ("YES" if obs.found else "no",
                 QColor(SUCCESS) if obs.found else QColor(TEXT_DIM)),
                (f"{obs.confidence:.2f}",
                 QColor(SUCCESS) if obs.confidence >= min_conf else QColor(DANGER)),
                (f"{obs.radius:.1f}", None),
                (f"{bearing_deg:+.1f}", None),
                (f"{obs.px:.0f}", None),
                (f"{obs.py:.0f}", None),
                (f"{age:.2f}",
                 QColor(SUCCESS) if fresh else QColor(DANGER)),
                ("YES" if cam_has else "no",
                 QColor(SUCCESS) if cam_has else QColor(TEXT_DIM)),
                (dist_txt, None),
                ("YES" if world_has else "no",
                 QColor(SUCCESS) if world_has else QColor(TEXT_DIM)),
                ("✓" if agree else "✗",
                 QColor(SUCCESS) if agree else QColor(DANGER)),
            ]
            for col, (text, color) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if color is not None:
                    item.setForeground(color)
                if col == 0:
                    item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                self._table.setItem(i, col, item)

        # World verdict: closest robot to the ball
        world_claim = self._world_closest(robot_poses, ball_xy, poss_dist)

        self._cam_verdict.setText(self._format_claim(cam_claims))
        self._cam_verdict.setStyleSheet(
            f"color:{SUCCESS if cam_claims else TEXT_DIM}; font-weight:bold;")

        self._world_verdict.setText(
            self._format_claim([world_claim]) if world_claim else "— (no ball / no robots)")
        self._world_verdict.setStyleSheet(
            f"color:{SUCCESS if world_claim else TEXT_DIM}; font-weight:bold;")

        agree_world = {world_claim} if world_claim else set()
        agree_cam = set(cam_claims)
        if not agree_cam and not agree_world:
            self._agree_lbl.setText("— (nobody claims)")
            self._agree_lbl.setStyleSheet(f"color:{TEXT_DIM};")
        elif agree_cam == agree_world:
            self._agree_lbl.setText("YES — camera matches world model")
            self._agree_lbl.setStyleSheet(f"color:{SUCCESS}; font-weight:bold;")
        else:
            self._agree_lbl.setText("NO — mismatch")
            self._agree_lbl.setStyleSheet(f"color:{DANGER}; font-weight:bold;")

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _ball_xy(snap):
        if snap is None or snap.ball is None:
            return None
        return (snap.ball.x, snap.ball.y)

    @staticmethod
    def _robot_poses(snap):
        poses = {}
        if snap is None:
            return poses
        for r in snap.yellow:
            poses[(True, r.id)] = (r.x, r.y, r.o)
        for r in snap.blue:
            poses[(False, r.id)] = (r.x, r.y, r.o)
        return poses

    @staticmethod
    def _world_closest(poses, ball_xy, poss_dist):
        if not poses or ball_xy is None:
            return None
        best = None
        best_d = float("inf")
        for key, pose in poses.items():
            d = math.hypot(pose[0] - ball_xy[0], pose[1] - ball_xy[1])
            if d < best_d:
                best_d = d
                best = key
        if best is None or best_d > poss_dist:
            return None
        return best

    @staticmethod
    def _format_claim(claims):
        if not claims:
            return "— (nobody)"
        parts = [f"{'Yellow' if y else 'Blue'} #{rid}" for (y, rid) in claims]
        return ", ".join(parts)
