"""
Robot Dashboard — shows live telemetry for every robot on the field.

Columns: Team | ID | X | Y | θ° | Confidence | Role
Yellow robots at top, blue below, with team-coloured highlights.
"""

import math
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget,
                                QTableWidgetItem, QHeaderView, QLabel,
                                QHBoxLayout, QGroupBox, QAbstractItemView)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

from TeamControl.ui.theme import YELLOW_TEAM, BLUE_TEAM, TEXT_DIM, ACCENT


class RobotPanel(QWidget):
    """Live robot telemetry table."""

    COLUMNS = ["Team", "ID", "X (mm)", "Y (mm)", "θ (deg)", "Conf", "Role"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Robot Dashboard")
        header.setStyleSheet(f"font-size:15px; font-weight:bold; color:{ACCENT};")
        self._layout.addWidget(header)

        # Summary bar
        self._summary = QLabel("Waiting for data…")
        self._summary.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
        self._layout.addWidget(self._summary)

        # Table
        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setMinimumSectionSize(50)
        self._table.setShowGrid(False)

        self._layout.addWidget(self._table)

        # Detail area
        self._detail_group = QGroupBox("Selected Robot")
        dl = QVBoxLayout(self._detail_group)
        self._detail_label = QLabel("Click a robot row for details.")
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(f"color:{TEXT_DIM}; padding:4px;")
        dl.addWidget(self._detail_label)
        self._layout.addWidget(self._detail_group)

        self._table.currentCellChanged.connect(self._on_select)

        self._last_snap = None

    def update_frame(self, snap):
        self._last_snap = snap
        robots = []
        for r in snap.yellow:
            robots.append(("Yellow", r))
        for r in snap.blue:
            robots.append(("Blue", r))

        self._table.setRowCount(len(robots))
        for row, (team, r) in enumerate(robots):
            color = QColor(YELLOW_TEAM) if team == "Yellow" else QColor(BLUE_TEAM)
            items = [
                team,
                str(r.id),
                f"{r.x:.1f}",
                f"{r.y:.1f}",
                f"{math.degrees(r.o):.1f}",
                f"{r.confidence:.2f}",
                "—",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    item.setForeground(color)
                    item.setFont(QFont("Segoe UI", 11, QFont.Bold))
                self._table.setItem(row, col, item)

        ny = len(snap.yellow)
        nb = len(snap.blue)
        ball_str = (f"Ball ({snap.ball.x:.0f}, {snap.ball.y:.0f})"
                    if snap.ball else "No ball")
        self._summary.setText(
            f"Yellow: {ny}   Blue: {nb}   {ball_str}   "
            f"Frame #{snap.frame_number}")

    def _on_select(self, row, col, prev_row, prev_col):
        if self._last_snap is None or row < 0:
            return
        all_robots = list(self._last_snap.yellow) + list(self._last_snap.blue)
        if row >= len(all_robots):
            return
        r = all_robots[row]
        team = "Yellow" if r.team == "yellow" else "Blue"
        deg = math.degrees(r.o)
        self._detail_label.setText(
            f"<b>{team} Robot #{r.id}</b><br>"
            f"Position: ({r.x:.1f}, {r.y:.1f}) mm<br>"
            f"Orientation: {deg:.1f}°  ({r.o:.4f} rad)<br>"
            f"Confidence: {r.confidence:.3f}<br>"
        )
