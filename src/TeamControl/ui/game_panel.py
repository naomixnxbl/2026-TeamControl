"""
Game State panel — shows referee commands, stage, score, and state machine.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGroupBox,
                                QGridLayout, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from TeamControl.ui.theme import (ACCENT, TEXT_DIM, SUCCESS, WARNING,
                                   DANGER, YELLOW_TEAM, BLUE_TEAM)


def _pill(text, color):
    return (f'<span style="background:{color}; color:#111; '
            f'padding:2px 10px; border-radius:8px; font-weight:bold;">'
            f'{text}</span>')


class GamePanel(QWidget):
    """Displays live game-controller state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        title = QLabel("Game Controller")
        title.setStyleSheet(f"font-size:15px; font-weight:bold; color:{ACCENT};")
        lay.addWidget(title)

        # State group
        sg = QGroupBox("State")
        gl = QGridLayout(sg)
        gl.setVerticalSpacing(6)

        self._state_lbl = self._make_value("WAITING")
        self._stage_lbl = self._make_value("—")
        self._command_lbl = self._make_value("—")
        self._our_team_lbl = self._make_value("—")

        gl.addWidget(QLabel("Game State:"), 0, 0)
        gl.addWidget(self._state_lbl, 0, 1)
        gl.addWidget(QLabel("Stage:"), 1, 0)
        gl.addWidget(self._stage_lbl, 1, 1)
        gl.addWidget(QLabel("Command:"), 2, 0)
        gl.addWidget(self._command_lbl, 2, 1)
        gl.addWidget(QLabel("Our Team:"), 3, 0)
        gl.addWidget(self._our_team_lbl, 3, 1)
        lay.addWidget(sg)

        # Score
        score_box = QGroupBox("Score")
        sl = QGridLayout(score_box)
        self._yellow_score = QLabel("0")
        self._yellow_score.setStyleSheet(
            f"font-size:32px; font-weight:bold; color:{YELLOW_TEAM};")
        self._yellow_score.setAlignment(Qt.AlignCenter)
        self._blue_score = QLabel("0")
        self._blue_score.setStyleSheet(
            f"font-size:32px; font-weight:bold; color:{BLUE_TEAM};")
        self._blue_score.setAlignment(Qt.AlignCenter)
        vs = QLabel("vs")
        vs.setAlignment(Qt.AlignCenter)
        vs.setStyleSheet("font-size:18px; font-weight:bold;")

        sl.addWidget(QLabel("Yellow"), 0, 0, Qt.AlignCenter)
        sl.addWidget(QLabel("Blue"), 0, 2, Qt.AlignCenter)
        sl.addWidget(self._yellow_score, 1, 0)
        sl.addWidget(vs, 1, 1)
        sl.addWidget(self._blue_score, 1, 2)
        lay.addWidget(score_box)

        # Possession / tactical
        tact_box = QGroupBox("Tactical Overview")
        tl = QGridLayout(tact_box)
        self._possession_lbl = self._make_value("—")
        self._active_robots_lbl = self._make_value("—")

        tl.addWidget(QLabel("Possession:"), 0, 0)
        tl.addWidget(self._possession_lbl, 0, 1)
        tl.addWidget(QLabel("Active Robots:"), 1, 0)
        tl.addWidget(self._active_robots_lbl, 1, 1)
        lay.addWidget(tact_box)

        lay.addStretch()

    @staticmethod
    def _make_value(text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        return lbl

    # ── Update API ────────────────────────────────────────────────

    def update_game_state(self, state):
        if state is None:
            self._state_lbl.setText("NO DATA")
            self._state_lbl.setStyleSheet(f"color:{TEXT_DIM};")
            return

        name = state.name if hasattr(state, "name") else str(state)
        color_map = {
            "HALTED": DANGER,
            "STOPPED": WARNING,
            "RUNNING": SUCCESS,
        }
        c = color_map.get(name, ACCENT)
        self._state_lbl.setText(name)
        self._state_lbl.setStyleSheet(f"color:{c}; font-weight:bold;")

    def update_frame(self, snap):
        ny = len(snap.yellow)
        nb = len(snap.blue)
        self._active_robots_lbl.setText(f"Yellow: {ny}  Blue: {nb}")

    def set_our_team(self, is_yellow):
        if is_yellow:
            self._our_team_lbl.setText("YELLOW")
            self._our_team_lbl.setStyleSheet(f"color:{YELLOW_TEAM}; font-weight:bold;")
        else:
            self._our_team_lbl.setText("BLUE")
            self._our_team_lbl.setStyleSheet(f"color:{BLUE_TEAM}; font-weight:bold;")

    def set_mode(self, mode):
        self._command_lbl.setText(mode.upper())
