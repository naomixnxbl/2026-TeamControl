"""
Network Monitor — shows connection status and packet rates.
"""

import time
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox,
                                QGridLayout, QLabel, QProgressBar)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from TeamControl.ui.theme import (ACCENT, TEXT_DIM, SUCCESS, DANGER,
                                   WARNING, YELLOW_TEAM, BLUE_TEAM)


class _StatusDot(QLabel):
    def __init__(self, parent=None):
        super().__init__("●", parent)
        self.setFixedWidth(20)
        self.setAlignment(Qt.AlignCenter)
        self.set_status(False)

    def set_status(self, ok):
        color = SUCCESS if ok else DANGER
        self.setStyleSheet(f"color:{color}; font-size:16px;")


class NetworkPanel(QWidget):
    """Live network connection status dashboard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        title = QLabel("Network Monitor")
        title.setStyleSheet(f"font-size:15px; font-weight:bold; color:{ACCENT};")
        lay.addWidget(title)

        # Connections group
        cg = QGroupBox("Connections")
        gl = QGridLayout(cg)
        gl.setVerticalSpacing(6)

        self._vision_dot = _StatusDot()
        self._gc_dot = _StatusDot()
        self._grsim_dot = _StatusDot()

        self._vision_rate = QLabel("—")
        self._gc_rate = QLabel("—")
        self._grsim_rate = QLabel("—")

        gl.addWidget(QLabel("Vision Multicast"), 0, 0)
        gl.addWidget(self._vision_dot, 0, 1)
        gl.addWidget(self._vision_rate, 0, 2)

        gl.addWidget(QLabel("Game Controller"), 1, 0)
        gl.addWidget(self._gc_dot, 1, 1)
        gl.addWidget(self._gc_rate, 1, 2)

        gl.addWidget(QLabel("grSim"), 2, 0)
        gl.addWidget(self._grsim_dot, 2, 1)
        gl.addWidget(self._grsim_rate, 2, 2)
        lay.addWidget(cg)

        # Frame rate
        fr_box = QGroupBox("Performance")
        fl = QGridLayout(fr_box)
        self._fps_lbl = QLabel("0 fps")
        self._fps_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self._fps_lbl.setAlignment(Qt.AlignCenter)
        self._fps_bar = QProgressBar()
        self._fps_bar.setRange(0, 60)
        self._fps_bar.setValue(0)
        self._fps_bar.setTextVisible(False)
        self._fps_bar.setFixedHeight(12)

        self._latency_lbl = QLabel("Latency: —")
        self._latency_lbl.setStyleSheet(f"color:{TEXT_DIM};")

        fl.addWidget(QLabel("Frame Rate"), 0, 0)
        fl.addWidget(self._fps_lbl, 0, 1)
        fl.addWidget(self._fps_bar, 1, 0, 1, 2)
        fl.addWidget(self._latency_lbl, 2, 0, 1, 2)
        lay.addWidget(fr_box)

        # Process status
        proc_box = QGroupBox("Processes")
        pl = QGridLayout(proc_box)
        self._proc_labels = {}
        names = ["VisionProcess", "GCfsm", "WMWorker", "Dispatcher"]
        for i, name in enumerate(names):
            dot = _StatusDot()
            lbl = QLabel(name)
            pl.addWidget(dot, i, 0)
            pl.addWidget(lbl, i, 1)
            self._proc_labels[name] = dot
        lay.addWidget(proc_box)

        lay.addStretch()

        # FPS tracking
        self._frame_times: list[float] = []
        self._last_frame_time = 0.0

    def on_frame(self):
        now = time.time()
        self._frame_times.append(now)
        cutoff = now - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff]
        fps = len(self._frame_times)
        self._fps_lbl.setText(f"{fps} fps")
        self._fps_bar.setValue(min(fps, 60))

        if self._last_frame_time > 0:
            lat = (now - self._last_frame_time) * 1000
            self._latency_lbl.setText(f"Latency: {lat:.1f} ms")
        self._last_frame_time = now

        self._vision_dot.set_status(fps > 0)

    def set_engine_running(self, running):
        for dot in self._proc_labels.values():
            dot.set_status(running)
        self._grsim_dot.set_status(running)
        self._gc_dot.set_status(running)
        if not running:
            self._fps_lbl.setText("0 fps")
            self._fps_bar.setValue(0)
