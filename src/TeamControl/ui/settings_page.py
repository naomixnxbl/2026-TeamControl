"""
Settings page — combines Simulation controls, Config editor, and Network
details into one unified settings/tools page.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
)
from PySide6.QtCore import Qt

from TeamControl.ui.sim_panel import SimPanel
from TeamControl.ui.config_panel import ConfigPanel
from TeamControl.ui.network_panel import NetworkPanel


class SettingsPage(QWidget):
    """Unified settings: Simulation + Config + Network."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Left: sim + network stacked
        left_tabs = QTabWidget()
        self.sim_panel = SimPanel()
        self.network_panel = NetworkPanel()
        left_tabs.addTab(self.sim_panel, "Simulation")
        left_tabs.addTab(self.network_panel, "Network")
        splitter.addWidget(left_tabs)

        # Right: config editor
        self.config_panel = ConfigPanel()
        splitter.addWidget(self.config_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)
