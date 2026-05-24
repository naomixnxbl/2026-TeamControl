"""
Configuration Editor — live YAML editor for ipconfig.yaml.

Tree view of all settings with inline editing.  Changes are written
back to the YAML file and the engine config is reloaded.
"""

from pathlib import Path

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QTreeWidget, QTreeWidgetItem, QPushButton,
                                QGroupBox, QCheckBox, QMessageBox,
                                QHeaderView, QLineEdit)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from TeamControl.ui.theme import ACCENT, TEXT_DIM, SUCCESS, DANGER, WARNING


CONFIG_PATH = Path(__file__).resolve().parent.parent / "utils" / "ipconfig.yaml"


class ConfigPanel(QWidget):
    """Editable YAML configuration tree."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        title = QLabel("Configuration")
        title.setStyleSheet(f"font-size:15px; font-weight:bold; color:{ACCENT};")
        lay.addWidget(title)

        path_lbl = QLabel(str(CONFIG_PATH))
        path_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        path_lbl.setWordWrap(True)
        lay.addWidget(path_lbl)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Key", "Value"])
        self._tree.setAlternatingRowColors(True)
        hh = self._tree.header()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        self._tree.setEditTriggers(QTreeWidget.DoubleClicked |
                                    QTreeWidget.EditKeyPressed)
        lay.addWidget(self._tree)

        # Quick toggles
        tg = QGroupBox("Quick Toggles")
        tgl = QHBoxLayout(tg)

        self._sim_vision = QCheckBox("Use grSim Vision")
        self._send_grsim = QCheckBox("Send to grSim")
        self._us_yellow = QCheckBox("We are Yellow")

        for cb in (self._sim_vision, self._send_grsim, self._us_yellow):
            cb.setStyleSheet("font-weight:bold;")
            tgl.addWidget(cb)
        tgl.addStretch()
        lay.addWidget(tg)

        # Buttons
        btn_row = QHBoxLayout()
        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self.load)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("startBtn")
        self._save_btn.clicked.connect(self._save)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{TEXT_DIM};")

        btn_row.addWidget(self._reload_btn)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._status)
        lay.addLayout(btn_row)

        self._raw: dict = {}
        self.load()

    # ── IO ────────────────────────────────────────────────────────

    def load(self):
        try:
            with open(CONFIG_PATH, "r") as f:
                self._raw = yaml.load(f, Loader)
        except Exception as e:
            self._status.setText(f"Load failed: {e}")
            self._status.setStyleSheet(f"color:{DANGER};")
            return

        self._tree.clear()
        self._populate(self._tree.invisibleRootItem(), self._raw)
        self._tree.expandAll()

        self._sim_vision.setChecked(bool(self._raw.get("use_grSim_vision", True)))
        self._send_grsim.setChecked(bool(self._raw.get("send_to_grSim", True)))
        self._us_yellow.setChecked(bool(self._raw.get("us_yellow", True)))

        self._status.setText("Loaded")
        self._status.setStyleSheet(f"color:{SUCCESS};")

    def _populate(self, parent_item, data, path=""):
        if isinstance(data, dict):
            for key, val in data.items():
                item = QTreeWidgetItem([str(key), ""])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                parent_item.addChild(item)
                if isinstance(val, dict):
                    self._populate(item, val, f"{path}.{key}")
                else:
                    item.setText(1, str(val))
                    item.setData(1, Qt.UserRole, f"{path}.{key}")
        elif isinstance(data, list):
            for i, val in enumerate(data):
                item = QTreeWidgetItem([f"[{i}]", str(val)])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                parent_item.addChild(item)

    def _save(self):
        self._collect(self._tree.invisibleRootItem(), self._raw)

        self._raw["use_grSim_vision"] = self._sim_vision.isChecked()
        self._raw["send_to_grSim"] = self._send_grsim.isChecked()
        self._raw["us_yellow"] = self._us_yellow.isChecked()

        try:
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(self._raw, f, Dumper=Dumper, default_flow_style=False,
                          sort_keys=False)
            self._status.setText("Saved!")
            self._status.setStyleSheet(f"color:{SUCCESS};")
            self.config_changed.emit()
        except Exception as e:
            self._status.setText(f"Save failed: {e}")
            self._status.setStyleSheet(f"color:{DANGER};")

    def _collect(self, parent_item, data):
        """Walk tree items and write edited values back into dict."""
        for i in range(parent_item.childCount()):
            item = parent_item.child(i)
            key = item.text(0)
            if item.childCount() > 0:
                if key in data and isinstance(data[key], dict):
                    self._collect(item, data[key])
            else:
                if key in data:
                    data[key] = self._coerce(item.text(1), data[key])

    @staticmethod
    def _coerce(text: str, original):
        """Try to keep the original type."""
        if isinstance(original, bool):
            return text.lower() in ("true", "1", "yes")
        if isinstance(original, int):
            try:
                return int(text)
            except ValueError:
                return original
        if isinstance(original, float):
            try:
                return float(text)
            except ValueError:
                return original
        return text
