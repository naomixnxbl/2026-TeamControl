"""
Log console — scrolling log viewer with level filtering and search.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPlainTextEdit,
                                QHBoxLayout, QPushButton, QLineEdit,
                                QComboBox, QLabel)
from PySide6.QtCore import Qt, QTime
from PySide6.QtGui import QTextCharFormat, QColor, QFont

from TeamControl.ui.theme import ACCENT, TEXT_DIM, SUCCESS, WARNING, DANGER, BLUE_TEAM


class LogPanel(QWidget):
    """Scrolling log console with filtering."""

    MAX_LINES = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        # Toolbar
        tb = QHBoxLayout()
        tb.setSpacing(4)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "engine", "sim", "vision", "gc", "recv", "error"])
        self._filter_combo.setFixedWidth(100)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search logs…")
        self._search.setFixedWidth(180)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear)

        self._auto_scroll = True
        scroll_btn = QPushButton("Auto-scroll: ON")
        scroll_btn.setFixedWidth(120)
        scroll_btn.clicked.connect(lambda: self._toggle_scroll(scroll_btn))

        self._count_lbl = QLabel("0 lines")
        self._count_lbl.setStyleSheet(f"color:{TEXT_DIM};")

        tb.addWidget(QLabel("Filter:"))
        tb.addWidget(self._filter_combo)
        tb.addWidget(self._search)
        tb.addStretch()
        tb.addWidget(self._count_lbl)
        tb.addWidget(scroll_btn)
        tb.addWidget(clear_btn)
        lay.addLayout(tb)

        # Text area
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self.MAX_LINES)
        self._text.setFont(QFont("Cascadia Code", 11))
        lay.addWidget(self._text)

        self._line_count = 0

    def append(self, message: str):
        ts = QTime.currentTime().toString("HH:mm:ss.zzz")

        # Apply filter
        filt = self._filter_combo.currentText()
        if filt != "All" and f"[{filt}]" not in message.lower():
            pass  # still store but could skip

        search = self._search.text().lower()
        if search and search not in message.lower():
            return

        color = "#eaeaea"
        if "[error]" in message.lower() or "error" in message.lower():
            color = DANGER
        elif "[warn" in message.lower():
            color = WARNING
        elif "[recv]" in message.lower():
            color = BLUE_TEAM
        elif "[sim]" in message.lower():
            color = SUCCESS
        elif "[engine]" in message.lower():
            color = ACCENT

        html_line = f'<span style="color:{TEXT_DIM}">{ts}</span> ' \
                    f'<span style="color:{color}">{message}</span>'
        self._text.appendHtml(html_line)

        self._line_count += 1
        self._count_lbl.setText(f"{self._line_count} lines")

        if self._auto_scroll:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _clear(self):
        self._text.clear()
        self._line_count = 0
        self._count_lbl.setText("0 lines")

    def _toggle_scroll(self, btn):
        self._auto_scroll = not self._auto_scroll
        btn.setText(f"Auto-scroll: {'ON' if self._auto_scroll else 'OFF'}")
