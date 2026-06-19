"""
Dashboard page — field canvas on the left, tabbed sidebar on the right.

Right panel:
  Monitor   — runtime channels, robot table, game state
"""

import math
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGridLayout, QFrame, QScrollArea, QCheckBox, QTabWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

from TeamControl.ui.theme import (
    ACCENT, TEXT_DIM, SUCCESS, WARNING, DANGER,
    YELLOW_TEAM, BLUE_TEAM, BALL_COLOR, BORDER, BG_CARD, BG_DARK,
    ACCENT, TEXT_DIM, SUCCESS, WARNING, DANGER, TEXT,
    YELLOW_TEAM, BLUE_TEAM, BALL_COLOR,
)


_ROLE_COLOR = {
    "GOALIE":    YELLOW_TEAM,
    "ATTACKER":  DANGER,
    "DEFENDER":  BLUE_TEAM,
    "SUPPORTER": TEXT_DIM,
}

_PHASE_COLOR = {
    "RUNNING":   SUCCESS,
    "STOPPED":   WARNING,
    "HALTED":    DANGER,
    "HALF_TIME": TEXT_DIM,
}


def _fmt_intent(intent_type, target):
    if not intent_type:
        return "—"
    if target:
        return f"{intent_type}({target[0]:.2f},{target[1]:.2f})"
    return intent_type


class _BTInspectorPanel(QWidget):
    """Fixed-grid BT state display — updates labels in place, no scroll."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 4)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        self._tick_lbl  = QLabel("tick —")
        self._phase_lbl = QLabel("—")
        self._ball_lbl  = QLabel("ball (—, —)")
        for lbl in (self._tick_lbl, self._phase_lbl, self._ball_lbl):
            lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        pipe = QLabel(" | ")
        pipe.setStyleSheet(f"color:{TEXT_DIM};")
        pipe2 = QLabel(" | ")
        pipe2.setStyleSheet(f"color:{TEXT_DIM};")
        hdr.addWidget(self._tick_lbl)
        hdr.addWidget(pipe)
        hdr.addWidget(self._phase_lbl)
        hdr.addWidget(pipe2)
        hdr.addWidget(self._ball_lbl)
        hdr.addStretch()
        lay.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};")
        lay.addWidget(sep)

        grid = QGridLayout()
        grid.setSpacing(3)
        grid.setContentsMargins(0, 0, 0, 0)
        for col, txt in enumerate(["ID", "Role", "Pos (m)", "θ", "Intent"]):
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color:{ACCENT}; font-weight:bold; font-size:10px;")
            grid.addWidget(lbl, 0, col)
        grid.setColumnStretch(4, 1)

        self._robot_rows: dict[int, dict[str, QLabel]] = {}
        mono = QFont("Consolas", 10)
        for i in range(6):
            labels = {
                "id":     QLabel(str(i)),
                "role":   QLabel("—"),
                "pos":    QLabel("(—,—)"),
                "ori":    QLabel("—"),
                "intent": QLabel("—"),
            }
            for lbl in labels.values():
                lbl.setFont(mono)
            grid.addWidget(labels["id"],     i + 1, 0)
            grid.addWidget(labels["role"],   i + 1, 1)
            grid.addWidget(labels["pos"],    i + 1, 2)
            grid.addWidget(labels["ori"],    i + 1, 3)
            grid.addWidget(labels["intent"], i + 1, 4)
            self._robot_rows[i] = labels

        lay.addLayout(grid)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color:{BORDER};")
        lay.addWidget(sep2)

        self._status_lbl = QLabel("Waiting for BT data…")
        self._status_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        lay.addWidget(self._status_lbl)
        lay.addStretch()

    def update_state(self, state: dict) -> None:
        tick  = state.get("tick", 0)
        phase = state.get("phase", "?")
        bpos  = state.get("ball", (0.0, 0.0))
        is_y  = state.get("is_yellow", True)

        self._tick_lbl.setText(f"tick {tick}")

        pc = _PHASE_COLOR.get(phase, TEXT)
        self._phase_lbl.setText(phase)
        self._phase_lbl.setStyleSheet(f"color:{pc}; font-weight:bold;")

        tc = YELLOW_TEAM if is_y else BLUE_TEAM
        self._ball_lbl.setText(f"ball ({bpos[0]:.2f},{bpos[1]:.2f})")
        self._ball_lbl.setStyleSheet(f"color:{tc};")

        for rb in state.get("robots", []):
            rid = rb.get("id", -1)
            row = self._robot_rows.get(rid)
            if row is None:
                continue
            role = rb.get("role", "?")
            pos  = rb.get("pos")
            ori  = rb.get("ori")
            it   = rb.get("intent_type")
            tgt  = rb.get("intent_target")

            rc = _ROLE_COLOR.get(role, TEXT)
            row["role"].setText(role)
            row["role"].setStyleSheet(f"color:{rc}; font-weight:bold;")
            row["pos"].setText(f"({pos[0]:.2f},{pos[1]:.2f})" if pos else "N/A")
            row["ori"].setText(f"{ori:.2f}" if ori is not None else "—")
            row["intent"].setText(_fmt_intent(it, tgt))

        self._status_lbl.setText(f"Live  ·  tick {tick}")
        self._status_lbl.setStyleSheet(f"color:{SUCCESS}; font-size:10px;")


def _card(title_text):
    card = QFrame()
    card.setObjectName("card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(6)
    title = QLabel(title_text)
    title.setStyleSheet(f"font-size:13px; font-weight:bold; color:{ACCENT}; padding:0;")
    lay.addWidget(title)
    return card, lay


def _val_label(text="—", size=12, bold=True):
    lbl = QLabel(text)
    weight = QFont.Bold if bold else QFont.Normal
    lbl.setFont(QFont("Segoe UI", size, weight))
    return lbl


class DashboardPage(QWidget):
    """Field canvas + tabbed right sidebar."""

    coordinate_hover = Signal(float, float)
    _DASHBOARD_HIDDEN_MAP_LAYERS = {"Robots", "Ball"}

    def __init__(self, field_canvas, parent=None, engine=None,
                 test_panel=None):
        super().__init__(parent)
        self._field = field_canvas
        self._engine = engine
        self._map_layer_checks: dict[str, QCheckBox] = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._field)

        # ── Right sidebar: tabbed monitor + BT inspector ──────────
        self._bt_panel = _BTInspectorPanel()
        sidebar = QTabWidget()
        sidebar.setDocumentMode(True)
        sidebar.addTab(self._build_monitor_panel(), "Monitor")
        sidebar.addTab(self._bt_panel, "BT")
        sidebar.setMinimumWidth(260)
        sidebar.setMaximumWidth(480)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([900, 340])

        root.addWidget(splitter)

        self._field.coordinate_hover.connect(self.coordinate_hover.emit)
        if hasattr(self._field, "layers_changed"):
            self._field.layers_changed.connect(self._sync_map_layer_controls)

        self._frame_times: list[float] = []
        self._current_mode = "vision_only"

    # ── Monitor tab ───────────────────────────────────────────────

    def _build_monitor_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(8, 8, 8, 8)
        inner_lay.setSpacing(8)

        self._build_channels_card(inner_lay)
        self._build_map_layers_card(inner_lay)
        self._build_robot_card(inner_lay)
        self._build_game_card(inner_lay)
        inner_lay.addStretch()

        scroll.setWidget(inner)
        lay.addWidget(scroll)
        return panel

    # ── Runtime channels card ─────────────────────────────────────

    _CHANNEL_NAMES = {
        "vision":     "Vision",
        "gc":         "Game Controller",
        "robot_recv": "Robot Recv",
        "use_grsim":  "Use grSim",
        "send_grsim": "Send → grSim",
        "record_wm":  "Record WM",
    }

    def _build_channels_card(self, parent_lay):
        card, lay = _card("Runtime Channels")
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)

        self._channel_dots = {}
        self._channel_name_lbls = {}
        self._channel_latency = {}

        for row, (key, name) in enumerate(self._CHANNEL_NAMES.items()):
            dot = QLabel("●")
            dot.setFixedWidth(18)
            dot.setAlignment(Qt.AlignCenter)
            dot.setStyleSheet(f"color:{TEXT_DIM}; font-size:14px;")
            self._channel_dots[key] = dot

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"color:{TEXT_DIM};")
            self._channel_name_lbls[key] = name_lbl

            lat = QLabel("—")
            lat.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lat.setMinimumWidth(52)
            lat.setStyleSheet(f"color:{TEXT_DIM};")
            self._channel_latency[key] = lat

            grid.addWidget(dot,      row, 0)
            grid.addWidget(name_lbl, row, 1)
            grid.addWidget(lat,      row, 2)
            if key == "vision":
                self._fps_lbl = QLabel("0 fps")
                self._fps_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._fps_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
                self._fps_lbl.setStyleSheet(f"color:{TEXT_DIM};")
                grid.addWidget(self._fps_lbl, row, 3)

        lay.addLayout(grid)
        parent_lay.addWidget(card)

    def update_channel_status(self, status):
        for key in self._CHANNEL_NAMES:
            item = status.get(key, {})
            enabled    = bool(item.get("enabled", False))
            latency_ms = item.get("latency_ms")
            stale      = bool(item.get("stale", True))

            dot      = self._channel_dots[key]
            name_lbl = self._channel_name_lbls[key]
            lat      = self._channel_latency[key]

            if not enabled:
                dot.setStyleSheet(f"color:{TEXT_DIM}; font-size:14px;")
                name_lbl.setStyleSheet(f"color:{TEXT_DIM};")
                lat.setText("OFF")
                lat.setStyleSheet(f"color:{TEXT_DIM};")
            elif stale or latency_ms is None:
                dot.setStyleSheet(f"color:{DANGER}; font-size:14px;")
                name_lbl.setStyleSheet(f"color:{TEXT_DIM};")
                lat.setText(">99ms")
                lat.setStyleSheet(f"color:{DANGER}; font-weight:bold;")
            else:
                dot.setStyleSheet(f"color:{SUCCESS}; font-size:14px;")
                name_lbl.setStyleSheet(f"color:{TEXT};")
                lat.setText(f"{latency_ms}ms")
                lat.setStyleSheet(f"color:{SUCCESS}; font-weight:bold;")

    # ── Robot table card ──────────────────────────────────────────

    def _build_robot_card(self, parent_lay):
        card, lay = _card("Robots")
        self._robot_summary = QLabel("Waiting for data…")
        self._robot_summary.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        lay.addWidget(self._robot_summary)

        cols = ["Team", "ID", "X", "Y", "θ°", "Conf"]
        self._robot_table = QTableWidget(0, len(cols))
        self._robot_table.setHorizontalHeaderLabels(cols)
        self._robot_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._robot_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._robot_table.setAlternatingRowColors(True)
        self._robot_table.verticalHeader().setVisible(False)
        self._robot_table.setShowGrid(False)
        self._robot_table.setMinimumHeight(150)
        self._robot_table.setMaximumHeight(300)
        hh = self._robot_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        hh.setMinimumSectionSize(40)
        self._robot_table.verticalHeader().setDefaultSectionSize(32)
        lay.addWidget(self._robot_table)
        parent_lay.addWidget(card)
        self._robot_table_items: dict[tuple[int, int], QTableWidgetItem] = {}
        self._robot_table_n_rows: int = -1

    # ── Game state card ───────────────────────────────────────────

    def _build_game_card(self, parent_lay):
        card, lay = _card("Game State")
        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("State:"), 0, 0)
        self._gs_state = _val_label("WAITING")
        grid.addWidget(self._gs_state, 0, 1)

        grid.addWidget(QLabel("Mode:"), 1, 0)
        self._gs_mode = _val_label("—")
        grid.addWidget(self._gs_mode, 1, 1)

        grid.addWidget(QLabel("Score:"), 2, 0)
        score_row = QHBoxLayout()
        self._score_y = QLabel("0")
        self._score_y.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{YELLOW_TEAM};")
        self._score_vs = QLabel(" vs ")
        self._score_vs.setStyleSheet("font-size:14px; font-weight:bold;")
        self._score_b = QLabel("0")
        self._score_b.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{BLUE_TEAM};")
        score_row.addWidget(self._score_y)
        score_row.addWidget(self._score_vs)
        score_row.addWidget(self._score_b)
        score_row.addStretch()
        grid.addLayout(score_row, 2, 1)

        grid.addWidget(QLabel("Command:"), 3, 0)
        self._gs_command = _val_label("NO DATA")
        grid.addWidget(self._gs_command, 3, 1)

        grid.addWidget(QLabel("Stage:"), 4, 0)
        self._gs_stage = _val_label("NO DATA")
        grid.addWidget(self._gs_stage, 4, 1)

        grid.addWidget(QLabel("Us Yellow:"), 5, 0)
        self._gs_us_yellow = _val_label("NO DATA")
        grid.addWidget(self._gs_us_yellow, 5, 1)

        grid.addWidget(QLabel("Cards:"), 6, 0)
        self._gs_cards = _val_label("NO DATA")
        grid.addWidget(self._gs_cards, 6, 1)

        grid.addWidget(QLabel("Fouls:"), 7, 0)
        self._gs_fouls = _val_label("NO DATA")
        grid.addWidget(self._gs_fouls, 7, 1)

        grid.addWidget(QLabel("YC Timers:"), 8, 0)
        self._gs_yellow_card_times = _val_label("NO DATA", size=10)
        self._gs_yellow_card_times.setWordWrap(True)
        grid.addWidget(self._gs_yellow_card_times, 8, 1)

        lay.addLayout(grid)
        parent_lay.addWidget(card)

    # ── Public update methods ─────────────────────────────────────

    def update_bt_state(self, state: dict) -> None:
        self._bt_panel.update_state(state)

    def update_frame(self, snap):
        self._field.set_frame(snap)
        self._update_robot_table(snap)
        self._update_fps()

    def update_game_state(self, state):
        gc_status = state if isinstance(state, dict) else {"state": state}
        game_state = gc_status.get("state")
        if game_state is None:
            self._gs_state.setText("NO DATA")
            self._gs_state.setStyleSheet(f"color:{TEXT_DIM};")
        else:
            name = game_state.name if hasattr(game_state, "name") else str(game_state)
            color_map = {"HALTED": DANGER, "STOPPED": WARNING, "RUNNING": SUCCESS}
            c = color_map.get(name, ACCENT)
            self._gs_state.setText(name)
            self._gs_state.setStyleSheet(f"color:{c}; font-weight:bold;")

        self._set_gc_value(self._gs_command, gc_status.get("command"))
        self._set_gc_value(self._gs_stage, gc_status.get("stage"))

        us_yellow = gc_status.get("us_yellow")
        if us_yellow is None:
            self._gs_us_yellow.setText("NO DATA")
            self._gs_us_yellow.setStyleSheet(f"color:{TEXT_DIM};")
        else:
            self._gs_us_yellow.setText("YES" if us_yellow else "NO")
            color = YELLOW_TEAM if us_yellow else BLUE_TEAM
            self._gs_us_yellow.setStyleSheet(f"color:{color}; font-weight:bold;")

        self._update_gc_discipline(gc_status)

    def _set_gc_value(self, label, value):
        if value is None:
            label.setText("NO DATA")
            label.setStyleSheet(f"color:{TEXT_DIM};")
            return
        label.setText(value.name if hasattr(value, "name") else str(value))
        label.setStyleSheet(f"color:{TEXT}; font-weight:bold;")

    def _update_gc_discipline(self, gc_status):
        yellow_cards = gc_status.get("yellow_cards")
        red_cards = gc_status.get("red_cards")
        fouls = gc_status.get("fouls")
        timers = gc_status.get("yellow_card_times") or []

        if yellow_cards is None and red_cards is None:
            self._gs_cards.setText("NO DATA")
            self._gs_cards.setStyleSheet(f"color:{TEXT_DIM};")
        else:
            yellow_text = "?" if yellow_cards is None else str(yellow_cards)
            red_text = "?" if red_cards is None else str(red_cards)
            self._gs_cards.setText(f"Y {yellow_text} / R {red_text}")
            color = DANGER if red_cards else WARNING if yellow_cards else SUCCESS
            self._gs_cards.setStyleSheet(f"color:{color}; font-weight:bold;")

        if fouls is None:
            self._gs_fouls.setText("NO DATA")
            self._gs_fouls.setStyleSheet(f"color:{TEXT_DIM};")
        else:
            self._gs_fouls.setText(str(fouls))
            self._gs_fouls.setStyleSheet(f"color:{WARNING if fouls else SUCCESS}; font-weight:bold;")

        if not timers:
            self._gs_yellow_card_times.setText("NONE")
            self._gs_yellow_card_times.setStyleSheet(f"color:{TEXT_DIM};")
        else:
            self._gs_yellow_card_times.setText(", ".join(self._format_gc_time(t) for t in timers))
            self._gs_yellow_card_times.setStyleSheet(f"color:{WARNING}; font-weight:bold;")

    def _format_gc_time(self, value):
        try:
            seconds = max(0, int(value) // 1_000_000)
        except (TypeError, ValueError):
            return str(value)
        return f"{seconds // 60}:{seconds % 60:02d}"

    def set_mode(self, mode):
        self._current_mode = mode
        self._gs_mode.setText(mode.upper())
        if mode == "coop":
            from TeamControl.robot.coop import HOME_YELLOW, HOME_BLUE, BALL_START
            self._field.set_targets([
                (*HOME_YELLOW, YELLOW_TEAM),
                (*HOME_BLUE, BLUE_TEAM),
                (*BALL_START, BALL_COLOR),
            ])
            self._field.set_paths([([HOME_YELLOW, HOME_BLUE], ACCENT)])
        else:
            self._field.set_targets([])
            self._field.set_paths([])

    def set_engine_running(self, running):
        if not running:
            self._fps_lbl.setText("0 fps")

    def get_fps(self):
        return len(self._frame_times)

    # ── Internal ──────────────────────────────────────────────────

    def _update_robot_table(self, snap):
        yellow = [r for r in snap.yellow if r is not None]
        blue   = [r for r in snap.blue   if r is not None]
        robots = [(YELLOW_TEAM, "Y", r) for r in yellow] + \
                 [(BLUE_TEAM,   "B", r) for r in blue]
        n = len(robots)

        self._robot_table.blockSignals(True)
        if n != self._robot_table_n_rows:
            self._robot_table.setRowCount(n)
            self._robot_table_n_rows = n
            # Populate any new rows with fresh items; existing rows keep theirs.
            for row in range(n):
                for col in range(6):
                    if (row, col) not in self._robot_table_items:
                        item = QTableWidgetItem()
                        item.setTextAlignment(Qt.AlignCenter)
                        self._robot_table.setItem(row, col, item)
                        self._robot_table_items[(row, col)] = item

        for row, (color_hex, team_short, r) in enumerate(robots):
            vals = [team_short, str(r.id), f"{r.x:.0f}",
                    f"{r.y:.0f}", f"{math.degrees(r.o):.1f}",
                    f"{r.confidence:.2f}"]
            for col, text in enumerate(vals):
                item = self._robot_table_items.get((row, col))
                if item is None:
                    continue
                if item.text() != text:
                    item.setText(text)
                if col == 0:
                    item.setForeground(QColor(color_hex))
                    item.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self._robot_table.blockSignals(False)

        ny, nb = len(yellow), len(blue)
        ball = (f"Ball ({snap.ball.x:.0f}, {snap.ball.y:.0f})"
                if snap.ball else "No ball")
        self._robot_summary.setText(
            f"Y:{ny}  B:{nb}  |  {ball}  |  Frame #{snap.frame_number}")

    def _update_fps(self):
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if t > now - 1.0]
        self._fps_lbl.setText(f"{len(self._frame_times)} fps")

    # ── Map layer controls ───────────────────────────────────────

    def _build_map_layers_card(self, parent_lay):
        card, lay = _card("Map Layers")
        self._map_layer_hint = QLabel("Waiting for map data...")
        self._map_layer_hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        lay.addWidget(self._map_layer_hint)

        self._map_layer_lay = QVBoxLayout()
        self._map_layer_lay.setContentsMargins(0, 0, 0, 0)
        self._map_layer_lay.setSpacing(2)
        lay.addLayout(self._map_layer_lay)
        parent_lay.addWidget(card)

    def set_render_data(self, render_data):
        if hasattr(self._field, "set_render_data"):
            self._field.set_render_data(render_data)

    def _sync_map_layer_controls(self, layers):
        active_names = {layer.name for layer in layers}
        self._map_layer_hint.setVisible(not bool(active_names))

        for layer in layers:
            if layer.name in self._map_layer_checks:
                continue

            visible = self._field.is_layer_visible(layer.name)
            if layer.name in self._DASHBOARD_HIDDEN_MAP_LAYERS:
                visible = False
                self._field.set_layer_visible(layer.name, False)

            check = QCheckBox(layer.name)
            check.setChecked(visible)
            check.toggled.connect(
                lambda checked, name=layer.name: self._field.set_layer_visible(
                    name, checked
                )
            )
            self._map_layer_lay.addWidget(check)
            self._map_layer_checks[layer.name] = check

        for name, check in tuple(self._map_layer_checks.items()):
            if name not in active_names:
                self._map_layer_lay.removeWidget(check)
                check.deleteLater()
                del self._map_layer_checks[name]
