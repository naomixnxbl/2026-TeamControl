"""
Behavior-Tree Visualiser — maps out the decision logic for each robot role.

Draws an interactive tree graph with QPainter.  Nodes light up based on the
current game state so you can see exactly which branch each robot is executing.

Editable parameters are shown in a side panel when a node is selected.
"""

import math
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QComboBox, QScrollArea, QGroupBox,
                                QGridLayout, QDoubleSpinBox, QFrame,
                                QSplitter, QSizePolicy, QToolTip,
                                QPushButton)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QSize
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                           QPainterPath, QMouseEvent, QPaintEvent,
                           QWheelEvent, QTransform)

from TeamControl.ui.theme import (ACCENT, BG_DARK, BG_MID, BG_PANEL,
                                   BORDER, TEXT, TEXT_DIM, SUCCESS,
                                   WARNING, DANGER, ROLE_GOALIE,
                                   ROLE_ATTACKER, ROLE_SUPPORT, ROLE_DEFENDER)


# ── Tree data model ──────────────────────────────────────────────────

class BTNode:
    """Single behaviour-tree node."""
    __slots__ = ("name", "kind", "children", "params", "description",
                 "active", "rect")

    def __init__(self, name, kind="action", children=None, params=None,
                 description=""):
        self.name = name
        self.kind = kind  # "selector", "sequence", "condition", "action"
        self.children = children or []
        self.params = params or {}
        self.description = description
        self.active = False
        self.rect = QRectF()


# ── Static tree definitions (derived from robot/*.py) ────────────────

def _goalie_tree():
    return BTNode("Goalie", "selector", [
        BTNode("Clear / Distribute", "sequence", [
            BTNode("Ball in box?", "condition",
                   description="ball_in_box AND ball_slow AND d_ball < CLEAR_DIST"),
            BTNode("Ball in kick range?", "condition",
                   description="d_ball < KICK_DIST AND rel_ball.x > 0"),
            BTNode("Find pass target", "action",
                   description="Scan teammates for clear passing lane"),
            BTNode("Smart distribute", "selector", [
                BTNode("Pass to mate", "action",
                       params={"PASS_CLEAR": 400},
                       description="Kick to teammate with best lane"),
                BTNode("Clear to sideline", "action",
                       description="Emergency clear toward nearest sideline"),
            ]),
        ]),
        BTNode("Save Shot", "sequence", [
            BTNode("Shot incoming?", "condition",
                   params={"SHOT_SPEED": 500},
                   description="Ball velocity toward goal > threshold"),
            BTNode("Predict crossing", "action",
                   description="Compute predicted Y at goal line (incl. wall bounces)"),
            BTNode("Dive to intercept", "action",
                   params={"SAVE_SPEED": 2.5},
                   description="Sprint to predicted crossing point"),
        ]),
        BTNode("Position", "sequence", [
            BTNode("Compute angle", "action",
                   description="Narrow shooting angle based on ball position"),
            BTNode("Detect danger", "action",
                   description="Find most dangerous attacker, bias positioning"),
            BTNode("Move to arc", "action",
                   params={"MAX_ADVANCE": 1100, "POSITION_SPEED": 1.6},
                   description="Position on arc between ball and goal center"),
        ]),
    ])


def _striker_tree():
    return BTNode("Striker", "selector", [
        BTNode("Wait outside box", "sequence", [
            BTNode("Ball in defense area?", "condition",
                   description="Ball inside opponent penalty box"),
            BTNode("Hold position", "action",
                   description="Wait just outside defense area"),
        ]),
        BTNode("One-Touch Redirect", "sequence", [
            BTNode("Ball moving toward me?", "condition",
                   params={"ONETOUCH_MIN_SPEED": 300},
                   description="Ball velocity component toward robot"),
            BTNode("Close enough?", "condition",
                   description="d_ball < BALL_NEAR * 1.5"),
            BTNode("Aligned with goal?", "condition",
                   params={"ONETOUCH_ANGLE": 0.8},
                   description="Aim angle within tolerance"),
            BTNode("Intercept & redirect", "action",
                   params={"ONETOUCH_SPEED": 1.6},
                   description="Move into ball path, kick toward goal"),
        ]),
        BTNode("Possess Ball", "sequence", [
            BTNode("In kick range?", "condition",
                   params={"KICK_RANGE": 175},
                   description="d_ball < KICK_RANGE AND front-facing"),
            BTNode("Decide action", "selector", [
                BTNode("Emergency flick", "action",
                       params={"PRESSURE_DIST": 500},
                       description="Under pressure → blast toward goal"),
                BTNode("Shoot on goal", "action",
                       description="Aligned → kick"),
                BTNode("Dribble", "action",
                       params={"DRIBBLE_SPEED": 1.0},
                       description="Hold ball, turn toward aim"),
            ]),
        ]),
        BTNode("Charge", "sequence", [
            BTNode("Lined up?", "condition",
                   description="Behind ball, facing aim, close range"),
            BTNode("Drive through", "action",
                   params={"CHARGE_SPEED": 1.4},
                   description="Sprint forward through ball"),
        ]),
        BTNode("Recover", "sequence", [
            BTNode("Close but wrong angle?", "condition",
                   description="d_ball < BALL_NEAR but not aligned"),
            BTNode("Go behind ball", "action",
                   description="Navigate to behind-ball point"),
        ]),
        BTNode("Position / Approach", "sequence", [
            BTNode("Navigate to behind", "action",
                   description="Go to point behind ball toward goal"),
            BTNode("Curve if wrong side", "action",
                   params={"AVOID_RADIUS": 400, "BEHIND_DIST": 300},
                   description="Swing around ball if on wrong side"),
        ]),
    ])


def _team_attacker_tree():
    return BTNode("Attacker (Team)", "selector", [
        BTNode("Ball in box → wait", "sequence", [
            BTNode("Ball in defense area?", "condition"),
            BTNode("Wait outside", "action"),
        ]),
        BTNode("Possess Ball", "sequence", [
            BTNode("In kick range?", "condition",
                   params={"KICK_RANGE": 175}),
            BTNode("Evaluate options", "action",
                   description="Score shot vs pass vs through-ball"),
            BTNode("Execute", "selector", [
                BTNode("Shoot", "action",
                       params={"SHOOT_THRESH": 0.16},
                       description="High shot score → kick at goal"),
                BTNode("Pass", "action",
                       params={"PASS_THRESH": 0.05},
                       description="Best pass score → kick to teammate"),
                BTNode("Through-ball", "action",
                       params={"THROUGH_LEAD": 800},
                       description="Pass into space ahead of runner"),
                BTNode("Emergency flick", "action",
                       description="Under pressure → blast away"),
                BTNode("Dribble forward", "action",
                       params={"SPD_DRIBBLE": 1.1}),
            ]),
        ]),
        BTNode("Approach ball", "sequence", [
            BTNode("Intercept point", "action",
                   description="Friction-aware optimal intercept"),
            BTNode("Line up behind", "action",
                   params={"BEHIND_DIST": 270}),
            BTNode("Sprint / cruise / approach", "action",
                   params={"SPD_SPRINT": 2.5, "SPD_CRUISE": 1.8}),
        ]),
    ])


def _support_tree():
    return BTNode("Support", "sequence", [
        BTNode("Compute target", "action",
               description="Scored potential-field candidates"),
        BTNode("Check pass lane", "condition",
               params={"PASS_LANE_CLR": 320},
               description="Lane to ball must be clear of opponents"),
        BTNode("Evaluate position", "action",
               description="Score: advance, shot potential, spacing, lane clarity"),
        BTNode("Move to target", "action",
               params={"SPD_POSITION": 1.5, "SPD_CRUISE": 1.8},
               description="Navigate to best support position, face ball"),
    ])


def _defender_tree():
    return BTNode("Defender", "selector", [
        BTNode("Press (when attacking half)", "sequence", [
            BTNode("Ball in opp territory?", "condition",
                   params={"PRESS_ZONE": 0.3}),
            BTNode("First presser → ball carrier", "action",
                   params={"SPD_PRESS": 2.0}),
            BTNode("Second presser → cut lane", "action",
                   params={"PRESS_DIST": 600}),
        ]),
        BTNode("Mark opponent", "sequence", [
            BTNode("Find nearest threat", "action",
                   description="Sort opponents by distance to our goal"),
            BTNode("Position between opp & goal", "action",
                   params={"DEFEND_MARK_RATIO": 0.48}),
        ]),
        BTNode("Hold shape", "action",
               params={"DEFEND_LINE": 1800, "DEFEND_SPREAD": 1400},
               description="Fall back to defensive line, spread across width"),
    ])


def _goalie_team_tree():
    return BTNode("Goalie (Team)", "selector", [
        BTNode("Clear / Distribute", "sequence", [
            BTNode("Ball in box & slow?", "condition"),
            BTNode("Smart pass or clear", "action",
                   description="Find teammate or clear to sideline"),
        ]),
        BTNode("Save", "sequence", [
            BTNode("Shot with wall bounce?", "condition",
                   params={"GK_SHOT_SPEED": 480}),
            BTNode("Dive to predicted Y", "action",
                   params={"SPD_SAVE": 2.5}),
        ]),
        BTNode("Position on arc", "action",
               params={"GK_MAX_ADV": 1100},
               description="Angle narrowing with danger-weighted bias"),
    ])


ROLE_TREES = {
    "Goalie": _goalie_tree,
    "Striker": _striker_tree,
    "Team Attacker": _team_attacker_tree,
    "Support": _support_tree,
    "Defender": _defender_tree,
    "Team Goalie": _goalie_team_tree,
}

# ── Kind → colors ────────────────────────────────────────────────────
KIND_COLORS = {
    "selector": "#e67e22",
    "sequence": "#2ecc71",
    "condition": "#3498db",
    "action": "#9b59b6",
}

KIND_SYMBOLS = {
    "selector": "?",
    "sequence": "→",
    "condition": "◆",
    "action": "●",
}


# ── Tree rendering widget ────────────────────────────────────────────

NODE_W = 200
NODE_H = 48
H_GAP  = 28
V_GAP  = 72
CANVAS_PADDING = 32


def _layout_tree(node: BTNode, x=0, depth=0) -> float:
    """Recursive layout — returns total width consumed."""
    if not node.children:
        node.rect = QRectF(x, depth * (NODE_H + V_GAP), NODE_W, NODE_H)
        return NODE_W + H_GAP

    child_x = x
    total = 0
    for child in node.children:
        w = _layout_tree(child, child_x, depth + 1)
        child_x += w
        total += w

    left = node.children[0].rect.center().x()
    right = node.children[-1].rect.center().x()
    cx = (left + right) / 2
    node.rect = QRectF(cx - NODE_W / 2, depth * (NODE_H + V_GAP),
                       NODE_W, NODE_H)
    return total


class _TreeCanvas(QWidget):
    """Draws a single behaviour tree with zoom and pan."""

    node_selected = Signal(object)  # BTNode

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root: BTNode | None = None
        self._hovered: BTNode | None = None
        self._selected: BTNode | None = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._base_size = QSize(800, 500)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

    def set_tree(self, root: BTNode):
        self._root = root
        self._selected = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        if root:
            _layout_tree(root, CANVAS_PADDING, 0)
            br = self._bounding_rect(root)
            self._base_size = QSize(int(br.width()) + CANVAS_PADDING * 2,
                                    int(br.height()) + CANVAS_PADDING * 2)
            self.setMinimumSize(
                min(500, self._base_size.width() + 80),
                min(400, self._base_size.height() + 80))
        self.update()

    def _bounding_rect(self, node: BTNode) -> QRectF:
        r = QRectF(node.rect)
        for c in node.children:
            r = r.united(self._bounding_rect(c))
        return r

    def _view_transform(self) -> QTransform:
        t = QTransform()
        t.translate(self.width() / 2 + self._offset.x(),
                    self.height() / 2 + self._offset.y())
        t.scale(self._scale, self._scale)
        t.translate(-self._base_size.width() / 2, -self._base_size.height() / 2)
        return t

    def _widget_to_canvas(self, pos: QPointF) -> QPointF:
        inv, ok = self._view_transform().inverted()
        return inv.map(pos) if ok else pos

    def paintEvent(self, ev: QPaintEvent):
        if not self._root:
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(BG_DARK))
            p.setPen(QColor(TEXT_DIM))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect(), Qt.AlignCenter, "Select a role to view its behavior tree")
            p.end()
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(BG_DARK))
        p.setTransform(self._view_transform())
        self._draw_node(p, self._root)
        p.end()

    def _draw_node(self, p: QPainter, node: BTNode):
        r = node.rect
        color = QColor(KIND_COLORS.get(node.kind, "#888"))

        if node.active:
            color = QColor(SUCCESS)
        elif node is self._hovered:
            color = color.lighter(125)
        if node is self._selected:
            color = color.lighter(115)
            border_color = QColor(ACCENT)
        else:
            border_color = color.darker(150)

        # Connections to children (smooth lines)
        pen = QPen(QColor(BORDER), 2)
        p.setPen(pen)
        for child in node.children:
            p.drawLine(r.center().x(), r.bottom(),
                       child.rect.center().x(), child.rect.top())

        # Node rectangle with subtle rounding
        path = QPainterPath()
        path.addRoundedRect(r, 10, 10)
        p.fillPath(path, QBrush(color))
        p.setPen(QPen(border_color, 2 if node is self._selected else 1.5))
        p.drawPath(path)

        # Symbol + text
        p.setPen(QPen(QColor("#ffffff"), 1))
        sym = KIND_SYMBOLS.get(node.kind, "")
        font = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(font)
        text = f"{sym} {node.name}"
        p.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft,
                   text)

        # Param indicator
        if node.params:
            p.setPen(QPen(QColor("#fff8dc"), 1))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(r.adjusted(0, 0, -8, -4), Qt.AlignBottom | Qt.AlignRight,
                       f"⋮ {len(node.params)}")

        for child in node.children:
            self._draw_node(p, child)

    def _hit_test(self, pos: QPointF, node: BTNode) -> BTNode | None:
        if node.rect.contains(pos):
            return node
        for c in node.children:
            hit = self._hit_test(pos, c)
            if hit:
                return hit
        return None

    def _node_at_widget_pos(self, pos: QPointF) -> BTNode | None:
        if not self._root:
            return None
        canvas_pt = self._widget_to_canvas(pos)
        return self._hit_test(canvas_pt, self._root)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if not self._root:
            return
        hit = self._node_at_widget_pos(QPointF(ev.position()))
        if hit != self._hovered:
            self._hovered = hit
            self.update()
        if hit:
            tip = hit.name
            if hit.description:
                tip += "\n\n" + hit.description
            QToolTip.showText(ev.globalPosition().toPoint(), tip)

    def mousePressEvent(self, ev: QMouseEvent):
        if not self._root or ev.button() != Qt.LeftButton:
            return
        hit = self._node_at_widget_pos(QPointF(ev.position()))
        if hit:
            self._selected = hit
            self.node_selected.emit(hit)
            self.update()

    def wheelEvent(self, ev: QWheelEvent):
        if not self._root:
            return
        delta = ev.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._scale = max(0.4, min(2.0, self._scale * factor))
        self.update()
        ev.accept()

    def zoom_in(self):
        self._scale = min(2.0, self._scale * 1.2)
        self.update()

    def zoom_out(self):
        self._scale = max(0.4, self._scale / 1.2)
        self.update()

    def zoom_fit(self):
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self.update()


# ── Parameter editor ─────────────────────────────────────────────────

class _ParamEditor(QWidget):
    """Side panel to view and edit the selected node's parameters and description."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(400)
        self._spinners = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Card frame for the whole editor
        card = QFrame()
        card.setObjectName("btEditorCard")
        card.setStyleSheet(f"""
            QFrame#btEditorCard {{
                background: {BG_PANEL};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 0;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        self._header = QLabel("Click a node to view or edit its parameters")
        self._header.setWordWrap(True)
        self._header.setStyleSheet(f"""
            color: {TEXT_DIM};
            font-size: 12px;
            padding: 4px 0;
            line-height: 1.35;
        """)
        self._header.setMinimumHeight(44)
        card_layout.addWidget(self._header)

        # Scrollable param grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(0, 4, 0, 0)
        scroll.setWidget(self._grid_widget)
        card_layout.addWidget(scroll, 1)

        outer.addWidget(card)

    def show_node(self, node: BTNode):
        # Clear old widgets
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._spinners.clear()

        kind_label = node.kind.replace("_", " ").capitalize()
        desc = (node.description or "No description.").strip()
        self._header.setText(
            f"<div style='color:{ACCENT}; font-weight:bold; font-size:13px; margin-bottom:4px;'>{node.name}</div>"
            f"<div style='color:{TEXT_DIM}; font-size:11px;'>{kind_label}</div>"
            f"<div style='color:{TEXT_DIM}; font-size:11px; margin-top:6px;'>{desc}</div>")

        if not node.params:
            return

        row = 0
        for key, val in node.params.items():
            lbl = QLabel(key)
            lbl.setStyleSheet(f"color: {TEXT}; font-weight: bold; font-size: 12px;")
            lbl.setToolTip(key)
            spin = QDoubleSpinBox()
            spin.setRange(-99999, 99999)
            spin.setDecimals(3)
            spin.setValue(float(val))
            spin.setSingleStep(0.1 if abs(val) < 10 else 10)
            spin.setMinimumHeight(28)
            self._grid.addWidget(lbl, row, 0)
            self._grid.addWidget(spin, row, 1)
            self._spinners[key] = spin
            row += 1


# ── Composite widget ─────────────────────────────────────────────────

class BehaviorTreePanel(QWidget):
    """Full behavior-tree panel with role selector, tree view, and param editor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Canvas first (needed for zoom button connections)
        self._canvas = _TreeCanvas()

        # Header bar: title + role + zoom
        hdr = QHBoxLayout()
        hdr.setSpacing(16)
        title = QLabel("Behavior Tree")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {ACCENT};")
        hdr.addWidget(title)

        hdr.addWidget(QLabel("Role:"))
        self._role_combo = QComboBox()
        self._role_combo.addItems(list(ROLE_TREES.keys()))
        self._role_combo.setMinimumWidth(140)
        self._role_combo.currentTextChanged.connect(self._on_role_changed)
        hdr.addWidget(self._role_combo)

        hdr.addSpacing(20)
        zoom_lbl = QLabel("Zoom:")
        zoom_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        hdr.addWidget(zoom_lbl)
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedSize(28, 28)
        zoom_out_btn.setToolTip("Zoom out")
        zoom_out_btn.clicked.connect(self._canvas.zoom_out)
        hdr.addWidget(zoom_out_btn)
        zoom_fit_btn = QPushButton("Fit")
        zoom_fit_btn.setFixedSize(36, 28)
        zoom_fit_btn.setToolTip("Reset zoom")
        zoom_fit_btn.clicked.connect(self._canvas.zoom_fit)
        hdr.addWidget(zoom_fit_btn)
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(28, 28)
        zoom_in_btn.setToolTip("Zoom in")
        zoom_in_btn.clicked.connect(self._canvas.zoom_in)
        hdr.addWidget(zoom_in_btn)

        hdr.addStretch()

        # Legend (compact)
        legend = QHBoxLayout()
        legend.setSpacing(14)
        for kind, col in KIND_COLORS.items():
            sym = KIND_SYMBOLS[kind]
            legend.addWidget(QLabel(
                f'<span style="color:{col}; font-weight:bold;">{sym}</span> '
                f'<span style="color:{TEXT_DIM}; font-size:11px;">{kind.capitalize()}</span>'))
        legend.addStretch()
        hdr.addLayout(legend)
        lay.addLayout(hdr)

        # Splitter: tree (scroll) | param editor
        splitter = QSplitter(Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        scroll.setWidget(self._canvas)
        scroll.setMinimumWidth(400)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(scroll)

        self._editor = _ParamEditor()
        splitter.addWidget(self._editor)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 280])  # initial split
        lay.addWidget(splitter, 1)

        self._canvas.node_selected.connect(self._editor.show_node)

        # Load default
        self._on_role_changed(self._role_combo.currentText())

    def _on_role_changed(self, role_name):
        factory = ROLE_TREES.get(role_name)
        if factory:
            tree = factory()
            self._canvas.set_tree(tree)
