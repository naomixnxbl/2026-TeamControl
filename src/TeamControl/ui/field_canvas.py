"""
2-D SSL field rendered with QPainter.

Features:
  - Anti-aliased green pitch with white markings
  - Yellow / blue robots (circles with orientation arrows + ID labels)
  - Orange ball
  - Click-to-place: left-click places ball, right-click queues robot placement
  - Hover tooltip with mm coordinates
  - Zoom / pan via mouse wheel + middle-drag
"""

import math
import time
from PySide6.QtWidgets import QWidget, QToolTip, QMenu, QSizePolicy,QHBoxLayout,QLabel,QPushButton
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QSize
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                           QPainterPath, QTransform, QWheelEvent,
                           QMouseEvent, QPaintEvent, QResizeEvent)

from TeamControl.ui.theme import (FIELD_GREEN, FIELD_LINE, YELLOW_TEAM,
                                   BLUE_TEAM, BALL_COLOR, BG_DARK, ACCENT,
                                   ROLE_GOALIE, ROLE_ATTACKER, ROLE_SUPPORT,
                                   ROLE_DEFENDER, TEXT, BG_PANEL, BG_MID, BORDER)
from TeamControl.robot.constants import (
    CENTER_RADIUS as DEFAULT_CENTER_RADIUS,
    ROBOT_RADIUS,
    FIELD_MARGIN as DEFAULT_MARGIN,
)
from TeamControl.world.field_config import (
    FIELD_LENGTH_MM as DEFAULT_FIELD_LENGTH,
    FIELD_WIDTH_MM as DEFAULT_FIELD_WIDTH,
    DEFENCE_X_MM as DEFAULT_PENALTY_DEPTH,
    DEFENCE_Y_MM as _DEFAULT_PENALTY_HALF_WIDTH,
    GOAL_DEPTH_MM as DEFAULT_GOAL_DEPTH,
    GOAL_WIDTH_MM as DEFAULT_GOAL_WIDTH,
    DASHBOARD_BALL_PLACE_CONFIRM_SECONDS,
    DASHBOARD_BALL_PLACE_CONFIRM_TOLERANCE_MM,
)

# field_config's DEFENCE_Y_MM is a half-width; this default wants the
# total penalty-box width (matches the SSL-Vision geometry field this
# defaults for).
DEFAULT_PENALTY_WIDTH = 2.0 * _DEFAULT_PENALTY_HALF_WIDTH


class FieldCanvas(QWidget):
    """Interactive 2-D SSL field widget."""

    ball_placed = Signal(float, float)            # x_mm, y_mm
    robot_placed = Signal(int, bool, float, float)  # id, yellow, x, y
    point_picked = Signal(float, float)           # x_mm, y_mm — for go-to-point
    action_requested = Signal(str)                # action name
    coordinate_hover = Signal(float, float)       # x_mm, y_mm
    robot_selected = Signal(bool, int)            # is_yellow, robot_id — left-click on robot

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Data
        self._yellow: list = []
        self._blue: list = []
        self._ball = None
        self._targets: list[tuple] = []
        self._ball_place_marker: tuple[float, float] | None = None
        self._ball_place_marker_seen_since_s: float | None = None
        self._paths: list[list[tuple]] = []
        self._frame_number = 0
        self._field_geometry = None


        # View transform
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPointF()

        # Placement state (set by sim panel)
        self._place_mode = None   # "ball", ("robot", id, yellow)

        # Velocity overlay
        self._show_velocity = False
        self._vel_prev: dict = {}   # (team, id) -> (x, y, timestamp)
        self._velocities: dict = {} # (team, id) -> (vx_mm_s, vy_mm_s)

        # Zoom overlay (built last so it sits on top)
        self._build_zoom_overlay()

    # ── Public API ────────────────────────────────────────────────

    def set_show_velocity(self, enabled: bool):
        self._show_velocity = enabled
        self.update()

    def set_frame(self, snap):
        self._yellow = [r for r in snap.yellow if r is not None]
        self._blue = [r for r in snap.blue if r is not None]
        self._ball = snap.ball
        self._frame_number = snap.frame_number
        self._update_ball_place_marker_confirmation()
        self.update()


    def set_targets(self, targets):
        self._targets = list(targets)
        self.update()

    def set_ball_place_marker(self, x_mm: float | None, y_mm: float | None = None):
        if x_mm is None or y_mm is None:
            self._ball_place_marker = None
        else:
            self._ball_place_marker = (float(x_mm), float(y_mm))
        self._ball_place_marker_seen_since_s = None
        self.update()

    def set_paths(self, paths):
        self._paths = list(paths)
        self.update()

    def set_field_geometry(self, field):
        """Use the latest SSL-Vision geometry, falling back to local defaults."""
        self._field_geometry = field
        self.update()

    def set_place_mode(self, mode):
        self._place_mode = mode
        if mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    # ── Zoom overlay ──────────────────────────────────────────────

    def _build_zoom_overlay(self):
        self._zoom_overlay = QWidget(self)
        self._zoom_overlay.setStyleSheet(
            f"QWidget {{ background: {BG_MID}; border: 1px solid {BORDER}; border-radius: 5px; }}"
        )

        lay = QHBoxLayout(self._zoom_overlay)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        icon = QLabel("🔍")
        icon.setStyleSheet(f"background: transparent; border: none; font-size: 14px; color: {TEXT};")
        lay.addWidget(icon)

        btn_style = (
            f"QPushButton {{ background: {BG_PANEL}; color: {TEXT}; border: none;"
            f" border-radius: 3px; font-size: 16px; font-weight: bold; padding: 0px; }}"
            f"QPushButton:hover {{ background: {ACCENT}; color: white; }}"
            f"QPushButton:pressed {{ background: {ACCENT}; color: white; }}"
        )
        for text, tip, slot in [
            ("+", "Zoom In",     self._zoom_in),
            ("-", "Zoom Out",    self._zoom_out),
            ("↺", "Fit to View", self._zoom_fit),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(32, 28)
            btn.setToolTip(tip)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(slot)
            lay.addWidget(btn)

        self._zoom_overlay.adjustSize()

    def _zoom_in(self):
        self._scale = min(5.0, self._scale * 1.2)
        self.update()

    def _zoom_out(self):
        self._scale = max(0.2, self._scale * 0.8)
        self.update()

    def _zoom_fit(self):
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self.update()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._zoom_overlay.adjustSize()
        margin = 10
        self._zoom_overlay.move(
            self.width() - self._zoom_overlay.width() - margin, margin)
        self._zoom_overlay.raise_()

    # ── Coordinate transforms ─────────────────────────────────────

    def _view_transform(self) -> QTransform:
        w, h = self.width(), self.height()
        total_w = self._field_length() + 2 * self._margin() + 2 * self._goal_depth()
        total_h = self._field_width() + 2 * self._margin()
        sx = w / total_w * self._scale
        sy = h / total_h * self._scale
        s = min(sx, sy)

        t = QTransform()
        t.translate(w / 2 + self._offset.x(), h / 2 + self._offset.y())
        t.scale(s, -s)  # flip Y so positive Y is up
        return t

    def _widget_to_field(self, pos: QPointF) -> QPointF:
        inv, ok = self._view_transform().inverted()
        if ok:
            return inv.map(pos)
        return QPointF(0, 0)

    # ── Painting ──────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        p.fillRect(self.rect(), QColor(BG_DARK))

        p.setTransform(self._view_transform())
        self._draw_field(p)
        self._draw_targets(p)
        self._draw_paths(p)
        self._draw_robots(p, self._yellow, QColor(YELLOW_TEAM))
        self._draw_robots(p, self._blue, QColor(BLUE_TEAM))
        self._draw_ball(p)
        self._draw_overlays(p)

        # Frame counter overlay
        p.resetTransform()
        p.setPen(QColor(TEXT))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(10, self.height() - 10, f"Frame: {self._frame_number}")
        p.end()

    def _draw_field(self, p: QPainter):
        field_length = self._field_length()
        field_width = self._field_width()
        margin = self._margin()
        goal_depth = self._goal_depth()
        goal_width = self._goal_width()
        penalty_depth = self._penalty_depth()
        penalty_width = self._penalty_width()
        half_len = field_length / 2
        half_wid = field_width / 2
        outer = QRectF(-(half_len + margin + goal_depth),
                       -(half_wid + margin),
                       field_length + 2 * margin + 2 * goal_depth,
                       field_width + 2 * margin)
        p.fillRect(outer, QColor(FIELD_GREEN))

        pen = QPen(QColor(FIELD_LINE), 20)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Outer boundary
        p.drawRect(QRectF(-half_len, -half_wid, field_length, field_width))

        # Center line
        p.drawLine(QPointF(0, -half_wid), QPointF(0, half_wid))

        # Center circle
        center_radius = self._center_radius()
        p.drawEllipse(QPointF(0, 0), center_radius, center_radius)

        # Center dot
        p.setBrush(QColor(FIELD_LINE))
        p.drawEllipse(QPointF(0, 0), 20, 20)
        p.setBrush(Qt.NoBrush)

        # Left penalty area
        ph = penalty_width / 2
        p.drawRect(QRectF(-half_len, -ph, penalty_depth, penalty_width))

        # Right penalty area
        p.drawRect(QRectF(half_len - penalty_depth, -ph,
                          penalty_depth, penalty_width))

        # Left goal
        gh = goal_width / 2
        goal_pen = QPen(QColor("#cccccc"), 16)
        p.setPen(goal_pen)
        p.drawRect(QRectF(-half_len - goal_depth, -gh,
                          goal_depth, goal_width))

        # Right goal
        p.drawRect(QRectF(half_len, -gh, goal_depth, goal_width))

    def _field_value(self, name, default):
        value = getattr(self._field_geometry, name, None)
        return default if value is None or float(value) <= 0 else float(value)

    def _field_length(self):
        return self._field_value("field_length", DEFAULT_FIELD_LENGTH)

    def _field_width(self):
        return self._field_value("field_width", DEFAULT_FIELD_WIDTH)

    def _margin(self):
        return self._field_value("boundary_width", DEFAULT_MARGIN)

    def _goal_depth(self):
        return self._field_value("goal_depth", DEFAULT_GOAL_DEPTH)

    def _goal_width(self):
        return self._field_value("goal_width", DEFAULT_GOAL_WIDTH)

    def _penalty_depth(self):
        return self._field_value("penalty_area_depth", DEFAULT_PENALTY_DEPTH)

    def _penalty_width(self):
        return self._field_value("penalty_area_width", DEFAULT_PENALTY_WIDTH)

    def _center_radius(self):
        arcs = getattr(self._field_geometry, "field_arcs", ())
        for arc in arcs:
            if "center" in str(getattr(arc, "name", "")).lower():
                return float(arc.radius)
        return DEFAULT_CENTER_RADIUS


    def _draw_robots(self, p: QPainter, robots, color: QColor):
        if not robots:
            return
        for r in robots:
            cx, cy = r.x, r.y
            self._draw_robot_body(p, cx, cy, r.o, color)

            # Orientation arrow
            arr_len = ROBOT_RADIUS * 1.6
            ax = cx + math.cos(r.o) * arr_len
            ay = cy + math.sin(r.o) * arr_len
            arrow_pen = QPen(color.darker(140), 18)
            arrow_pen.setCapStyle(Qt.RoundCap)
            p.setPen(arrow_pen)
            p.drawLine(QPointF(cx, cy), QPointF(ax, ay))

            # Arrowhead
            head = 50
            a1 = r.o + math.pi * 0.82
            a2 = r.o - math.pi * 0.82
            p.drawLine(QPointF(ax, ay),
                       QPointF(ax + math.cos(a1) * head,
                               ay + math.sin(a1) * head))
            p.drawLine(QPointF(ax, ay),
                       QPointF(ax + math.cos(a2) * head,
                               ay + math.sin(a2) * head))

            self._draw_robot_id_label(p, QPointF(cx, cy), str(r.id))

    def _draw_robot_id_label(self, p: QPainter, center: QPointF, text: str):
        """Draw robot IDs in screen pixels so labels stay readable at any zoom."""
        screen_center = p.transform().map(center)
        p.save()
        p.resetTransform()
        font = QFont("Segoe UI", 12, QFont.Bold)
        p.setFont(font)
        rect = QRectF(screen_center.x() - 14, screen_center.y() - 12, 28, 24)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 145))
        p.drawRoundedRect(rect.adjusted(1, 2, -1, -2), 4, 4)
        p.setPen(QPen(QColor("#ffffff"), 1))
        p.drawText(rect, Qt.AlignCenter, text)
        p.restore()

    def _draw_robot_body(self, p: QPainter, cx, cy, theta, color: QColor):
        """Draw the SSL robot footprint: circular rear with a flat kicker front."""
        front_x = ROBOT_RADIUS * 0.72
        cut_angle = math.acos(front_x / ROBOT_RADIUS)
        path = QPainterPath()
        for index in range(25):
            angle = cut_angle + (2 * math.pi - 2 * cut_angle) * index / 24
            local_x = math.cos(angle) * ROBOT_RADIUS
            local_y = math.sin(angle) * ROBOT_RADIUS
            x = cx + math.cos(theta) * local_x - math.sin(theta) * local_y
            y = cy + math.sin(theta) * local_x + math.cos(theta) * local_y
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        p.setPen(QPen(QColor("#111111"), 12))
        p.setBrush(QBrush(color))
        p.drawPath(path)

    def _draw_ball(self, p: QPainter):
        if not self._ball:
            return
        bx, by = self._ball.x, self._ball.y
        p.setPen(QPen(QColor("#000000"), 8))
        p.setBrush(QBrush(QColor(BALL_COLOR)))
        p.drawEllipse(QPointF(bx, by), 45, 45)

    def _update_ball_place_marker_confirmation(self):
        if self._ball_place_marker is None:
            self._ball_place_marker_seen_since_s = None
            return
        if not self._ball:
            self._ball_place_marker_seen_since_s = None
            return

        tx, ty = self._ball_place_marker
        dist = math.hypot(float(self._ball.x) - tx, float(self._ball.y) - ty)
        if dist > DASHBOARD_BALL_PLACE_CONFIRM_TOLERANCE_MM:
            self._ball_place_marker_seen_since_s = None
            return

        now_s = time.monotonic()
        if self._ball_place_marker_seen_since_s is None:
            self._ball_place_marker_seen_since_s = now_s
            return
        if now_s - self._ball_place_marker_seen_since_s >= (
            DASHBOARD_BALL_PLACE_CONFIRM_SECONDS
        ):
            self._ball_place_marker = None
            self._ball_place_marker_seen_since_s = None

    def _draw_overlays(self, p: QPainter):
        """Draw optional subclasses' field-space overlays."""
        return

    def _draw_targets(self, p: QPainter):
        if self._ball_place_marker is not None:
            tx, ty = self._ball_place_marker
            self._draw_target_x(p, tx, ty, BALL_COLOR, size=85, width=18)
        for item in self._targets:
            if len(item) == 3:
                tx, ty, color_hex = item
            else:
                tx, ty = item
                color_hex = ACCENT
            self._draw_target_x(p, tx, ty, color_hex)

    def _draw_target_x(self, p: QPainter, tx, ty, color_hex, size=60, width=16):
        p.setPen(QPen(QColor(color_hex), width))
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(tx - size, ty - size),
                   QPointF(tx + size, ty + size))
        p.drawLine(QPointF(tx - size, ty + size),
                   QPointF(tx + size, ty - size))

    def _draw_paths(self, p: QPainter):
        for path_entry in self._paths:
            if len(path_entry) < 2:
                continue
            # Support (points, color_hex) tuple or plain list of points
            if isinstance(path_entry, tuple) and len(path_entry) == 2 \
               and isinstance(path_entry[1], str):
                points, color_hex = path_entry
            else:
                points = path_entry
                color_hex = ACCENT
            pen = QPen(QColor(color_hex), 10, Qt.DashLine)
            p.setPen(pen)
            for i in range(len(points) - 1):
                p.drawLine(QPointF(*points[i]), QPointF(*points[i + 1]))

    # ── Input events ──────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MiddleButton:
            self._dragging = True
            self._drag_start = ev.position()
            ev.accept()
            return

        if ev.button() == Qt.LeftButton and self._place_mode:
            pt = self._widget_to_field(ev.position())
            if self._place_mode == "ball":
                self.ball_placed.emit(pt.x(), pt.y())
            elif self._place_mode == "go_to_point":
                self.point_picked.emit(pt.x(), pt.y())
            elif isinstance(self._place_mode, tuple):
                _, rid, yellow = self._place_mode
                self.robot_placed.emit(rid, yellow, pt.x(), pt.y())
            self._place_mode = None
            self.setCursor(Qt.ArrowCursor)
            ev.accept()
            return

        if ev.button() == Qt.LeftButton:
            pt = self._widget_to_field(ev.position())
            robot = self._find_robot_at(pt.x(), pt.y())
            if robot is not None:
                self.robot_selected.emit(robot.isYellow, robot.robot_id)
                ev.accept()
                return

        if ev.button() == Qt.RightButton:
            self._show_field_menu(ev)
            ev.accept()
            return

        super().mousePressEvent(ev)

    def _find_robot_at(self, fx: float, fy: float):
        """Return the first robot whose body contains field point (fx, fy), or None."""
        hit_r = ROBOT_RADIUS * 2.0
        for r in self._yellow:
            if math.hypot(r.x - fx, r.y - fy) <= hit_r:
                return r
        for r in self._blue:
            if math.hypot(r.x - fx, r.y - fy) <= hit_r:
                return r
        return None

    def _show_field_menu(self, ev: QMouseEvent):
        pt = self._widget_to_field(ev.position())
        x, y = pt.x(), pt.y()

        menu = QMenu(self)
        go_action = menu.addAction(f"Go to ({x:.0f}, {y:.0f})")
        go_ball_action = menu.addAction("Go to Ball")
        go_ball_kick_action = menu.addAction("Go to Ball && Kick")
        draw_square_action = menu.addAction("Draw Square")
        menu.addSeparator()
        ball_action = menu.addAction("Place ball here")
        menu.addSeparator()
        stop_action = menu.addAction("Stop")

        chosen = menu.exec(ev.globalPosition().toPoint())
        if chosen == go_action:
            self.point_picked.emit(x, y)
        elif chosen == go_ball_action:
            self.action_requested.emit("go_to_ball")
        elif chosen == go_ball_kick_action:
            self.action_requested.emit("go_to_ball_kick")
        elif chosen == draw_square_action:
            self.action_requested.emit("draw_square")
        elif chosen == ball_action:
            self.ball_placed.emit(x, y)
        elif chosen == stop_action:
            self.action_requested.emit("stop")

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MiddleButton:
            self._dragging = False
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._dragging:
            delta = ev.position() - self._drag_start
            self._offset += delta
            self._drag_start = ev.position()
            self.update()
            ev.accept()
            return

        pt = self._widget_to_field(ev.position())
        self.coordinate_hover.emit(pt.x(), pt.y())
        QToolTip.showText(ev.globalPosition().toPoint(),
                          f"({pt.x():.0f}, {pt.y():.0f}) mm",
                          self)
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev: QWheelEvent):
        delta = ev.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._scale = max(0.2, min(5.0, self._scale * factor))
        self.update()
        ev.accept()

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self.update()

    def sizeHint(self):
        return QSize(900, 600)
