"""Qt canvas for inspecting toggleable world-map debug layers."""

import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from TeamControl.ui.field_canvas import FieldCanvas
from TeamControl.ui.theme import BG_MID, BORDER, TEXT_DIM
from TeamControl.world.map.renderer import MapRenderData, RenderLayer


class MapCanvas(FieldCanvas):
    """Field canvas that renders serializable map layers."""

    layers_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._render_data = MapRenderData(layers=())
        self._layer_visibility: dict[str, bool] = {}

    def set_render_data(self, render_data: MapRenderData):
        self._render_data = render_data
        layer_names = {layer.name for layer in render_data.layers}
        for layer in render_data.layers:
            self._layer_visibility.setdefault(layer.name, layer.visible_by_default)
        self._layer_visibility = {
            name: visible
            for name, visible in self._layer_visibility.items()
            if name in layer_names
        }
        self.layers_changed.emit(render_data.layers)
        self.update()

    def set_layer_visible(self, name: str, visible: bool):
        self._layer_visibility[name] = visible
        self.update()

    def is_layer_visible(self, name: str) -> bool:
        return self._layer_visibility.get(name, False)

    def _draw_overlays(self, p: QPainter):
        for layer in self._render_data.layers:
            if self.is_layer_visible(layer.name):
                self._draw_layer(p, layer)

    def _draw_layer(self, p: QPainter, layer: RenderLayer):
        for robot in layer.robots:
            self._draw_robot_body(
                p,
                robot.center_mm[0],
                robot.center_mm[1],
                robot.orientation_rad,
                QColor(robot.color),
            )

        for polyline in layer.polylines:
            points = list(polyline.points_mm)
            if polyline.closed and points:
                points.append(points[0])
            p.setPen(QPen(QColor(polyline.color), 12))
            p.setBrush(Qt.NoBrush)
            for start, end in zip(points, points[1:]):
                p.drawLine(QPointF(*start), QPointF(*end))

        for circle in layer.circles:
            color = QColor(circle.color)
            p.setPen(QPen(color, 12))
            p.setBrush(QBrush(color) if circle.filled else Qt.NoBrush)
            p.drawEllipse(QPointF(*circle.center_mm), circle.radius_mm, circle.radius_mm)

        for vector in layer.vectors:
            self._draw_vector(p, vector.start_mm, vector.end_mm, vector.color)

    def _draw_vector(self, p: QPainter, start, end, color: str):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return

        p.setPen(QPen(QColor(color), 14))
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(*start), QPointF(*end))

        angle = math.atan2(dy, dx)
        head = min(100.0, max(35.0, length * 0.25))
        for offset in (math.pi * 0.82, -math.pi * 0.82):
            p.drawLine(
                QPointF(*end),
                QPointF(
                    end[0] + math.cos(angle + offset) * head,
                    end[1] + math.sin(angle + offset) * head,
                ),
            )


class MapDebugWidget(QWidget):
    """Map canvas with checkboxes generated from the available layers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QWidget()
        controls.setStyleSheet(
            f"background:{BG_MID}; border:1px solid {BORDER}; padding:4px;"
        )
        self._controls_layout = QHBoxLayout(controls)
        self._controls_layout.setContentsMargins(8, 2, 8, 2)
        self._controls_layout.addWidget(QLabel("Map layers:"))
        self._controls_layout.addStretch()
        self._checks: dict[str, QCheckBox] = {}

        self.canvas = MapCanvas()
        self.canvas.layers_changed.connect(self._sync_layer_controls)

        note = QLabel(
            "Tracked world-map view. Velocity arrows show 250 ms of travel; "
            "predicted clearance includes observation age."
        )
        note.setStyleSheet(f"color:{TEXT_DIM};")

        layout.addWidget(controls)
        layout.addWidget(note)
        layout.addWidget(self.canvas, 1)

    def set_render_data(self, render_data: MapRenderData):
        self.canvas.set_render_data(render_data)

    def set_field_geometry(self, field):
        self.canvas.set_field_geometry(field)

    def _sync_layer_controls(self, layers):
        for layer in layers:
            if layer.name in self._checks:
                continue
            check = QCheckBox(layer.name)
            check.setChecked(self.canvas.is_layer_visible(layer.name))
            check.toggled.connect(
                lambda checked, name=layer.name: self.canvas.set_layer_visible(
                    name, checked
                )
            )
            self._controls_layout.insertWidget(
                self._controls_layout.count() - 1,
                check,
            )
            self._checks[layer.name] = check

        active_names = {layer.name for layer in layers}
        for name, check in tuple(self._checks.items()):
            if name not in active_names:
                self._controls_layout.removeWidget(check)
                check.deleteLater()
                del self._checks[name]
