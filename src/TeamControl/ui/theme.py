"""Dark theme stylesheet and color palette for the TurtleRabbit UI."""

from PySide6.QtGui import QColor

# ── Palette ──────────────────────────────────────────────────────────
BG_DARK      = "#111118"
BG_MID       = "#1a1a28"
BG_PANEL     = "#222236"
BG_INPUT     = "#181824"
BG_CARD      = "#1e1e30"
ACCENT       = "#e94560"
ACCENT_HOVER = "#ff6b81"
ACCENT_DIM   = "#b33550"
TEXT         = "#eaeaea"
TEXT_DIM     = "#8a8a9a"
BORDER       = "#2e2e48"
SUCCESS      = "#2ecc71"
WARNING      = "#f39c12"
DANGER       = "#e74c3c"

YELLOW_TEAM  = "#ffd700"
BLUE_TEAM    = "#4a90d9"
BALL_COLOR   = "#ff8c00"
FIELD_GREEN  = "#1b5e20"
FIELD_LINE   = "#ffffff"

ROLE_GOALIE   = "#e74c3c"
ROLE_ATTACKER = "#e67e22"
ROLE_SUPPORT  = "#2ecc71"
ROLE_DEFENDER = "#3498db"

QSS = f"""
* {{
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background: {BG_DARK};
}}

/* ── Top tab bar (main navigation) ─────────────────────────────── */
QTabWidget#mainTabs::pane {{
    border: none;
    background: {BG_DARK};
}}
QTabWidget#mainTabs > QTabBar {{
    background: {BG_MID};
}}
QTabWidget#mainTabs > QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 12px 28px;
    border: none;
    border-bottom: 3px solid transparent;
    font-size: 14px;
    font-weight: 600;
    min-width: 100px;
}}
QTabWidget#mainTabs > QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 3px solid {ACCENT};
}}
QTabWidget#mainTabs > QTabBar::tab:hover {{
    color: {ACCENT_HOVER};
    background: rgba(233, 69, 96, 0.06);
}}

/* ── Generic sub-tabs ──────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG_DARK};
    border-radius: 0 0 6px 6px;
}}
QTabBar::tab {{
    background: {BG_MID};
    color: {TEXT_DIM};
    padding: 8px 18px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background: {BG_DARK};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{
    color: {ACCENT_HOVER};
}}

/* ── Toolbar ───────────────────────────────────────────────────── */
QToolBar {{
    background: {BG_MID};
    border-bottom: 1px solid {BORDER};
    spacing: 8px;
    padding: 6px 12px;
}}

/* ── Menu ──────────────────────────────────────────────────────── */
QMenuBar {{
    background: {BG_MID};
    border-bottom: 1px solid {BORDER};
    padding: 2px;
}}
QMenuBar::item:selected {{
    background: {ACCENT};
    border-radius: 4px;
}}
QMenu {{
    background: {BG_MID};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item:selected {{
    background: {ACCENT};
}}

/* ── Buttons ───────────────────────────────────────────────────── */
QPushButton {{
    background: {BG_PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {ACCENT_HOVER};
}}
QPushButton:disabled {{
    background: {BG_MID};
    color: {TEXT_DIM};
}}
QPushButton#startBtn {{
    background: {SUCCESS};
    color: #111;
    font-weight: bold;
}}
QPushButton#startBtn:hover {{
    background: #27ae60;
}}
QPushButton#stopBtn {{
    background: {DANGER};
    color: #fff;
    font-weight: bold;
}}
QPushButton#stopBtn:hover {{
    background: #c0392b;
}}

/* ── Inputs ────────────────────────────────────────────────────── */
QComboBox {{
    background: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 10px;
    min-width: 100px;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {BG_MID};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 10px;
    color: {TEXT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

/* ── Tables / Trees ────────────────────────────────────────────── */
QTreeWidget, QTableWidget {{
    background: {BG_DARK};
    border: 1px solid {BORDER};
    alternate-background-color: {BG_MID};
    gridline-color: {BORDER};
    border-radius: 6px;
}}
QTreeWidget::item:selected, QTableWidget::item:selected {{
    background: {ACCENT};
}}
QHeaderView::section {{
    background: {BG_MID};
    border: 1px solid {BORDER};
    padding: 5px 10px;
    font-weight: bold;
}}

/* ── Text areas ────────────────────────────────────────────────── */
QPlainTextEdit {{
    background: #0b0b16;
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    color: {TEXT};
}}

/* ── Scrollbars ────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_DARK};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BG_DARK};
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── GroupBox ──────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 10px;
    padding: 16px 10px 10px 10px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {ACCENT};
}}

/* ── Splitter ─────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {BORDER};
    width: 2px;
    height: 2px;
}}

/* ── StatusBar ────────────────────────────────────────────────── */
QStatusBar {{
    background: {BG_MID};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
    font-size: 12px;
    padding: 2px 8px;
}}

/* ── Tooltip (e.g. field hover coordinates) ───────────────────── */
QToolTip {{
    background: #f5f5f5;
    color: #000000;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── CheckBox ─────────────────────────────────────────────────── */
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Slider ───────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: {ACCENT};
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}

/* ── Cards (custom) ───────────────────────────────────────────── */
QFrame#card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 12px;
}}

/* ── Dock (kept as fallback) ──────────────────────────────────── */
QDockWidget {{
    titlebar-close-icon: none;
    font-weight: bold;
    color: {TEXT};
}}
QDockWidget::title {{
    background: {BG_MID};
    border: 1px solid {BORDER};
    padding: 6px 10px;
}}
"""
