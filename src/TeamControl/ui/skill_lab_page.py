"""Skill Lab — pick a behaviour from ``bt/behaviours.py`` and run it live on a
robot (real hardware or grSim) through the normal dispatcher path.

Follows the same conventions as ``test_panel.py``: shares the engine /
dispatcher / manual-override mechanism, and reuses the field canvas for
robot selection and target-point picking. Behaviours are picked from the
same uniform Intent-producing registry the behaviour trees use
(``bt/behaviours.py``), but the Intent -> velocity step intentionally does
NOT go through the BT v2 pipeline (``bt/adapter.py`` /
``bt/skill_intent_executor.py`` with its configurable ``PDMode``/
``ActuatorMode``) — it reuses ``robot/Movement.py`` (``RobotMovement.
velocity_to_target`` / ``get_movement``), the same simple per-robot
movement controller ``robot/goal.py`` and the legacy game roles already use.
"""
import importlib
import json
import math
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QLineEdit,
    QListWidget, QListWidgetItem, QFrame, QPlainTextEdit, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, QTime
from PySide6.QtGui import QColor, QFont

from TeamControl.ui.theme import (
    ACCENT, TEXT_DIM, SUCCESS, DANGER, WARNING, BG_DARK, BORDER,
)
from TeamControl.network.robot_command import RobotCommand
from TeamControl.utils.yaml_config import Config
from TeamControl.world.transform_cords import world2robot
from TeamControl.skills.skills import BEHAVIOURS, BEHAVIOURS_BY_ID, Behaviour, reset_robot_state
from TeamControl.bt.adapter import build_snapshot_from_world_model
from TeamControl.bt.contracts.intent import (
    IntentDribble, IntentKick, IntentMove, IntentOrient, IntentPass, IntentReceive,
)
from TeamControl.robot.Movement import get_movement


_FALLBACK_LETTERS = [chr(ord("A") + i) for i in range(16)]  # used if ipconfig.yaml can't be read
_CUSTOM_SKILLS_PATH = Path(__file__).resolve().parent.parent / "skills" / "custom_skills.json"


def _ts():
    return QTime.currentTime().toString("HH:mm:ss.zzz")


def _heading(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-size:13px; font-weight:bold; color:{ACCENT}; padding:2px 0;")
    return lbl


def _sep():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color:{BORDER};")
    return line


class _LogView(QPlainTextEdit):
    MAX_LINES = 1000

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setFont(QFont("Cascadia Code", 10))

    def ok(self, msg):
        self.appendHtml(f'<span style="color:{TEXT_DIM}">{_ts()}</span> '
                         f'<span style="color:{SUCCESS}">{msg}</span>')

    def err(self, msg):
        self.appendHtml(f'<span style="color:{TEXT_DIM}">{_ts()}</span> '
                         f'<span style="color:{DANGER}">{msg}</span>')

    def info(self, msg):
        self.appendHtml(f'<span style="color:{TEXT_DIM}">{_ts()}</span> '
                         f'<span style="color:#eaeaea">{msg}</span>')


class SkillLabPage(QWidget):
    """Pick a robot + behaviour + (optional) target, then run it live."""

    TICK_MS = 50  # 20 Hz, matches the dispatcher's manual-command path

    def __init__(self, engine=None, field=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._field = field

        self._dashboard_rid: int | None = None
        self._dashboard_yellow = True
        self._target_point_mm: tuple[float, float] | None = None
        self._behaviour: Behaviour | None = None
        self._running = False

        # Robots are keyed by letter (A, B, C…) in ipconfig.yaml, mapping to
        # the numeric shell ID the dispatcher/RobotCommand actually use.
        self._robots_by_team: dict[str, dict[str, int]] = {"Yellow": {}, "Blue": {}}

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

        self._chaotic_robots: dict[int, tuple[Behaviour, bool]] = {}
        self._chaotic_timer = QTimer(self)
        self._chaotic_timer.setInterval(self.TICK_MS)
        self._chaotic_timer.timeout.connect(self._chaotic_tick)

        self._behaviours: list[Behaviour] = list(BEHAVIOURS)
        # custom skills created in the builder; parallel meta dict holds serialisable params
        self._custom_skills: list[Behaviour] = []
        self._custom_skill_meta: dict[str, dict] = {}

        self._build_ui()
        self._load_robots()
        self._load_custom_skills()
        if self._behaviours:
            self._list.setCurrentRow(0)

    # ══════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(16)

        # ── Left column: robot target ───────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(_heading("Robot"))

        rg = QGridLayout()
        rg.setSpacing(8)
        rg.addWidget(QLabel("Team:"), 0, 0)
        self._team_combo = QComboBox()
        self._team_combo.addItems(["Yellow", "Blue"])
        self._team_combo.currentTextChanged.connect(self._on_team_changed)
        rg.addWidget(self._team_combo, 0, 1)
        rg.addWidget(QLabel("Robot:"), 1, 0)
        self._id_combo = QComboBox()
        rg.addWidget(self._id_combo, 1, 1)
        left.addLayout(rg)

        self._robot_src_label = QLabel("Set via the controls above, or click a robot on the field.")
        self._robot_src_label.setWordWrap(True)
        self._robot_src_label.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        left.addWidget(self._robot_src_label)

        left.addWidget(_sep())
        beh_row = QHBoxLayout()
        beh_row.addWidget(_heading("Behaviour"))
        self._reload_btn = QPushButton("↺ Reload")
        self._reload_btn.setFixedHeight(24)
        self._reload_btn.setToolTip("Hot-reload behaviours.py without restarting")
        self._reload_btn.clicked.connect(self._reload_behaviours)
        beh_row.addWidget(self._reload_btn)
        left.addLayout(beh_row)

        self._list = QListWidget()
        for b in self._behaviours:
            item = QListWidgetItem(b.label)
            item.setData(Qt.UserRole, b.id)
            self._list.addItem(item)
        self._list.currentRowChanged.connect(self._on_behaviour_changed)
        self._list.setMinimumWidth(220)
        left.addWidget(self._list, 1)

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; padding:2px;")
        left.addWidget(self._desc_label)

        # ── Custom Skill Builder ────────────────────────────────────
        left.addWidget(_sep())
        left.addWidget(_heading("Custom Skill"))

        cb_row = QHBoxLayout()
        cb_row.setSpacing(6)
        cb_row.addWidget(QLabel("Base:"))
        self._custom_base_combo = QComboBox()
        for b in BEHAVIOURS:
            self._custom_base_combo.addItem(b.label, b.id)
        cb_row.addWidget(self._custom_base_combo, 1)
        left.addLayout(cb_row)

        cl_row = QHBoxLayout()
        cl_row.setSpacing(6)
        cl_row.addWidget(QLabel("Label:"))
        self._custom_label_edit = QLineEdit()
        self._custom_label_edit.setPlaceholderText("Leave blank for auto")
        cl_row.addWidget(self._custom_label_edit, 1)
        left.addLayout(cl_row)

        ctg = QGridLayout()
        ctg.setSpacing(6)
        ctg.addWidget(QLabel("X:"), 0, 0)
        self._custom_tx_spin = QDoubleSpinBox()
        self._custom_tx_spin.setRange(-6000, 6000)
        self._custom_tx_spin.setSuffix(" mm")
        ctg.addWidget(self._custom_tx_spin, 0, 1)
        ctg.addWidget(QLabel("Y:"), 0, 2)
        self._custom_ty_spin = QDoubleSpinBox()
        self._custom_ty_spin.setRange(-4500, 4500)
        self._custom_ty_spin.setSuffix(" mm")
        ctg.addWidget(self._custom_ty_spin, 0, 3)
        left.addLayout(ctg)
        # connect AFTER spinners exist so the signal never fires against missing attrs
        self._custom_base_combo.currentIndexChanged.connect(self._on_custom_base_changed)
        self._on_custom_base_changed(0)   # set initial X/Y enable state

        custom_btns = QHBoxLayout()
        self._add_skill_btn = QPushButton("+ Add Skill")
        self._add_skill_btn.setMinimumHeight(28)
        self._add_skill_btn.clicked.connect(self._add_custom_skill)
        custom_btns.addWidget(self._add_skill_btn)
        self._remove_skill_btn = QPushButton("Remove")
        self._remove_skill_btn.setMinimumHeight(28)
        self._remove_skill_btn.setEnabled(False)
        self._remove_skill_btn.clicked.connect(self._remove_custom_skill)
        custom_btns.addWidget(self._remove_skill_btn)
        left.addLayout(custom_btns)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(310)
        outer.addWidget(left_w)

        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet(f"color:{BORDER};")
        outer.addWidget(div)

        # ── Right column: target / run / readout ────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        right.addWidget(_heading("Target Point"))
        tg = QGridLayout()
        tg.setSpacing(8)
        tg.addWidget(QLabel("X:"), 0, 0)
        self._tx_spin = QDoubleSpinBox()
        self._tx_spin.setRange(-6000, 6000)
        self._tx_spin.setSuffix(" mm")
        self._tx_spin.valueChanged.connect(self._on_target_spin_changed)
        tg.addWidget(self._tx_spin, 0, 1)
        tg.addWidget(QLabel("Y:"), 0, 2)
        self._ty_spin = QDoubleSpinBox()
        self._ty_spin.setRange(-4500, 4500)
        self._ty_spin.setSuffix(" mm")
        self._ty_spin.valueChanged.connect(self._on_target_spin_changed)
        tg.addWidget(self._ty_spin, 0, 3)
        right.addLayout(tg)

        self._pick_btn = QPushButton("Pick on Field…")
        self._pick_btn.setMinimumHeight(32)
        self._pick_btn.clicked.connect(self._pick_on_field)
        right.addWidget(self._pick_btn)

        right.addWidget(_sep())
        right.addWidget(_heading("Run"))

        run_grid = QGridLayout()
        run_grid.setSpacing(8)
        self._run_btn = QPushButton("Run")
        self._run_btn.setObjectName("startBtn")
        self._run_btn.setMinimumHeight(44)
        self._run_btn.setStyleSheet("font-size:14px;")
        self._run_btn.clicked.connect(self._start_skill)
        run_grid.addWidget(self._run_btn, 0, 0)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setMinimumHeight(44)
        self._stop_btn.setStyleSheet("font-size:14px;")
        self._stop_btn.clicked.connect(self._stop_skill)
        run_grid.addWidget(self._stop_btn, 0, 1)

        self._stop_all_btn = QPushButton("STOP ALL")
        self._stop_all_btn.setObjectName("stopBtn")
        self._stop_all_btn.setMinimumHeight(44)
        self._stop_all_btn.setStyleSheet("font-size:14px;")
        self._stop_all_btn.clicked.connect(self._stop_all)
        run_grid.addWidget(self._stop_all_btn, 0, 2)
        right.addLayout(run_grid)

        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px; padding:4px;")
        right.addWidget(self._status_label)

        right.addWidget(_sep())
        right.addWidget(_heading("Team Strategy"))

        cg = QGridLayout()
        cg.setSpacing(8)
        cg.addWidget(QLabel("Team:"), 0, 0)
        self._chaotic_team_combo = QComboBox()
        self._chaotic_team_combo.addItems(["Yellow", "Blue"])
        self._chaotic_team_combo.currentTextChanged.connect(self._on_chaotic_team_changed)
        cg.addWidget(self._chaotic_team_combo, 0, 1)
        cg.addWidget(QLabel("Goalie:"), 0, 2)
        self._chaotic_goalie_combo = QComboBox()
        cg.addWidget(self._chaotic_goalie_combo, 0, 3)
        cg.addWidget(QLabel("Defender:"), 1, 0)
        self._strategy_defender_combo = QComboBox()
        cg.addWidget(self._strategy_defender_combo, 1, 1)
        right.addLayout(cg)

        chaotic_btns = QHBoxLayout()
        self._chaotic_atk_btn = QPushButton("Possession")
        self._chaotic_atk_btn.setMinimumHeight(36)
        self._chaotic_atk_btn.clicked.connect(lambda: self._start_chaotic("attack"))
        chaotic_btns.addWidget(self._chaotic_atk_btn)
        self._chaotic_def_btn = QPushButton("No Possession")
        self._chaotic_def_btn.setMinimumHeight(36)
        self._chaotic_def_btn.clicked.connect(lambda: self._start_chaotic("defend"))
        chaotic_btns.addWidget(self._chaotic_def_btn)
        self._chaotic_stop_btn = QPushButton("Stop")
        self._chaotic_stop_btn.setObjectName("stopBtn")
        self._chaotic_stop_btn.setMinimumHeight(36)
        self._chaotic_stop_btn.clicked.connect(self._stop_chaotic)
        chaotic_btns.addWidget(self._chaotic_stop_btn)
        right.addLayout(chaotic_btns)

        self._chaotic_status = QLabel("Idle")
        self._chaotic_status.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; padding:2px;")
        right.addWidget(self._chaotic_status)

        right.addWidget(_sep())
        right.addWidget(_heading("Live Readout"))
        self._readout_label = QLabel("—")
        self._readout_label.setWordWrap(True)
        self._readout_label.setFont(QFont("Cascadia Code", 10))
        self._readout_label.setStyleSheet(
            f"background:{BG_DARK}; padding:10px; border:1px solid {BORDER}; "
            f"border-radius:6px; color:#eaeaea;")
        self._readout_label.setMinimumHeight(70)
        right.addWidget(self._readout_label)

        right.addWidget(_sep())
        self._log = _LogView()
        self._log.setMaximumHeight(160)
        right.addWidget(self._log)

        right.addStretch()
        outer.addLayout(right, 1)

    # ══════════════════════════════════════════════════════════════
    #  Robot / target selection
    # ══════════════════════════════════════════════════════════════

    def select_robot(self, is_yellow: bool, robot_id: int):
        """Override target robot (called from dashboard / field click)."""
        self._dashboard_rid = robot_id
        self._dashboard_yellow = bool(is_yellow)
        team_name = "Yellow" if is_yellow else "Blue"
        self._team_combo.setCurrentText(team_name)
        letter = self._letter_for_shell_id(team_name, robot_id)
        if letter is not None:
            self._id_combo.setCurrentText(letter)
            self._robot_src_label.setText(f"Target set from field: {team_name} {letter} (shell {robot_id})")
        else:
            self._robot_src_label.setText(
                f"Target set from field: {team_name} shell {robot_id} (no ipconfig.yaml entry)")

    def _letter_for_shell_id(self, team_name: str, shell_id: int) -> str | None:
        for letter, sid in self._robots_by_team.get(team_name, {}).items():
            if sid == shell_id:
                return letter
        return None

    def _load_robots(self):
        try:
            cfg = Config()
        except Exception as e:
            self._log.err(f"Failed to load ipconfig.yaml: {e}")
            cfg = None

        for team_name, team_data in [
            ("Yellow", getattr(cfg, "yellow", None)),
            ("Blue", getattr(cfg, "blue", None)),
        ]:
            mapping: dict[str, int] = {}
            if team_data:
                for letter, rd in team_data.items():
                    mapping[str(letter).upper()] = rd.get("shellID", 0)
            if not mapping:
                mapping = {letter: i for i, letter in enumerate(_FALLBACK_LETTERS)}
            self._robots_by_team[team_name] = mapping

        self._populate_id_combo_for_team(self._team_combo.currentText())
        self._populate_chaotic_goalie_combo(self._chaotic_team_combo.currentText())
        self._populate_strategy_defender_combo(self._chaotic_team_combo.currentText())
        self._log.info(
            f"Robots loaded — Yellow: {sorted(self._robots_by_team['Yellow'])}  "
            f"Blue: {sorted(self._robots_by_team['Blue'])}")

    def _on_team_changed(self, team_name: str):
        self._populate_id_combo_for_team(team_name)

    def _populate_id_combo_for_team(self, team_name: str):
        mapping = self._robots_by_team.get(team_name, {})
        current = self._id_combo.currentText()
        self._id_combo.blockSignals(True)
        self._id_combo.clear()
        for letter in sorted(mapping.keys()):
            self._id_combo.addItem(letter)
        if current in mapping:
            self._id_combo.setCurrentText(current)
        self._id_combo.blockSignals(False)

    def _get_rid_yellow(self) -> tuple[int, bool]:
        team_name = self._team_combo.currentText()
        is_yellow = (team_name == "Yellow")
        letter = self._id_combo.currentText()
        shell_id = self._robots_by_team.get(team_name, {}).get(letter, 0)
        return shell_id, is_yellow

    def _robot_label(self, is_yellow: bool, shell_id: int) -> str:
        team_name = "Yellow" if is_yellow else "Blue"
        letter = self._letter_for_shell_id(team_name, shell_id)
        return f"{team_name} {letter}" if letter else f"{team_name} shell {shell_id}"

    def _pick_on_field(self):
        if self._field is None:
            self._set_status("No field canvas available", DANGER)
            return
        self._field.set_place_mode("skill_target")
        self._set_status("Click a point on the Dashboard field view…", WARNING)

    def set_target_point(self, x_mm: float, y_mm: float):
        """Called when the user clicks the field after Pick on Field."""
        self._target_point_mm = (float(x_mm), float(y_mm))
        self._tx_spin.blockSignals(True)
        self._ty_spin.blockSignals(True)
        self._tx_spin.setValue(x_mm)
        self._ty_spin.setValue(y_mm)
        self._tx_spin.blockSignals(False)
        self._ty_spin.blockSignals(False)
        self._set_status(f"Target set to ({x_mm:.0f}, {y_mm:.0f}) mm", SUCCESS)

    def _on_target_spin_changed(self, _value):
        self._target_point_mm = (self._tx_spin.value(), self._ty_spin.value())

    # ══════════════════════════════════════════════════════════════
    #  Behaviour selection
    # ══════════════════════════════════════════════════════════════

    def _on_behaviour_changed(self, row: int):
        if self._running:
            self._stop_skill()
        if row < 0 or row >= len(self._behaviours):
            self._behaviour = None
            self._desc_label.setText("")
            if hasattr(self, "_remove_skill_btn"):
                self._remove_skill_btn.setEnabled(False)
            return
        self._behaviour = self._behaviours[row]
        self._desc_label.setText(self._behaviour.description)
        needs_target = self._behaviour.needs_target
        self._tx_spin.setEnabled(needs_target)
        self._ty_spin.setEnabled(needs_target)
        self._pick_btn.setEnabled(needs_target)
        if hasattr(self, "_remove_skill_btn"):
            self._remove_skill_btn.setEnabled(self._behaviour.id.startswith("custom_"))

    def _reload_behaviours(self):
        if self._running:
            self._stop_skill()
        try:
            import TeamControl.skills.skills as bmod
            importlib.reload(bmod)
            self._behaviours = list(bmod.BEHAVIOURS) + self._custom_skills
        except Exception as e:
            self._log.err(f"Reload failed: {e}")
            return
        prev_id = self._list.currentItem().data(Qt.UserRole) if self._list.currentItem() else None
        self._list.blockSignals(True)
        self._list.clear()
        for b in self._behaviours:
            if b.id.startswith("custom_"):
                item = QListWidgetItem(f"★ {b.label}")
                item.setForeground(QColor(ACCENT))
            else:
                item = QListWidgetItem(b.label)
            item.setData(Qt.UserRole, b.id)
            self._list.addItem(item)
        self._list.blockSignals(False)
        restore_row = next((i for i, b in enumerate(self._behaviours) if b.id == prev_id), 0)
        self._list.setCurrentRow(restore_row)
        n_base = len(self._behaviours) - len(self._custom_skills)
        self._log.ok(f"Reloaded skills.py — {n_base} built-in + {len(self._custom_skills)} custom")

    # ══════════════════════════════════════════════════════════════
    #  Run / Stop
    # ══════════════════════════════════════════════════════════════

    def _start_skill(self):
        if self._behaviour is None:
            self._set_status("Select a behaviour first", DANGER)
            return
        if not self._engine or not self._engine.is_running:
            self._set_status("Engine is not running — start it from the toolbar first", DANGER)
            return
        if self._behaviour.needs_target and self._target_point_mm is None:
            self._set_status("This behaviour needs a target point", DANGER)
            return

        rid, is_yellow = self._get_rid_yellow()
        label = self._robot_label(is_yellow, rid)
        reset_robot_state(rid)
        get_movement(rid, is_yellow=is_yellow).reset()
        self._engine.set_field_manual_control(rid, is_yellow, True)
        self._running = True
        self._timer.start()
        self._set_status(f"Running: {self._behaviour.label} on {label} — click Stop to cancel", SUCCESS)
        self._log.info(f"Skill started: {self._behaviour.label} → {label}")

    def _stop_skill(self):
        if not self._running:
            return
        self._timer.stop()
        self._running = False
        rid, is_yellow = self._get_rid_yellow()
        get_movement(rid, is_yellow=is_yellow).reset()
        if self._engine and self._engine.is_running:
            cmd = RobotCommand(robot_id=rid, isYellow=is_yellow)
            self._engine.send_robot_command(cmd, runtime=0.05)
            self._engine.set_field_manual_control(rid, is_yellow, False)
        self._set_status("Stopped", TEXT_DIM)
        self._log.info("Skill stopped")

    def stop_skill(self):
        """Public hook for MainWindow to call on engine-stop / app close."""
        self._stop_skill()
        self._stop_chaotic()

    def _stop_all(self):
        self._stop_skill()
        if not self._engine or not self._engine.is_running:
            return
        try:
            cfg = Config()
        except Exception as e:
            self._log.err(f"Failed to load config for STOP ALL: {e}")
            return
        n = 0
        for team_name, team_data in [("Yellow", cfg.yellow), ("Blue", cfg.blue)]:
            if not team_data:
                continue
            for _key, rd in team_data.items():
                cmd = RobotCommand(robot_id=rd.get("shellID", 0), isYellow=(team_name == "Yellow"))
                self._engine.send_robot_command(cmd, runtime=0.05)
                n += 1
        self._log.info(f"STOP ALL sent to {n} robots")

    # ══════════════════════════════════════════════════════════════
    #  Tick — runs the selected behaviour through the real Intent pipeline
    # ══════════════════════════════════════════════════════════════

    def _tick(self):
        if not self._engine or self._engine._wm is None or self._behaviour is None:
            return

        rid, is_yellow = self._get_rid_yellow()
        snap = build_snapshot_from_world_model(self._engine._wm, is_yellow)
        if snap is None:
            self._readout_label.setText("No vision frame yet")
            return

        robot = next((r for r in snap.own_robots if r.robot_id == rid), None)

        target_m = None
        if self._behaviour.needs_target:
            if self._target_point_mm is None:
                self._stop_skill()
                self._set_status("Target point cleared — stopped", WARNING)
                return
            target_m = (self._target_point_mm[0] / 1000.0, self._target_point_mm[1] / 1000.0)

        intent = self._behaviour.intent_fn(snap, robot, target_m)
        if intent is None:
            self._readout_label.setText(f"{self._robot_label(is_yellow, rid)} not visible — waiting…")
            return

        cmd = self._intent_to_command(intent, snap, robot, rid, is_yellow)
        if cmd is None:
            return
        self._engine.send_robot_command(cmd, runtime=0.20)
        self._update_readout(snap, robot, intent, cmd)

    def _intent_to_command(self, intent, snap, robot, rid: int, is_yellow: bool) -> RobotCommand | None:
        """Intent -> RobotCommand via robot/Movement.py (no BT v2 PD-controller pipeline).

        Mirrors the pattern already used by ``robot/goal.py``: world-frame
        positions in mm, ``RobotMovement.velocity_to_target`` handles the
        world->robot frame rotation internally.
        """
        if robot is None:
            return None
        movement = get_movement(rid, is_yellow=is_yellow)
        robot_pos_mm = (robot.position[0] * 1000.0, robot.position[1] * 1000.0, robot.orientation)

        if isinstance(intent, IntentMove):
            target_mm = (intent.target_pos[0] * 1000.0, intent.target_pos[1] * 1000.0)
            turning_target = None
            if intent.target_orientation is not None:
                theta = intent.target_orientation
                turning_target = (robot_pos_mm[0] + 1000.0 * math.cos(theta),
                                   robot_pos_mm[1] + 1000.0 * math.sin(theta))
            vx, vy, w = movement.velocity_to_target(
                robot_pos=robot_pos_mm, target=target_mm, turning_target=turning_target,
                speed=intent.max_speed, stop_threshold=0.0, stay_in_field=True)
            return RobotCommand(robot_id=rid, vx=vx, vy=vy, w=w, isYellow=is_yellow)

        if isinstance(intent, (IntentKick, IntentPass)):
            ball_mm = (snap.ball_position[0] * 1000.0, snap.ball_position[1] * 1000.0)
            target_mm = (intent.target_pos[0] * 1000.0, intent.target_pos[1] * 1000.0)
            vx, vy, w = movement.velocity_to_target(
                robot_pos=robot_pos_mm, target=ball_mm, turning_target=target_mm,
                stop_threshold=0.0, stay_in_field=True)
            ball_rel = world2robot(robot_pos_mm, ball_mm)
            dist_to_ball = math.hypot(ball_rel[0], ball_rel[1])
            target_rel = world2robot(robot_pos_mm, target_mm)
            angle_to_target = math.atan2(target_rel[1], target_rel[0])
            kick = 1 if (dist_to_ball < 150.0 and abs(angle_to_target) < 0.2) else 0
            return RobotCommand(robot_id=rid, vx=vx, vy=vy, w=w, kick=kick, isYellow=is_yellow)

        if isinstance(intent, IntentDribble):
            target_mm = (intent.target_pos[0] * 1000.0, intent.target_pos[1] * 1000.0)
            vx, vy, w = movement.velocity_to_target(
                robot_pos=robot_pos_mm, target=target_mm, turning_target=target_mm, stay_in_field=True)
            return RobotCommand(robot_id=rid, vx=vx, vy=vy, w=w, dribble=1, isYellow=is_yellow)

        if isinstance(intent, IntentOrient):
            theta = intent.target_orientation
            turn_point = (robot_pos_mm[0] + 1000.0 * math.cos(theta), robot_pos_mm[1] + 1000.0 * math.sin(theta))
            vx, vy, w = movement.velocity_to_target(
                robot_pos=robot_pos_mm, target=robot_pos_mm[:2], turning_target=turn_point, stay_in_field=False)
            return RobotCommand(robot_id=rid, vx=vx, vy=vy, w=w, isYellow=is_yellow)

        if isinstance(intent, IntentReceive):
            return RobotCommand(robot_id=rid, isYellow=is_yellow)

        return None

    def _update_readout(self, snap, robot, intent, cmd: RobotCommand):
        lines = [f"intent: {intent.__class__.__name__}{getattr(intent, 'target_pos', '')}"]
        if robot is not None:
            dist_ball = math.hypot(
                robot.position[0] - snap.ball_position[0], robot.position[1] - snap.ball_position[1])
            lines.append(f"robot pos: ({robot.position[0]:.2f}, {robot.position[1]:.2f}) m  "
                         f"heading: {math.degrees(robot.orientation):.0f}°")
            lines.append(f"dist to ball: {dist_ball:.2f} m")
        lines.append(f"cmd: vx={cmd.vx:.2f} vy={cmd.vy:.2f} w={cmd.w:.2f} "
                     f"kick={cmd.kick} dribble={cmd.dribble}")
        if self._behaviour and self._behaviour.compliance_fn:
            lines.extend(self._behaviour.compliance_fn(snap, robot))
        self._readout_label.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════
    #  Chaotic strategy — multi-robot simultaneous tick
    # ══════════════════════════════════════════════════════════════

    def _on_chaotic_team_changed(self, team_name: str):
        self._populate_chaotic_goalie_combo(team_name)
        self._populate_strategy_defender_combo(team_name)

    def _populate_chaotic_goalie_combo(self, team_name: str):
        mapping = self._robots_by_team.get(team_name, {})
        self._chaotic_goalie_combo.blockSignals(True)
        self._chaotic_goalie_combo.clear()
        for letter in sorted(mapping.keys()):
            self._chaotic_goalie_combo.addItem(letter)
        self._chaotic_goalie_combo.blockSignals(False)

    def _populate_strategy_defender_combo(self, team_name: str):
        mapping = self._robots_by_team.get(team_name, {})
        self._strategy_defender_combo.blockSignals(True)
        self._strategy_defender_combo.clear()
        for letter in sorted(mapping.keys()):
            self._strategy_defender_combo.addItem(letter)
        self._strategy_defender_combo.blockSignals(False)

    def _start_chaotic(self, mode: str):
        if not self._engine or not self._engine.is_running:
            self._set_chaotic_status("Engine not running", DANGER)
            return
        self._stop_chaotic()

        team_name = self._chaotic_team_combo.currentText()
        is_yellow = (team_name == "Yellow")
        goalie_letter = self._chaotic_goalie_combo.currentText()
        goalie_id = self._robots_by_team.get(team_name, {}).get(goalie_letter, 0)
        defender_letter = self._strategy_defender_combo.currentText()
        defender_id = self._robots_by_team.get(team_name, {}).get(defender_letter, -1)

        goalie_beh = BEHAVIOURS_BY_ID["goalie_intercept"]
        if mode == "attack":
            # Possession: everyone shoots, goalie stays back
            defender_beh = BEHAVIOURS_BY_ID["move_then_attack"]
            attacker_beh = BEHAVIOURS_BY_ID["move_then_attack"]
        else:
            # No possession: attackers chase ball, defender blocks goal, goalie intercepts
            defender_beh = BEHAVIOURS_BY_ID["defender_block"]
            attacker_beh = BEHAVIOURS_BY_ID["move_to_ball"]

        self._chaotic_robots = {}
        for letter, shell_id in self._robots_by_team.get(team_name, {}).items():
            if shell_id == goalie_id:
                beh = goalie_beh
            elif shell_id == defender_id:
                beh = defender_beh
            else:
                beh = attacker_beh
            reset_robot_state(shell_id)
            get_movement(shell_id, is_yellow=is_yellow).reset()
            self._engine.set_field_manual_control(shell_id, is_yellow, True)
            self._chaotic_robots[shell_id] = (beh, is_yellow)

        self._chaotic_timer.start()
        n = len(self._chaotic_robots)
        self._set_chaotic_status(
            f"{'Possession' if mode == 'attack' else 'No Possession'}: "
            f"{n} robots (goalie={goalie_letter}, def={defender_letter})", SUCCESS)
        self._log.info(
            f"Strategy {mode} — {n} robots, goalie={goalie_letter} (shell {goalie_id}), "
            f"defender={defender_letter} (shell {defender_id})")

    def _stop_chaotic(self):
        if not self._chaotic_timer.isActive() and not self._chaotic_robots:
            return
        self._chaotic_timer.stop()
        if self._engine and self._engine.is_running:
            for shell_id, (_, is_yellow) in self._chaotic_robots.items():
                cmd = RobotCommand(robot_id=shell_id, isYellow=is_yellow)
                self._engine.send_robot_command(cmd, runtime=0.05)
                self._engine.set_field_manual_control(shell_id, is_yellow, False)
        self._chaotic_robots = {}
        self._set_chaotic_status("Stopped", TEXT_DIM)
        self._log.info("Chaotic strategy stopped")

    def _chaotic_tick(self):
        if not self._engine or self._engine._wm is None or not self._chaotic_robots:
            return
        first_yellow = next(iter(self._chaotic_robots.values()))[1]
        snap = build_snapshot_from_world_model(self._engine._wm, first_yellow)
        if snap is None:
            return
        for shell_id, (beh, is_yellow) in self._chaotic_robots.items():
            robot = next((r for r in snap.own_robots if r.robot_id == shell_id), None)
            intent = beh.intent_fn(snap, robot, None)
            if intent is None:
                continue
            cmd = self._intent_to_command(intent, snap, robot, shell_id, is_yellow)
            if cmd is None:
                continue
            self._engine.send_robot_command(cmd, runtime=0.20)

    def _set_chaotic_status(self, text: str, color: str):
        self._chaotic_status.setStyleSheet(f"color:{color}; font-size:11px; padding:2px;")
        self._chaotic_status.setText(text)

    # ══════════════════════════════════════════════════════════════
    #  Custom Skill Builder
    # ══════════════════════════════════════════════════════════════

    def _on_custom_base_changed(self, index: int):
        base_id = self._custom_base_combo.itemData(index)
        base = BEHAVIOURS_BY_ID.get(base_id) if base_id else None
        enabled = base.needs_target if base else True
        self._custom_tx_spin.setEnabled(enabled)
        self._custom_ty_spin.setEnabled(enabled)
        tip = "" if enabled else "Target X/Y ignored — this skill does not use a target point"
        self._custom_tx_spin.setToolTip(tip)
        self._custom_ty_spin.setToolTip(tip)

    def _make_custom_behaviour(self, custom_id: str, label: str, base: Behaviour,
                               tx_m: float, ty_m: float) -> Behaviour:
        def intent_fn(snap, robot, _target, _tx=tx_m, _ty=ty_m, _base=base):
            return _base.intent_fn(snap, robot, (_tx, _ty))
        return Behaviour(
            id=custom_id,
            label=label,
            description=f"{base.label} with fixed target ({tx_m*1000:.0f}, {ty_m*1000:.0f}) mm",
            needs_target=False,
            intent_fn=intent_fn,
            compliance_fn=base.compliance_fn,
        )

    def _add_custom_skill(self):
        base_id = self._custom_base_combo.currentData()
        base = BEHAVIOURS_BY_ID.get(base_id)
        if base is None:
            self._log.err("No base skill selected")
            return
        tx_mm = self._custom_tx_spin.value()
        ty_mm = self._custom_ty_spin.value()
        tx_m  = tx_mm / 1000.0
        ty_m  = ty_mm / 1000.0
        label = self._custom_label_edit.text().strip()
        if not label:
            label = f"{base.label} @ ({tx_mm:.0f}, {ty_mm:.0f})"
        custom_id = f"custom_{len(self._custom_skills)}_{id(label)}"
        new_beh = self._make_custom_behaviour(custom_id, label, base, tx_m, ty_m)
        self._custom_skills.append(new_beh)
        self._custom_skill_meta[custom_id] = {
            "base_id": base_id,
            "label": label,
            "target_x_mm": tx_mm,
            "target_y_mm": ty_mm,
        }
        self._behaviours.append(new_beh)
        item = QListWidgetItem(f"★ {label}")
        item.setData(Qt.UserRole, custom_id)
        item.setForeground(QColor(ACCENT))
        self._list.addItem(item)
        self._list.setCurrentRow(len(self._behaviours) - 1)
        self._custom_label_edit.clear()
        self._save_custom_skills()
        self._log.ok(f"Added custom skill: {label}")

    def _remove_custom_skill(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._behaviours):
            return
        beh = self._behaviours[row]
        if not beh.id.startswith("custom_"):
            return
        self._behaviours.pop(row)
        self._custom_skills = [b for b in self._custom_skills if b.id != beh.id]
        self._custom_skill_meta.pop(beh.id, None)
        self._list.takeItem(row)
        self._save_custom_skills()
        self._log.ok(f"Removed custom skill: {beh.label}")

    def _save_custom_skills(self):
        data = [self._custom_skill_meta[b.id]
                for b in self._custom_skills if b.id in self._custom_skill_meta]
        try:
            _CUSTOM_SKILLS_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            self._log.err(f"Failed to save custom skills: {e}")

    def _load_custom_skills(self):
        if not _CUSTOM_SKILLS_PATH.exists():
            return
        try:
            data = json.loads(_CUSTOM_SKILLS_PATH.read_text())
        except Exception as e:
            self._log.err(f"Failed to load custom skills: {e}")
            return
        for entry in data:
            base_id = entry.get("base_id", "")
            base = BEHAVIOURS_BY_ID.get(base_id)
            if base is None:
                self._log.err(f"Custom skill references unknown base '{base_id}' — skipped")
                continue
            label  = entry.get("label", base.label)
            tx_mm  = float(entry.get("target_x_mm", 0))
            ty_mm  = float(entry.get("target_y_mm", 0))
            custom_id = f"custom_{len(self._custom_skills)}_{label}"
            new_beh = self._make_custom_behaviour(custom_id, label, base,
                                                   tx_mm / 1000.0, ty_mm / 1000.0)
            self._custom_skills.append(new_beh)
            self._custom_skill_meta[custom_id] = {
                "base_id": base_id, "label": label,
                "target_x_mm": tx_mm, "target_y_mm": ty_mm,
            }
            self._behaviours.append(new_beh)
            item = QListWidgetItem(f"★ {label}")
            item.setData(Qt.UserRole, custom_id)
            item.setForeground(QColor(ACCENT))
            self._list.addItem(item)
        if self._custom_skills:
            self._log.info(f"Loaded {len(self._custom_skills)} custom skill(s)")

    # ══════════════════════════════════════════════════════════════
    #  Status helper
    # ══════════════════════════════════════════════════════════════

    def _set_status(self, text: str, color: str):
        self._status_label.setStyleSheet(f"color:{color}; font-size:12px; padding:4px;")
        self._status_label.setText(text)
