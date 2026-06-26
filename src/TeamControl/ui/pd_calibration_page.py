import threading
import time

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QFrame,
)

from TeamControl.robot import constants as C
from TeamControl.robot.motion import get_motion_controller
from TeamControl.robot.motion import calibration_log
from TeamControl.robot.motion.pd_calibration import PDCalibration
from TeamControl.ui.theme import ACCENT, BG_CARD, BORDER, DANGER, SUCCESS, TEXT, TEXT_DIM, WARNING
from TeamControl.utils.yaml_config import Config


class EnginePoseSource:
    """Pose adapter for PDCalibration using SimEngine's WorldModel."""

    def __init__(self, engine):
        self.engine = engine

    def get_robot_pose(self, robot_id, is_yellow):
        wm = getattr(self.engine, "_wm", None)
        if wm is None:
            return None
        frame = wm.get_latest_frame()
        if frame is None:
            return None

        robots = frame.robots_yellow if is_yellow else frame.robots_blue
        for robot in robots or []:
            if int(robot.id) == int(robot_id):
                theta = getattr(robot, "o", getattr(robot, "orientation", 0.0))
                return (float(robot.x), float(robot.y), float(theta))
        return None


class PDCalibrationPage(QWidget):
    """GUI for running PD calibration tests, auto-tuning, and hardware gains."""

    log_line = Signal(str)
    busy_changed = Signal(bool)
    gains_updated = Signal(dict)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._last_result = None
        self._last_gains = None
        self._worker = None

        self.log_line.connect(self._append_log)
        self.busy_changed.connect(self._set_busy)
        self.gains_updated.connect(self._on_gains_updated)
        self._build_ui()

    def _card(self, title_text):
        """Create a styled card frame with title (mirrors calibration_page.py)."""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:6px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        title = QLabel(title_text)
        title.setStyleSheet(f"font-size:13px; font-weight:bold; color:{ACCENT}; padding:0;")
        lay.addWidget(title)
        return card, lay

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("PD Calibration")
        title.setStyleSheet(f"color:{ACCENT}; font-size:14px; font-weight:bold;")
        root.addWidget(title)

        # ── Robot picker — shared by both sections below ────────────
        robot_row = QHBoxLayout()
        robot_row.addWidget(QLabel("Robot:"))
        self._robot_combo = QComboBox()
        self._robot_combo.setMinimumWidth(220)
        robot_row.addWidget(self._robot_combo, 1)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._load_robot_choices)
        robot_row.addWidget(reload_btn)
        root.addLayout(robot_row)
        self._load_robot_choices()

        self._gain_source_lbl = QLabel("Source: constants.py (global defaults)")
        self._gain_source_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        root.addWidget(self._gain_source_lbl)

        # ── Section A: PD Gains ──────────────────────────────────────
        pd_card, pd_lay = self._card("PD Gains")

        toggles_row = QHBoxLayout()
        self._use_pd = QCheckBox("Use PD")
        self._use_pd.setChecked(True)
        self._use_hardware = QCheckBox("Use Hardware")
        self._use_hardware.setChecked(False)
        toggles_row.addWidget(self._use_pd)
        toggles_row.addWidget(self._use_hardware)
        toggles_row.addStretch()
        pd_lay.addLayout(toggles_row)

        # Spinbox defaults read from constants.py so they track tuning.json changes
        self._turn_kp = self._gain_spin(C.TURN_KP, step=0.05)
        self._turn_kd = self._gain_spin(C.TURN_KD, step=0.01)
        self._linear_kp = self._gain_spin(C.LINEAR_KP, step=0.0001, decimals=5)
        self._linear_kd = self._gain_spin(C.LINEAR_KD, step=0.0001, decimals=5)

        pd_grid = QGridLayout()
        pd_grid.setSpacing(8)
        pd_grid.addWidget(QLabel("turn_kp"), 0, 0)
        pd_grid.addWidget(self._turn_kp, 0, 1)
        pd_grid.addWidget(QLabel("turn_kd"), 0, 2)
        pd_grid.addWidget(self._turn_kd, 0, 3)
        pd_grid.addWidget(QLabel("linear_kp"), 1, 0)
        pd_grid.addWidget(self._linear_kp, 1, 1)
        pd_grid.addWidget(QLabel("linear_kd"), 1, 2)
        pd_grid.addWidget(self._linear_kd, 1, 3)
        pd_lay.addLayout(pd_grid)

        pd_actions = QGridLayout()
        self._start_btn = QPushButton("Start Calibration Backend")
        self._turn_btn = QPushButton("Run Angular Turn")
        self._linear_btn = QPushButton("Run Linear Forward")
        self._auto_turn_btn = QPushButton("Auto-Tune Turn")
        self._auto_linear_btn = QPushButton("Auto-Tune Linear")
        self._apply_default_btn = QPushButton("Apply Defaults")
        self._save_btn = QPushButton("Save Last Result")
        self._clear_btn = QPushButton("Clear Tuned Gains")

        pd_actions.addWidget(self._start_btn, 0, 0, 1, 2)
        pd_actions.addWidget(self._turn_btn, 1, 0)
        pd_actions.addWidget(self._linear_btn, 1, 1)
        pd_actions.addWidget(self._auto_turn_btn, 2, 0)
        pd_actions.addWidget(self._auto_linear_btn, 2, 1)
        pd_actions.addWidget(self._apply_default_btn, 3, 0)
        pd_actions.addWidget(self._save_btn, 3, 1)
        pd_actions.addWidget(self._clear_btn, 4, 0, 1, 2)
        pd_lay.addLayout(pd_actions)

        root.addWidget(pd_card)

        # ── Section B: Hardware Gains ───────────────────────────────
        hw_card, hw_lay = self._card("Hardware Gains")

        self._speed_scale = self._gain_spin(1.0, step=0.01)
        self._lateral_drift = self._signed_spin(0.0, -50.0, 50.0, step=0.5)
        self._stop_overshoot = self._signed_spin(0.0, 0.0, 500.0, step=5.0)
        self._min_v = self._signed_spin(C.MIN_V, 0.0, 1.0, step=0.01)
        self._min_w = self._signed_spin(C.MIN_W, 0.0, 3.0, step=0.01)

        hw_grid = QGridLayout()
        hw_grid.setSpacing(8)
        hw_grid.addWidget(QLabel("speed_scale"), 0, 0)
        hw_grid.addWidget(self._speed_scale, 0, 1)
        hw_grid.addWidget(QLabel("drift mm/m"), 0, 2)
        hw_grid.addWidget(self._lateral_drift, 0, 3)
        hw_grid.addWidget(QLabel("overshoot mm"), 1, 0)
        hw_grid.addWidget(self._stop_overshoot, 1, 1)
        hw_grid.addWidget(QLabel("min_v"), 1, 2)
        hw_grid.addWidget(self._min_v, 1, 3)
        hw_grid.addWidget(QLabel("min_w"), 2, 0)
        hw_grid.addWidget(self._min_w, 2, 1)
        hw_lay.addLayout(hw_grid)

        hw_note = QLabel(
            "Auto-calibrated separately on the Calibration page "
            "(Auto-Calibrate / Speed Sweep). Edit here only for manual overrides."
        )
        hw_note.setWordWrap(True)
        hw_note.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        hw_lay.addWidget(hw_note)

        root.addWidget(hw_card)

        # ── Section C: Wheel Geometry ────────────────────────────────
        # Manual entry only -- these are physical robot specs you measure
        # once, not control gains you search over, so there's no auto-tune
        # button here. Defaults match TurtleRabbit.ini (grSim's reference
        # robot model); max_wheel_speed/accel default to 0 (= "not
        # calibrated yet" -- the isotropic AccelLimiter/MAX_SPEED path
        # stays in effect until both are set).
        wheel_card, wheel_lay = self._card("Wheel Geometry")

        self._wheel1_angle = self._signed_spin(60.0, 0.0, 360.0, step=1.0, decimals=1)
        self._wheel2_angle = self._signed_spin(135.0, 0.0, 360.0, step=1.0, decimals=1)
        self._wheel3_angle = self._signed_spin(225.0, 0.0, 360.0, step=1.0, decimals=1)
        self._wheel4_angle = self._signed_spin(300.0, 0.0, 360.0, step=1.0, decimals=1)
        self._wheel_radius = self._signed_spin(32.5, 0.0, 200.0, step=0.5, decimals=2)
        self._robot_radius = self._signed_spin(88.5, 0.0, 200.0, step=0.5, decimals=2)
        self._max_wheel_speed = self._signed_spin(0.0, 0.0, 20.0, step=0.1, decimals=2)
        self._max_wheel_accel = self._signed_spin(0.0, 0.0, 100.0, step=0.5, decimals=2)

        wheel_grid = QGridLayout()
        wheel_grid.setSpacing(8)
        wheel_grid.addWidget(QLabel("wheel1 angle °"), 0, 0)
        wheel_grid.addWidget(self._wheel1_angle, 0, 1)
        wheel_grid.addWidget(QLabel("wheel2 angle °"), 0, 2)
        wheel_grid.addWidget(self._wheel2_angle, 0, 3)
        wheel_grid.addWidget(QLabel("wheel3 angle °"), 1, 0)
        wheel_grid.addWidget(self._wheel3_angle, 1, 1)
        wheel_grid.addWidget(QLabel("wheel4 angle °"), 1, 2)
        wheel_grid.addWidget(self._wheel4_angle, 1, 3)
        wheel_grid.addWidget(QLabel("wheel radius mm"), 2, 0)
        wheel_grid.addWidget(self._wheel_radius, 2, 1)
        wheel_grid.addWidget(QLabel("robot radius mm"), 2, 2)
        wheel_grid.addWidget(self._robot_radius, 2, 3)
        wheel_grid.addWidget(QLabel("max wheel speed m/s"), 3, 0)
        wheel_grid.addWidget(self._max_wheel_speed, 3, 1)
        wheel_grid.addWidget(QLabel("max wheel accel m/s²"), 3, 2)
        wheel_grid.addWidget(self._max_wheel_accel, 3, 3)
        wheel_lay.addLayout(wheel_grid)

        wheel_note = QLabel(
            "Angles are clockwise from facing forward (matches grSim's "
            "TurtleRabbit.ini). Leave speed/accel at 0 until measured -- "
            "0 means \"not calibrated\", so the regular speed/accel caps "
            "stay in effect."
        )
        wheel_note.setWordWrap(True)
        wheel_note.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        wheel_lay.addWidget(wheel_note)

        root.addWidget(wheel_card)

        self._status = QLabel("Start the backend, then run one test.")
        self._status.setStyleSheet(f"color:{TEXT_DIM};")
        root.addWidget(self._status)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(f"color:{TEXT}; background:#101418;")
        root.addWidget(self._log, 1)

        self._start_btn.clicked.connect(self._start_backend)
        self._turn_btn.clicked.connect(lambda: self._run_test("turn"))
        self._linear_btn.clicked.connect(lambda: self._run_test("linear"))
        self._auto_turn_btn.clicked.connect(lambda: self._run_auto_tune("turn"))
        self._auto_linear_btn.clicked.connect(lambda: self._run_auto_tune("linear"))
        self._apply_default_btn.clicked.connect(self._apply_defaults)
        self._save_btn.clicked.connect(self._save_last_result)
        self._clear_btn.clicked.connect(self._clear_tuned_gains)

        # Auto-load saved (or default) gains when the selected robot changes
        self._robot_combo.currentIndexChanged.connect(self._on_robot_changed)

    def _gain_spin(self, value, step, decimals=4):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 10.0)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _signed_spin(self, value, lo, hi, step, decimals=2):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _gains(self):
        return {
            "turn_kp": self._turn_kp.value(),
            "turn_kd": self._turn_kd.value(),
            "linear_kp": self._linear_kp.value(),
            "linear_kd": self._linear_kd.value(),
            "speed_scale": self._speed_scale.value(),
            "lateral_drift_per_m": self._lateral_drift.value(),
            "stop_overshoot_mm": self._stop_overshoot.value(),
            "min_v": self._min_v.value(),
            "min_w": self._min_w.value(),
            "wheel1_angle_deg": self._wheel1_angle.value(),
            "wheel2_angle_deg": self._wheel2_angle.value(),
            "wheel3_angle_deg": self._wheel3_angle.value(),
            "wheel4_angle_deg": self._wheel4_angle.value(),
            "wheel_radius_mm": self._wheel_radius.value(),
            "robot_radius_mm": self._robot_radius.value(),
            # 0 in the spinbox means "not calibrated" -> None (disabled).
            "max_wheel_speed_mps": self._max_wheel_speed.value() or None,
            "max_wheel_accel_mps2": self._max_wheel_accel.value() or None,
        }

    # ══════════════════════════════════════════════════════════════
    #  Robot picker — single combo for all configured robots
    # ══════════════════════════════════════════════════════════════

    def _load_robot_choices(self):
        if not hasattr(self, "_robot_combo"):
            return
        current = self._robot_combo.currentData()
        self._robot_combo.clear()
        try:
            cfg = Config()
        except Exception:
            self._robot_combo.addItem("Yellow shell 0", (0, True, "A"))
            return

        for team_name, team_data, is_yellow in (
            ("Yellow", cfg.yellow, True),
            ("Blue", cfg.blue, False),
        ):
            if not team_data:
                continue
            for letter, rd in team_data.items():
                shell_id = int(rd.get("shellID", 0))
                label = (
                    f"{team_name} {letter}  shell {shell_id}  "
                    f"{rd.get('ip', '127.0.0.1')}:{rd.get('port', 50514)}"
                )
                data = (shell_id, is_yellow, letter)
                self._robot_combo.addItem(label, data)

        if self._robot_combo.count() == 0:
            self._robot_combo.addItem("Yellow shell 0", (0, True, "A"))
        elif current is not None:
            idx = self._robot_combo.findData(current)
            if idx >= 0:
                self._robot_combo.setCurrentIndex(idx)

    def _robot_info(self):
        """Return (shell_id, is_yellow, letter) for the selected robot."""
        data = self._robot_combo.currentData()
        if data is None:
            return 0, True, "A"
        shell_id, is_yellow, letter = data
        return int(shell_id), bool(is_yellow), str(letter)

    def _motion(self):
        rid, is_yellow, _letter = self._robot_info()
        return get_motion_controller(rid, is_yellow)

    def _calibrator(self):
        dispatch_q = getattr(self._engine, "_dispatch_q", None)
        if dispatch_q is None:
            raise RuntimeError("Engine backend is not running; no dispatch queue available")
        return PDCalibration(
            motion=self._motion(),
            pose_source=EnginePoseSource(self._engine),
            dispatch_q=dispatch_q,
        )

    def _start_backend(self):
        if self._engine.is_running:
            self._append_log("[cal] Backend already running")
            return
        rid, _is_yellow, _letter = self._robot_info()
        self._engine.start(
            "calibration",
            our_id=rid,
            enemy_id=0,
        )
        self._append_log("[cal] Started calibration backend")

    # ══════════════════════════════════════════════════════════════
    #  Single tests
    # ══════════════════════════════════════════════════════════════

    def _run_test(self, test_name):
        if self._worker and self._worker.is_alive():
            self._append_log("[cal] A calibration test is already running")
            return

        gains = self._gains()
        use_pd = self._use_pd.isChecked()
        use_hardware = self._use_hardware.isChecked()
        self._last_gains = gains
        self._set_status(f"Running {test_name} test...", ACCENT)
        self._set_busy(True)

        def work():
            try:
                cal = self._calibrator()
                if test_name == "turn":
                    result = cal.run_angular_turn_test(
                        gains=gains,
                        use_pd=use_pd,
                        use_hardware=use_hardware,
                    )
                else:
                    result = cal.run_linear_forward_test(
                        gains=gains,
                        use_pd=use_pd,
                        use_hardware=use_hardware,
                    )
                self._last_result = result
                self.log_line.emit(
                    f"[cal] {result.test_name}: passed={result.passed} "
                    f"score={result.score:.2f} "
                    f"pos_err={result.final_position_error_mm:.1f}mm "
                    f"theta_err={result.final_heading_error_rad:.3f}rad "
                    f"samples={result.samples}"
                )
                self.log_line.emit("[cal] Use Save Last Result if this tuning is good")
            except Exception as exc:
                self.log_line.emit(f"[cal:error] {exc}")
            finally:
                self.log_line.emit("[cal] Test finished")
                self.busy_changed.emit(False)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    # ══════════════════════════════════════════════════════════════
    #  Auto-tune — coarse-to-fine grid sweep
    # ══════════════════════════════════════════════════════════════

    def _run_auto_tune(self, kind):
        if self._worker and self._worker.is_alive():
            self._append_log("[cal] A calibration test is already running")
            return

        use_pd = self._use_pd.isChecked()
        use_hardware = self._use_hardware.isChecked()
        shell_id, is_yellow, letter = self._robot_info()
        self._set_status(f"Auto-tuning {kind}...", ACCENT)
        self._set_busy(True)

        def work():
            try:
                cal = self._calibrator()
                auto_fn = cal.auto_tune_turn if kind == "turn" else cal.auto_tune_linear
                result = auto_fn(
                    on_candidate=self.log_line.emit,
                    use_pd=use_pd,
                    use_hardware=use_hardware,
                )
                self._last_result = result.best_result
                self._last_gains = result.gains

                log_path = calibration_log.write_autotune_log(
                    team="yellow" if is_yellow else "blue",
                    letter=letter,
                    shell_id=shell_id,
                    kind=kind,
                    result=result,
                )
                self.log_line.emit(
                    f"[cal] Logged {len(result.tried)} candidates to {log_path}"
                )
                self.gains_updated.emit(result.gains)
            except Exception as exc:
                self.log_line.emit(f"[cal:error] {exc}")
            finally:
                self.log_line.emit(f"[cal] Auto-tune {kind} finished")
                self.busy_changed.emit(False)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _on_gains_updated(self, gains):
        """Reflect auto-tune's saved gains back into the spinboxes (main thread)."""
        self._set_gain_spins(gains)
        self._update_source_label("robot")

    def _set_busy(self, busy):
        """Disable controls that shouldn't be touched mid-test/mid-sweep."""
        for widget in (
            self._turn_btn,
            self._linear_btn,
            self._auto_turn_btn,
            self._auto_linear_btn,
            self._save_btn,
            self._clear_btn,
            self._apply_default_btn,
            self._robot_combo,
        ):
            widget.setEnabled(not busy)

    def _on_robot_changed(self):
        """Auto-load saved or default gains when the robot selection changes."""
        motion = self._motion()
        gains, source = motion.reload_saved_or_default_gains()
        self._set_gain_spins(gains)
        self._update_source_label(source)

    def _update_source_label(self, source: str):
        rid, is_yellow, letter = self._robot_info()
        team = "yellow" if is_yellow else "blue"
        if source == "robot":
            self._gain_source_lbl.setText(
                f"Source: movement_calibration.json — robot {team}/{rid} ({letter}) (tuned)")
            self._gain_source_lbl.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        else:
            self._gain_source_lbl.setText(
                f"Source: constants.py — global defaults "
                f"(TURN_KP={C.TURN_KP}, LINEAR_KP={C.LINEAR_KP},"
                f" MAX_W={C.MAX_W:.2f} rad/s cap)")
            self._gain_source_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")

    def _apply_defaults(self):
        gains = self._motion().apply_default_gains()
        self._set_gain_spins(gains)
        self._update_source_label("default")
        self._append_log("[cal] Applied constants.py default gains")

    def _save_last_result(self):
        if self._last_result is None or self._last_gains is None:
            self._append_log("[cal] No calibration result to save")
            return
        saved = self._motion().calibrate(self._last_gains, score=self._last_result.score)
        self._update_source_label("robot")
        self._append_log(f"[cal] Saved gains: {saved}")

    def _clear_tuned_gains(self):
        removed = self._motion().clear_tuned_gains()
        gains = self._motion().get_gains()
        self._set_gain_spins(gains)
        self._update_source_label("default")
        self._append_log(f"[cal] Cleared tuned gains: {removed}; defaults applied")

    def _set_gain_spins(self, gains):
        self._turn_kp.setValue(float(gains["turn_kp"]))
        self._turn_kd.setValue(float(gains["turn_kd"]))
        self._linear_kp.setValue(float(gains["linear_kp"]))
        self._linear_kd.setValue(float(gains["linear_kd"]))
        self._speed_scale.setValue(float(gains.get("speed_scale", 1.0)))
        self._lateral_drift.setValue(float(gains.get("lateral_drift_per_m", 0.0)))
        self._stop_overshoot.setValue(float(gains.get("stop_overshoot_mm", 0.0)))
        self._min_v.setValue(float(gains.get("min_v", 0.0)))
        self._min_w.setValue(float(gains.get("min_w", 0.0)))
        self._wheel1_angle.setValue(float(gains.get("wheel1_angle_deg", 60.0)))
        self._wheel2_angle.setValue(float(gains.get("wheel2_angle_deg", 135.0)))
        self._wheel3_angle.setValue(float(gains.get("wheel3_angle_deg", 225.0)))
        self._wheel4_angle.setValue(float(gains.get("wheel4_angle_deg", 300.0)))
        self._wheel_radius.setValue(float(gains.get("wheel_radius_mm", 32.5)))
        self._robot_radius.setValue(float(gains.get("robot_radius_mm", 88.5)))
        # None ("not calibrated") displays as 0 in the spinbox.
        self._max_wheel_speed.setValue(float(gains.get("max_wheel_speed_mps") or 0.0))
        self._max_wheel_accel.setValue(float(gains.get("max_wheel_accel_mps2") or 0.0))

    def _set_status(self, text, color=TEXT_DIM):
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color};")

    def _append_log(self, text):
        self._log.appendPlainText(f"{time.strftime('%H:%M:%S')} {text}")
        if "[cal:error]" in text:
            self._set_status(text, DANGER)
        elif "finished" in text:
            self._set_status("Ready", SUCCESS)
