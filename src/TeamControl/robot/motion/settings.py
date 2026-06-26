import json
import os
import time

from TeamControl.robot import constants as C
from TeamControl.robot.motion.hardware import DEFAULT_HARDWARE_GAINS
from TeamControl.robot.motion.wheel_kinematics import DEFAULT_WHEEL_SPEC

_GAIN_KEYS = (
    "turn_kp",
    "turn_kd",
    "linear_kp",
    "linear_kd",
    "speed_scale",
    "lateral_drift_per_m",
    "stop_overshoot_mm",
    "min_v",
    "min_w",
    "wheel1_angle_deg",
    "wheel2_angle_deg",
    "wheel3_angle_deg",
    "wheel4_angle_deg",
    "wheel_radius_mm",
    "robot_radius_mm",
)

# These two may be None ("not calibrated yet" -- isotropic limiter stays
# active). Every key in _GAIN_KEYS above is always a plain float.
_OPTIONAL_GAIN_KEYS = (
    "max_wheel_speed_mps",
    "max_wheel_accel_mps2",
)


class PDSettingsStore:
    def __init__(self, path="movement_calibration.json"):
        self.path = os.path.abspath(path)
        self.data = self._load_file()

    def _default_gains(self):
        gains = {
            "turn_kp": C.TURN_KP,
            "turn_kd": C.TURN_KD,
            "linear_kp": C.LINEAR_KP,
            "linear_kd": C.LINEAR_KD,
        }
        gains.update(DEFAULT_HARDWARE_GAINS)
        gains.update(DEFAULT_WHEEL_SPEC)
        return gains

    def _team_key(self, is_yellow):
        return "yellow" if is_yellow else "blue"

    def _load_file(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("yellow", {})
                data.setdefault("blue", {})
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        return {
            "yellow": {},
            "blue": {},
        }

    def _save_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")

    def load_gains(self, robot_id, is_yellow) -> dict:
        gains = self._default_gains()

        team = self._team_key(is_yellow)
        robot_key = str(int(robot_id))

        robot_gains = self.data.get(team, {}).get(robot_key, {})

        for key in _GAIN_KEYS:
            if key in robot_gains:
                try:
                    gains[key] = float(robot_gains[key])
                except (TypeError, ValueError):
                    pass

        for key in _OPTIONAL_GAIN_KEYS:
            if key in robot_gains:
                value = robot_gains[key]
                try:
                    gains[key] = None if value is None else float(value)
                except (TypeError, ValueError):
                    pass

        return gains

    def has_robot_gains(self, robot_id, is_yellow) -> bool:
        """True if this robot has saved tuned gains in the settings file."""
        team = self._team_key(is_yellow)
        robot_key = str(int(robot_id))
        return robot_key in self.data.get(team, {})

    def load_default_gains(self) -> dict:
        """Return global fallback gains from constants.py."""
        return self._default_gains()

    def load_gains_with_source(self, robot_id, is_yellow) -> tuple[dict, str]:
        """Return gains plus 'robot' if tuned, otherwise 'default'."""
        source = "robot" if self.has_robot_gains(robot_id, is_yellow) else "default"
        return self.load_gains(robot_id, is_yellow), source

    def save_gains(self, robot_id, is_yellow, gains, score=None):
        team = self._team_key(is_yellow)
        robot_key = str(int(robot_id))

        saved = self._default_gains()

        for key in _GAIN_KEYS:
            if key in gains:
                saved[key] = float(gains[key])

        for key in _OPTIONAL_GAIN_KEYS:
            if key in gains:
                value = gains[key]
                saved[key] = None if value is None else float(value)

        if score is not None:
            saved["score"] = float(score)

        saved["updated_at"] = time.time()

        self.data.setdefault("yellow", {})
        self.data.setdefault("blue", {})
        self.data[team][robot_key] = saved

        self._save_file()
        return saved

    def delete_robot_gains(self, robot_id, is_yellow) -> bool:
        """Delete saved tuned gains for one robot. Returns True if removed."""
        team = self._team_key(is_yellow)
        robot_key = str(int(robot_id))
        robots = self.data.setdefault(team, {})
        if robot_key not in robots:
            return False

        del robots[robot_key]
        self._save_file()
        return True
