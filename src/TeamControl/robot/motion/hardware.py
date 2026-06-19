import math

from TeamControl.robot import constants as C


DEFAULT_HARDWARE_GAINS = {
    "speed_scale": 1.0,
    "lateral_drift_per_m": 0.0,
    "stop_overshoot_mm": 0.0,
    "min_v": C.MIN_V,
    "min_w": C.MIN_W,
}


def shorten_target_for_overshoot(target_in_robot_frame, stop_overshoot_mm):
    """
    Aim slightly short if the robot usually rolls past the target.

    target_in_robot_frame is in mm.
    """
    x, y = target_in_robot_frame
    dist = math.hypot(x, y)
    overshoot = max(float(stop_overshoot_mm), 0.0)

    if dist <= 0.0 or overshoot <= 0.0:
        return x, y

    new_dist = max(dist - overshoot, 0.0)
    scale = new_dist / dist
    return x * scale, y * scale


def apply_hardware_gains(vx, vy, gains):
    """
    Apply real-robot response gains after PD.

    speed_scale:
        actual_speed / commanded_speed from calibration.
        If the robot is slow, this value is below 1, so dividing boosts output.

    lateral_drift_per_m:
        Side drift in mm per meter of forward travel.
        Positive means the robot drifts left while driving forward, so we
        command a small rightward correction.
    """
    speed_scale = float(gains.get("speed_scale", 1.0))
    if speed_scale > 0.01:
        vx /= speed_scale
        vy /= speed_scale

    drift_ratio = float(gains.get("lateral_drift_per_m", 0.0)) / 1000.0
    vy -= vx * drift_ratio

    return vx, vy


def apply_min_linear_command(vx, vy, min_v):
    """Boost tiny nonzero linear commands so the robot actually moves."""
    speed = math.hypot(vx, vy)
    min_v = max(float(min_v), 0.0)

    if speed <= 0.0 or min_v <= 0.0 or speed >= min_v:
        return vx, vy

    scale = min_v / speed
    return vx * scale, vy * scale


def apply_min_angular_command(w, min_w):
    """Boost tiny nonzero angular commands so the robot actually turns."""
    min_w = max(float(min_w), 0.0)

    if w == 0.0 or min_w <= 0.0 or abs(w) >= min_w:
        return w

    return math.copysign(min_w, w)
