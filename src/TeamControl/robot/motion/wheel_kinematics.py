"""
4-omniwheel inverse/forward kinematics for wheel-aware speed/accel limits.

Wheel angles are degrees, clockwise from "facing forward" -- matches
grSim's RobotWheelAngles convention
(network/proto2/ssl_simulation_config.proto) and TurtleRabbit.ini's
Wheel1Angle..Wheel4Angle fields. Robot radius is in meters; vx/vy are
robot-frame m/s, w is rad/s.

Why this exists: MAX_SPEED/AccelLimiter treat "how fast can this robot go"
as an isotropic circle. A real 4-wheel omnidirectional robot with
asymmetric wheel angles (e.g. 60/135/225/300 degrees, not evenly spaced)
actually has a direction-dependent envelope -- some directions hit a
wheel's speed/torque limit sooner than others. These functions let
RobotMotionController clamp to that true envelope instead, once a robot's
wheel geometry has been measured and entered (see PDSettingsStore's
wheel*_angle_deg / wheel_radius_mm / robot_radius_mm / max_wheel_speed_mps
/ max_wheel_accel_mps2 fields).

The exact sign convention below is validated by a round-trip test (recover
(vx, vy, w) from the computed wheel speeds) rather than by comparing
against grSim's internal source -- self-consistency is what the limiter
needs; matching grSim's *geometry* (not its math) is what keeps simulated
physics in line with this model (see docs/pd-controller-design.md).
"""

import math
import time
from typing import Optional, Sequence, Tuple


# Reference geometry from TurtleRabbit.ini (SSL/grSim/config_files) -- the
# physical default until a robot's real wheel spec is measured and entered.
# max_wheel_speed_mps/max_wheel_accel_mps2 default to None ("not calibrated
# yet"): these are motor specs that must be measured, never guessed, and
# None means the isotropic AccelLimiter/MAX_SPEED path stays active.
DEFAULT_WHEEL_SPEC = {
    "wheel1_angle_deg": 60.0,
    "wheel2_angle_deg": 135.0,
    "wheel3_angle_deg": 225.0,
    "wheel4_angle_deg": 300.0,
    "wheel_radius_mm": 32.5,
    "robot_radius_mm": 88.5,
    "max_wheel_speed_mps": None,
    "max_wheel_accel_mps2": None,
}


def _wheel_jacobian_row(angle_deg: float, robot_radius_m: float) -> Tuple[float, float, float]:
    """One row of the body-velocity -> wheel-speed matrix for one wheel.

    wheel_speed = -sin(angle)*vx + cos(angle)*vy + robot_radius*w
    """
    a = math.radians(angle_deg)
    return (-math.sin(a), math.cos(a), robot_radius_m)


def wheel_speeds(
    vx: float,
    vy: float,
    w: float,
    wheel_angles_deg: Sequence[float],
    robot_radius_m: float,
) -> Tuple[float, ...]:
    """Body-frame velocity -> one speed per wheel (m/s)."""
    speeds = []
    for angle in wheel_angles_deg:
        jx, jy, jw = _wheel_jacobian_row(angle, robot_radius_m)
        speeds.append(jx * vx + jy * vy + jw * w)
    return tuple(speeds)


def _solve_3x3(a, b) -> Tuple[float, float, float]:
    """Solve a 3x3 linear system via Cramer's rule (small, fixed-size)."""

    def det3(m):
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    d = det3(a)
    if abs(d) < 1e-12:
        return (0.0, 0.0, 0.0)

    result = []
    for col in range(3):
        m = [row[:] for row in a]
        for r in range(3):
            m[r][col] = b[r]
        result.append(det3(m) / d)
    return tuple(result)


def body_velocity_from_wheel_speeds(
    speeds: Sequence[float],
    wheel_angles_deg: Sequence[float],
    robot_radius_m: float,
) -> Tuple[float, float, float]:
    """Least-squares inverse of wheel_speeds() -- recovers (vx, vy, w).

    With 4 wheels and 3 degrees of freedom the system is over-determined;
    solving the normal equations (J^T J) x = J^T s gives the best fit,
    which is exact when *speeds* actually came from wheel_speeds().
    """
    rows = [_wheel_jacobian_row(a, robot_radius_m) for a in wheel_angles_deg]

    jtj = [[0.0, 0.0, 0.0] for _ in range(3)]
    jts = [0.0, 0.0, 0.0]
    for row, s in zip(rows, speeds):
        for i in range(3):
            jts[i] += row[i] * s
            for j in range(3):
                jtj[i][j] += row[i] * row[j]

    return _solve_3x3(jtj, jts)


def max_angular_from_wheel_budget(
    robot_radius_m: float,
    max_wheel_speed_mps: float,
    share: float,
) -> float:
    """Angular-velocity ceiling from rotation's share of the wheel budget.

    A pure rotation's wheel-speed contribution is always exactly
    ``robot_radius_m * w`` on every wheel, regardless of wheel angle (the
    w-column of the Jacobian is the same robot_radius_m value for every
    row -- see _wheel_jacobian_row). So if rotation may only claim
    ``share`` of the total wheel speed budget (translation gets the
    rest), the angular ceiling is just::

        w_max = (share * max_wheel_speed_mps) / robot_radius_m

    Used by RobotMotionController to keep spinning from eating into the
    wheel budget translation needs, once a robot's wheel spec is
    calibrated (max_wheel_speed_mps not None).
    """
    if robot_radius_m <= 0.0:
        return 0.0
    return (share * max_wheel_speed_mps) / robot_radius_m


def scale_to_wheel_speed_limit(
    vx: float,
    vy: float,
    w: float,
    wheel_angles_deg: Sequence[float],
    robot_radius_m: float,
    max_wheel_speed_mps: Optional[float],
) -> Tuple[float, float, float]:
    """Scale (vx, vy, w) down (direction preserved) so no wheel exceeds the limit."""
    if max_wheel_speed_mps is None or max_wheel_speed_mps <= 0:
        return vx, vy, w

    speeds = wheel_speeds(vx, vy, w, wheel_angles_deg, robot_radius_m)
    worst = max((abs(s) for s in speeds), default=0.0)
    if worst <= max_wheel_speed_mps or worst == 0.0:
        return vx, vy, w

    scale = max_wheel_speed_mps / worst
    return vx * scale, vy * scale, w * scale


class WheelAccelLimiter:
    """Like AccelLimiter (accel.py), but the rate cap lives in wheel-speed
    space instead of an isotropic body-frame norm.

    Caps how fast each wheel's speed can change (max_wheel_accel_mps2),
    then maps the clamped wheel-speed vector back to (vx, vy, w) -- so
    acceleration is limited by what the wheels can actually do, not a
    circle that's the same in every direction.

    On the first call the command is accepted as-is (matches
    AccelLimiter's documented first-call behaviour), so the robot can
    start moving immediately.
    """

    def __init__(
        self,
        wheel_angles_deg: Sequence[float],
        robot_radius_m: float,
        max_wheel_accel_mps2: float,
    ):
        self.wheel_angles_deg = tuple(wheel_angles_deg)
        self.robot_radius_m = float(robot_radius_m)
        self.max_wheel_accel_mps2 = float(max_wheel_accel_mps2)
        self._prev: Optional[Tuple[float, float, float]] = None
        self._prev_t: Optional[float] = None

    def reset(self) -> None:
        self._prev = None
        self._prev_t = None

    def limit(
        self,
        desired: Tuple[float, float, float],
        now: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        if now is None:
            now = time.monotonic()

        vx, vy, w = (float(v) for v in desired)

        if self._prev is None or self._prev_t is None:
            self._prev = (vx, vy, w)
            self._prev_t = now
            return self._prev

        dt = max(now - self._prev_t, 1e-6)
        max_step = self.max_wheel_accel_mps2 * dt

        prev_speeds = wheel_speeds(*self._prev, self.wheel_angles_deg, self.robot_radius_m)
        desired_speeds = wheel_speeds(vx, vy, w, self.wheel_angles_deg, self.robot_radius_m)
        diffs = [d - p for d, p in zip(desired_speeds, prev_speeds)]
        worst_step = max((abs(d) for d in diffs), default=0.0)

        if worst_step <= max_step or worst_step == 0.0:
            result = (vx, vy, w)
        else:
            scale = max_step / worst_step
            clamped_speeds = [p + diff * scale for p, diff in zip(prev_speeds, diffs)]
            result = body_velocity_from_wheel_speeds(
                clamped_speeds, self.wheel_angles_deg, self.robot_radius_m
            )

        self._prev = result
        self._prev_t = now
        return result
