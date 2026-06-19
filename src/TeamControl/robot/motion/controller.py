import math
import time
from typing import Optional, Tuple

from TeamControl.robot import constants as C
from TeamControl.robot.ball_nav import (
    FieldGeometryCache,
    apply_boundary_braking,
    clamp_for_role,
    predict_position,
    regulate_speed_to_target,
)
from TeamControl.robot.motion.accel import AccelLimiter
from TeamControl.robot.motion.hardware import (
    apply_hardware_gains,
    apply_min_angular_command,
    apply_min_linear_command,
    shorten_target_for_overshoot,
)
from TeamControl.robot.motion.pd import PDController
from TeamControl.robot.motion.settings import PDSettingsStore
from TeamControl.robot.motion.wheel_kinematics import (
    WheelAccelLimiter,
    max_angular_from_wheel_budget,
    scale_to_wheel_speed_limit,
)
from TeamControl.world.transform_cords import world2robot

_MOTION_CONTROLLERS = {}


def get_motion_controller(robot_id, is_yellow: bool = True):
    """Get the one motion brain for this robot."""
    key = (bool(is_yellow), int(robot_id))
    ctrl = _MOTION_CONTROLLERS.get(key)
    if ctrl is None:
        ctrl = RobotMotionController(robot_id=robot_id, is_yellow=is_yellow)
        _MOTION_CONTROLLERS[key] = ctrl
    return ctrl


def _wrap_angle(a: float) -> float:
    """Change any angle into the shortest matching angle."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _speed_from_deadline(dist_mm: float, deadline: float) -> float:
    """How fast to drive so we arrive on time."""
    time_remaining = max(deadline - time.monotonic(), 0.001)
    return min(dist_mm / 1000.0 / time_remaining, C.MAX_SPEED)


def _w_from_deadline(angle_rad: float, deadline: float) -> float:
    """How fast to turn so we arrive on time."""
    time_remaining = max(deadline - time.monotonic(), 0.001)
    return min(abs(angle_rad) / time_remaining, C.MAX_W)


def _run_pd_with_limit(pd: PDController, error, limit: float):
    """Run one PD update with a temporary speed limit."""
    old_limit = pd.out_limit
    pd.out_limit = limit
    try:
        return pd.update(error)
    finally:
        pd.out_limit = old_limit


class RobotMotionController:
    """
    Simple motion controller for one robot.

    Think of it as:
    - turn to face the target
    - drive to the target
    - optionally do both together
    """

    def __init__(
        self,
        robot_id,
        is_yellow: bool = True,
    ):
        self.robot_id = int(robot_id)
        self.is_yellow = bool(is_yellow)
        self.settings = PDSettingsStore()

        # Rule: motion controller is aware of field-size changes.
        self._field_cache = FieldGeometryCache()
        # Rule: this robot's role for stay_in_field's penalty-box clamp.
        self.is_goalie = False
        # Rule: anticipated dt is the measured time between calls (target
        # ~0.05s), used for the predictive lookahead in translational_motion.
        self._last_call_t: Optional[float] = None

        gains = self.settings.load_gains(self.robot_id, self.is_yellow)
        self.speed_scale = float(gains["speed_scale"])
        self.lateral_drift_per_m = float(gains["lateral_drift_per_m"])
        self.stop_overshoot_mm = float(gains["stop_overshoot_mm"])
        self.min_v = float(gains["min_v"])
        self.min_w = float(gains["min_w"])

        self.wheel1_angle_deg = float(gains["wheel1_angle_deg"])
        self.wheel2_angle_deg = float(gains["wheel2_angle_deg"])
        self.wheel3_angle_deg = float(gains["wheel3_angle_deg"])
        self.wheel4_angle_deg = float(gains["wheel4_angle_deg"])
        self.wheel_radius_mm = float(gains["wheel_radius_mm"])
        self.robot_radius_mm = float(gains["robot_radius_mm"])
        # None means "not calibrated yet" -- isotropic limiter stays active.
        self.max_wheel_speed_mps = gains["max_wheel_speed_mps"]
        self.max_wheel_accel_mps2 = gains["max_wheel_accel_mps2"]

        self.angular_pd = PDController(
            kp=gains["turn_kp"],
            kd=gains["turn_kd"],
            out_limit=C.MAX_W,
        )
        self.linear_pd = PDController(
            kp=gains["linear_kp"],
            kd=gains["linear_kd"],
            out_limit=C.MAX_SPEED,
        )

        # Isotropic fallback -- always built, used whenever the wheel spec
        # below isn't active (see _rebuild_wheel_limiters).
        self.linear_accel = AccelLimiter(C.LINEAR_AMAX)
        self.angular_accel = AccelLimiter(C.ANGULAR_AMAX)
        self._rebuild_wheel_limiters()

    def set_role(self, is_goalie: bool) -> None:
        """Set whether this robot is the goalie (used by stay_in_field's
        penalty-box clamp: goalie stays in, non-goalie stays out)."""
        self.is_goalie = bool(is_goalie)

    def _wheel_angles_deg(self) -> Tuple[float, float, float, float]:
        return (
            self.wheel1_angle_deg,
            self.wheel2_angle_deg,
            self.wheel3_angle_deg,
            self.wheel4_angle_deg,
        )

    def _wheel_spec_active(self) -> bool:
        """True once a robot's real wheel motor specs have been measured.

        Geometry (angles/radii) alone isn't enough -- without a measured
        max wheel speed/accel there's nothing to limit against, so the
        isotropic AccelLimiter/MAX_SPEED path stays in charge until both
        are set.
        """
        return self.max_wheel_speed_mps is not None and self.max_wheel_accel_mps2 is not None

    def _rebuild_wheel_limiters(self) -> None:
        """(Re)build the wheel-aware accel limiters from current gains.

        Three independent instances, mirroring how linear_accel/
        angular_accel are already independent -- translational_motion,
        rotational_motion, and tuned_velocity are never the same logical
        "channel" within one tick, so each gets its own limiter state
        rather than sharing one (which would make one channel's call
        starve the other's acceleration budget for that tick).
        """
        if self._wheel_spec_active():
            angles = self._wheel_angles_deg()
            radius_m = self.robot_radius_mm / 1000.0
            accel = self.max_wheel_accel_mps2
            self.wheel_linear_accel = WheelAccelLimiter(angles, radius_m, accel)
            self.wheel_angular_accel = WheelAccelLimiter(angles, radius_m, accel)
            self.wheel_tuned_accel = WheelAccelLimiter(angles, radius_m, accel)
        else:
            self.wheel_linear_accel = None
            self.wheel_angular_accel = None
            self.wheel_tuned_accel = None

    def _cap_wheel_speed(self, vx: float, vy: float, w: float) -> Tuple[float, float, float]:
        """Steady-state cap to the wheel-feasible envelope (no-op if not active)."""
        if not self._wheel_spec_active():
            return vx, vy, w
        return scale_to_wheel_speed_limit(
            vx, vy, w,
            self._wheel_angles_deg(),
            self.robot_radius_mm / 1000.0,
            self.max_wheel_speed_mps,
        )

    def reset(self) -> None:
        """Forget previous errors."""
        self.angular_pd.reset()
        self.linear_pd.reset()
        self.linear_accel.reset()
        self.angular_accel.reset()
        if self.wheel_linear_accel is not None:
            self.wheel_linear_accel.reset()
        if self.wheel_angular_accel is not None:
            self.wheel_angular_accel.reset()
        if self.wheel_tuned_accel is not None:
            self.wheel_tuned_accel.reset()
        self._last_call_t = None

    def get_gains(self) -> dict:
        """Show the gains being used right now."""
        return {
            "turn_kp": self.angular_pd.kp,
            "turn_kd": self.angular_pd.kd,
            "linear_kp": self.linear_pd.kp,
            "linear_kd": self.linear_pd.kd,
            "speed_scale": self.speed_scale,
            "lateral_drift_per_m": self.lateral_drift_per_m,
            "stop_overshoot_mm": self.stop_overshoot_mm,
            "min_v": self.min_v,
            "min_w": self.min_w,
            "wheel1_angle_deg": self.wheel1_angle_deg,
            "wheel2_angle_deg": self.wheel2_angle_deg,
            "wheel3_angle_deg": self.wheel3_angle_deg,
            "wheel4_angle_deg": self.wheel4_angle_deg,
            "wheel_radius_mm": self.wheel_radius_mm,
            "robot_radius_mm": self.robot_radius_mm,
            "max_wheel_speed_mps": self.max_wheel_speed_mps,
            "max_wheel_accel_mps2": self.max_wheel_accel_mps2,
        }

    def apply_gains(self, gains: dict) -> dict:
        """Use these gains now, but do not save them."""
        if "turn_kp" in gains:
            self.angular_pd.kp = float(gains["turn_kp"])
        if "turn_kd" in gains:
            self.angular_pd.kd = float(gains["turn_kd"])
        if "linear_kp" in gains:
            self.linear_pd.kp = float(gains["linear_kp"])
        if "linear_kd" in gains:
            self.linear_pd.kd = float(gains["linear_kd"])
        if "speed_scale" in gains:
            self.speed_scale = float(gains["speed_scale"])
        if "lateral_drift_per_m" in gains:
            self.lateral_drift_per_m = float(gains["lateral_drift_per_m"])
        if "stop_overshoot_mm" in gains:
            self.stop_overshoot_mm = float(gains["stop_overshoot_mm"])
        if "min_v" in gains:
            self.min_v = float(gains["min_v"])
        if "min_w" in gains:
            self.min_w = float(gains["min_w"])
        if "wheel1_angle_deg" in gains:
            self.wheel1_angle_deg = float(gains["wheel1_angle_deg"])
        if "wheel2_angle_deg" in gains:
            self.wheel2_angle_deg = float(gains["wheel2_angle_deg"])
        if "wheel3_angle_deg" in gains:
            self.wheel3_angle_deg = float(gains["wheel3_angle_deg"])
        if "wheel4_angle_deg" in gains:
            self.wheel4_angle_deg = float(gains["wheel4_angle_deg"])
        if "wheel_radius_mm" in gains:
            self.wheel_radius_mm = float(gains["wheel_radius_mm"])
        if "robot_radius_mm" in gains:
            self.robot_radius_mm = float(gains["robot_radius_mm"])
        if "max_wheel_speed_mps" in gains:
            value = gains["max_wheel_speed_mps"]
            self.max_wheel_speed_mps = None if value is None else float(value)
        if "max_wheel_accel_mps2" in gains:
            value = gains["max_wheel_accel_mps2"]
            self.max_wheel_accel_mps2 = None if value is None else float(value)

        self._rebuild_wheel_limiters()
        self.reset()
        return self.get_gains()

    def apply_default_gains(self) -> dict:
        """Use the default gains from constants.py."""
        return self.apply_gains(self.settings.load_default_gains())

    def has_tuned_gains(self) -> bool:
        """True if this robot has saved gains."""
        return self.settings.has_robot_gains(self.robot_id, self.is_yellow)

    def reload_saved_or_default_gains(self) -> tuple[dict, str]:
        """Use saved gains if they exist, otherwise use defaults."""
        gains, source = self.settings.load_gains_with_source(
            self.robot_id,
            self.is_yellow,
        )
        return self.apply_gains(gains), source

    def clear_tuned_gains(self) -> bool:
        """Delete saved gains for this robot, then use defaults."""
        removed = self.settings.delete_robot_gains(
            self.robot_id,
            self.is_yellow,
        )
        self.apply_default_gains()
        return removed

    def calibrate(self, gains: dict, score: Optional[float] = None) -> dict:
        """Use these gains now and save them for next time."""
        applied = self.apply_gains(gains)
        return self.settings.save_gains(
            self.robot_id,
            self.is_yellow,
            applied,
            score=score,
        )

    def is_close_to_target(
        self,
        current_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        threshold_mm: float = 100.0,
    ) -> bool:
        """True when the robot is close enough to stop driving."""
        dx = target_xy[0] - current_xy[0]
        dy = target_xy[1] - current_xy[1]
        close = math.hypot(dx, dy) < threshold_mm
        if close:
            self.linear_pd.reset()
        return close

    def is_facing_dir(
        self,
        current_theta: float,
        target_theta: float,
        threshold_rad: float = 0.1,
    ) -> bool:
        """True when the robot is facing close enough to stop turning."""
        angle_error = _wrap_angle(target_theta - current_theta)
        facing = abs(angle_error) < threshold_rad
        if facing:
            self.angular_pd.reset()
        return facing

    def rotational_motion(
        self,
        current_theta: float,
        target_theta: float,
        deadline: float,
        use_pd: bool = True,
        use_hardware: bool = True,
    ) -> float:
        """Turn only. Returns w in rad/s."""
        angle = _wrap_angle(target_theta - current_theta)
        if abs(angle) < C.ANGLE_EPSILON:
            self.angular_pd.reset()
            return 0.0

        w_limit = _w_from_deadline(angle, deadline)
        if use_pd:
            w = _run_pd_with_limit(self.angular_pd, angle, w_limit)
        else:
            self.angular_pd.reset()
            w = max(-w_limit, min(w_limit, self.angular_pd.kp * angle))

        if use_hardware:
            w = apply_min_angular_command(w, self.min_w)

        if self.wheel_angular_accel is not None:
            _, _, w = self.wheel_angular_accel.limit((0.0, 0.0, w))
            _, _, w = self._cap_wheel_speed(0.0, 0.0, w)
        else:
            w = self.angular_accel.limit(w)

        # Rule: rotation may only claim PD_ANGULAR_WHEEL_BUDGET_SHARE of the
        # wheel budget once it's calibrated -- keeps translation prioritized
        # over spinning.
        if self._wheel_spec_active():
            budget_w = max_angular_from_wheel_budget(
                self.robot_radius_mm / 1000.0,
                self.max_wheel_speed_mps,
                C.PD_ANGULAR_WHEEL_BUDGET_SHARE,
            )
            w = max(-budget_w, min(budget_w, w))
        return w

    def translational_motion(
        self,
        current_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float],
        deadline: float,
        use_pd: bool = True,
        use_hardware: bool = True,
        stay_in_field: bool = False,
    ) -> Tuple[float, float]:
        """Drive only. Returns vx, vy in robot frame.

        stay_in_field=True (renamed from field_limit): remain in field,
        get back into the field if already out -- applies dynamic boundary
        braking, goalie/non-goalie penalty-box clamping, and the goal-post
        no-go zone (shared with ball_nav, see ball_nav.apply_boundary_braking
        / clamp_for_role). False: the robot may exit the field.
        """
        # Rule: aware of any change in field geometry.
        self._field_cache.refresh()

        if stay_in_field:
            target_pos = clamp_for_role(target_pos, self.is_goalie)

        target_in_robot_frame = world2robot(current_pos, target_pos)

        if use_hardware:
            target_in_robot_frame = shorten_target_for_overshoot(
                target_in_robot_frame,
                self.stop_overshoot_mm,
            )

        dist = math.hypot(target_in_robot_frame[0], target_in_robot_frame[1])
        if dist < C.KICKER_ZONE:
            self.linear_pd.reset()
            return 0.0, 0.0

        speed = _speed_from_deadline(dist, deadline)
        if use_pd:
            vx, vy = _run_pd_with_limit(self.linear_pd, target_in_robot_frame, speed)
        else:
            self.linear_pd.reset()
            unit_x = target_in_robot_frame[0] / dist
            unit_y = target_in_robot_frame[1] / dist
            vx, vy = unit_x * speed, unit_y * speed

        if use_hardware:
            vx, vy = apply_hardware_gains(vx, vy, self.get_gains())
            vx, vy = apply_min_linear_command(vx, vy, self.min_v)

        if self.wheel_linear_accel is not None:
            vx, vy, _ = self.wheel_linear_accel.limit((vx, vy, 0.0))
            vx, vy, _ = self._cap_wheel_speed(vx, vy, 0.0)
        else:
            vx, vy = self.linear_accel.limit((vx, vy))

        # Rule: never overshoot -- anticipate where the robot will be next
        # loop (measured dt, target ~0.05s) and cap speed so it can still
        # stop in time at the target. Stateless: recomputed fresh from
        # current state every call, no derivative-of-error term.
        now = time.monotonic()
        dt = (now - self._last_call_t) if self._last_call_t is not None else C.LOOP_RATE
        self._last_call_t = now
        predicted_xy = predict_position(current_pos, vx, vy, dt)
        predicted_dist_mm = math.hypot(
            target_pos[0] - predicted_xy[0], target_pos[1] - predicted_xy[1]
        )
        speed_mps = math.hypot(vx, vy)
        regulated_speed = regulate_speed_to_target(predicted_dist_mm, speed_mps, C.LINEAR_AMAX)
        if speed_mps > 0.0 and regulated_speed < speed_mps:
            scale = regulated_speed / speed_mps
            vx *= scale
            vy *= scale

        if stay_in_field:
            vx, vy = apply_boundary_braking(current_pos, vx, vy)
        return vx, vy

    def general_motion(
        self,
        current_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float],
        target_theta: float,
        deadline: float,
        use_pd: bool = True,
        use_hardware: bool = True,
    ) -> Tuple[float, float, float]:
        """Turn first if badly misaligned, otherwise drive and turn together."""
        current_theta = current_pos[2]
        angle_err = abs(_wrap_angle(target_theta - current_theta))

        # If we point the wrong way, fix that first.
        if angle_err > math.radians(60):
            w = self.rotational_motion(
                current_theta,
                target_theta,
                deadline,
                use_pd,
                use_hardware,
            )
            self.linear_pd.reset()
            return 0.0, 0.0, w

        vx, vy = self.translational_motion(
            current_pos,
            target_pos,
            deadline,
            use_pd,
            use_hardware,
        )
        w = self.rotational_motion(
            current_theta,
            target_theta,
            deadline,
            use_pd,
            use_hardware,
        )

        target_in_robot_frame = world2robot(current_pos, target_pos)
        dist = math.hypot(target_in_robot_frame[0], target_in_robot_frame[1])

        # Drive less while turning. Turn less while far away.
        linear_scale = 1.0 - max(0.0, min(0.8, angle_err / math.pi))
        angular_scale = 1.0 - max(0.0, min(0.6, dist / C.BLEND_DIST))

        final_vx = vx * linear_scale
        final_vy = vy * linear_scale
        final_w = w * angular_scale
        # vx/vy and w were each accel-limited independently above (separate
        # channels, see _rebuild_wheel_limiters); combining them here can
        # still jointly exceed a wheel's steady-state speed even though
        # each was individually fine, so cap the combined triple too.
        return self._cap_wheel_speed(final_vx, final_vy, final_w)

    def face_while_moving(
        self,
        current_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float],
        face_xy: Tuple[float, float],
        deadline: float,
        use_pd: bool = True,
        use_hardware: bool = True,
    ) -> Tuple[float, float, float]:
        """Rule: support facing point A while moving to point B.

        general_motion() already takes position and heading targets
        independently -- this just names the common "face A while moving
        to B" pattern, computing the heading target from face_xy.
        """
        target_theta = math.atan2(
            face_xy[1] - current_pos[1], face_xy[0] - current_pos[0]
        )
        return self.general_motion(
            current_pos, target_pos, target_theta, deadline, use_pd, use_hardware
        )

    def tuned_velocity(
        self,
        vx: float,
        vy: float,
        w: float,
        use_hardware: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Apply this robot's saved motion calibration to an existing velocity.

        This is for behaviours that already made a tactical velocity decision
        such as kick/dribble logic, but still need the same per-robot hardware
        compensation used by the PD target controller.
        """
        vx = float(vx)
        vy = float(vy)
        w = float(w)

        speed = math.hypot(vx, vy)
        if speed > C.MAX_SPEED and speed > 0.0:
            scale = C.MAX_SPEED / speed
            vx *= scale
            vy *= scale

        w = max(-C.MAX_W, min(C.MAX_W, w))

        if use_hardware:
            vx, vy = apply_hardware_gains(vx, vy, self.get_gains())
            vx, vy = apply_min_linear_command(vx, vy, self.min_v)
            w = apply_min_angular_command(w, self.min_w)

            speed = math.hypot(vx, vy)
            if speed > C.MAX_SPEED and speed > 0.0:
                scale = C.MAX_SPEED / speed
                vx *= scale
                vy *= scale
            w = max(-C.MAX_W, min(C.MAX_W, w))

        if self.wheel_tuned_accel is not None:
            vx, vy, w = self.wheel_tuned_accel.limit((vx, vy, w))
            vx, vy, w = self._cap_wheel_speed(vx, vy, w)
        else:
            vx, vy = self.linear_accel.limit((vx, vy))
            w = self.angular_accel.limit(w)
        return vx, vy, w
