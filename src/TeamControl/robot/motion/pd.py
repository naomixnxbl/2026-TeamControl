import math
import time
from typing import Optional

# Canonical PDController — Motion/controller.py and pd_calibration.py use this.
# Movement.py contains an identical copy kept for historical reasons; prefer this one.


class PDController:
    """
    A reusable PD (Proportional-Derivative) controller.

    The control law:
        u(t) = Kp * e(t)  +  Kd * de(t)/dt

    In English:
      - e(t) is how far you are from the goal RIGHT NOW.
      - de/dt is how FAST that error is changing.
      - Kp * e is the "push" term: far from goal → push hard, close → push gently.
        If you only used P, the robot would overshoot and oscillate, because
        at the target e=0 but the robot is still moving.
      - Kd * de/dt is the "brake" term: if the error is shrinking quickly (the
        robot is closing in), this term opposes the push and damps the motion.
        Think of it as the shock absorber in a car suspension — without it,
        you'd bounce up and down forever.

    Works on either:
      - a scalar error (e.g. angle error, in radians), or
      - a 2D vector error (e.g. (ex, ey) position error, in mm).

    First tick: we can't compute de/dt from one sample, so D = 0 and we fall
    back to P-only until a second sample lets us measure the rate.

    Output saturation ("saturates" = hits a hard ceiling and stops responding
    to more input): if |u| would exceed out_limit, it's clipped. For vectors,
    the whole vector is scaled down so direction is preserved.
    """

    def __init__(self, kp: float, kd: float, out_limit: Optional[float] = None):
        self.kp = kp
        self.kd = kd
        self.out_limit = out_limit
        # Previous-sample memory. Needed to compute de/dt = (e - e_prev)/dt.
        self.prev_error = None
        self.prev_t: Optional[float] = None

    def reset(self) -> None:
        """
        Forget history. Call this when the target changes abruptly or the
        robot stops, so the next de/dt isn't computed against a stale error
        (which would produce a huge, spurious D-term "kick").
        """
        self.prev_error = None
        self.prev_t = None

    def update(self, error, now: Optional[float] = None):
        """
        Feed a new error sample, get back the controller output.
        """
        if now is None:
            now = time.monotonic()

        # Accept either scalar or vector error; remember which so we output
        # the matching shape.
        is_vec = isinstance(error, (tuple, list))
        err = tuple(float(e) for e in error) if is_vec else float(error)

        # --- D term: de/dt, the rate of change of error -----------------
        if self.prev_t is None or self.prev_error is None:
            # No history yet → can't compute a rate. Use 0 so the first tick
            # is pure P. D starts contributing on the second call onward.
            d_err = tuple(0.0 for _ in err) if is_vec else 0.0
        else:
            # dt is small but nonzero. Clamp to 1e-6 to avoid divide-by-zero
            # if two updates happen in the same microsecond.
            dt = max(now - self.prev_t, 1e-6)
            if is_vec:
                d_err = tuple((e - pe) / dt for e, pe in zip(err, self.prev_error))
            else:
                d_err = (err - self.prev_error) / dt

        # Store for next tick's derivative calculation.
        self.prev_error = err
        self.prev_t = now

        # --- Compute u = Kp*e + Kd*de/dt, then saturate -----------------
        if is_vec:
            u = tuple(self.kp * e + self.kd * de for e, de in zip(err, d_err))
            if self.out_limit is not None:
                # For a vector: if the magnitude is too big, scale BOTH
                # components down by the same factor. This preserves the
                # direction of motion — we just drive slower, not sideways.
                mag = math.sqrt(sum(x * x for x in u))
                if mag > self.out_limit and mag > 0:
                    scale = self.out_limit / mag
                    u = tuple(x * scale for x in u)
            return u

        # Scalar path: symmetric clip to [-out_limit, +out_limit].
        u = self.kp * err + self.kd * d_err
        if self.out_limit is not None:
            if u > self.out_limit:
                u = self.out_limit
            elif u < -self.out_limit:
                u = -self.out_limit
        return u
