import math
import time
from typing import Optional, Tuple, Union


class AccelLimiter:
    """
    Clamps the step between successive velocity commands so the implied
    acceleration never exceeds a_max (m/s² for linear, rad/s² for angular).

    Works on both:
      - scalar velocity  (angular w, rad/s)
      - 2-D vector velocity  ((vx, vy), m/s)

    On the very first call the velocity is accepted as-is so the robot can
    start moving immediately; limiting kicks in from the second call onward.
    """

    def __init__(self, a_max: float):
        self.a_max = float(a_max)
        self._prev: Optional[Union[float, Tuple[float, ...]]] = None
        self._prev_t: Optional[float] = None

    def reset(self) -> None:
        self._prev = None
        self._prev_t = None

    def limit(self, desired, now: Optional[float] = None):
        """
        Return the accel-limited version of *desired*.
        Shape of the return value matches the shape of *desired*.
        """
        if now is None:
            now = time.monotonic()

        is_vec = isinstance(desired, (tuple, list))
        val = tuple(float(v) for v in desired) if is_vec else float(desired)

        if self._prev is None or self._prev_t is None:
            self._prev = val
            self._prev_t = now
            return val

        dt = max(now - self._prev_t, 1e-6)
        max_step = self.a_max * dt

        if is_vec:
            diff = tuple(v - p for v, p in zip(val, self._prev))
            step = math.sqrt(sum(d * d for d in diff))
            if step > max_step and step > 0:
                scale = max_step / step
                result = tuple(p + d * scale for p, d in zip(self._prev, diff))
            else:
                result = val
        else:
            diff = val - self._prev
            result = self._prev + max(-max_step, min(max_step, diff))

        self._prev = result
        self._prev_t = now
        return result
