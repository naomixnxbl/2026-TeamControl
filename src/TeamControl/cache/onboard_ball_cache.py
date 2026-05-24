"""Per-robot onboard ball-observation cache.

Wraps an `OnboardObservationStore` and presents a tidy per-robot view
that's cheap to query each tick. The primary consumer is `BallCache`,
which falls back to onboard bearings when SSL-Vision loses the ball.
"""

import math
import time


# Default freshness window — discard onboard readings older than this.
DEFAULT_MAX_AGE = 1.0  # seconds

# Minimum confidence before we believe an onboard reading.
DEFAULT_MIN_CONFIDENCE = 0.3


class OnboardBallCache:
    """Query surface over a shared OnboardObservationStore.

    The store itself is written by `OnboardReceiver` on a background
    thread; this class only reads, so it's safe to use inside the
    single-threaded tick loop of each robot process.
    """

    def __init__(self, store=None,
                 max_age=DEFAULT_MAX_AGE,
                 min_confidence=DEFAULT_MIN_CONFIDENCE):
        self._store = store
        self.max_age = max_age
        self.min_confidence = min_confidence

    def attach(self, store):
        """Late-bind a store (useful when the receiver is set up after
        the cache is constructed)."""
        self._store = store

    @property
    def store(self):
        return self._store

    def enabled(self):
        return self._store is not None

    def latest(self, is_yellow, robot_id, now=None):
        """Return a usable OnboardObservation or None.

        Filters out stale readings, low-confidence readings, and
        readings where the camera didn't actually see the ball.
        """
        if self._store is None:
            return None
        obs = self._store.get(is_yellow, robot_id, max_age=self.max_age)
        if obs is None or not obs.found:
            return None
        if obs.confidence < self.min_confidence:
            return None
        return obs

    def bearing_world(self, is_yellow, robot_id, robot_pose):
        """Convert the onboard bearing to a world-frame direction vector.

        `robot_pose` is (x, y, theta) with theta in radians. Returns
        (ux, uy, obs) where (ux, uy) is a unit vector pointing from the
        robot toward the ball, or None if no usable reading.
        """
        obs = self.latest(is_yellow, robot_id)
        if obs is None or robot_pose is None:
            return None
        # Camera is roughly forward-facing on the robot.
        # Robot heading + bearing = absolute heading to ball.
        heading = robot_pose[2] + obs.bearing
        return math.cos(heading), math.sin(heading), obs

    def estimate_ball_position(self, is_yellow, robot_id, robot_pose,
                               distance_fn=None):
        """Project the onboard bearing to a world-frame (x, y) estimate.

        Without a calibrated focal length + ball size, the best we can
        do for distance is a rough pixel-radius → mm heuristic. Callers
        can override `distance_fn(radius_px) -> mm` for calibration.
        Returns None if no usable reading.
        """
        if self._store is None or robot_pose is None:
            return None
        obs = self.latest(is_yellow, robot_id)
        if obs is None:
            return None
        if distance_fn is None:
            distance_fn = _default_distance_from_radius
        dist = distance_fn(obs.radius)
        if dist is None or dist <= 0.0:
            return None
        heading = robot_pose[2] + obs.bearing
        return (
            robot_pose[0] + dist * math.cos(heading),
            robot_pose[1] + dist * math.sin(heading),
        )


def _default_distance_from_radius(radius_px):
    """Very rough fallback: SSL golf ball is ~21mm radius. At 320px image
    with ~60deg HFOV, a 1px-radius blob is roughly 4m away. This is
    intentionally coarse — calibrate per camera with `distance_fn`."""
    if radius_px is None or radius_px < 1.0:
        return None
    # Tuned so a 20px radius maps to ~200mm (robot-touching-ball) and
    # 2px maps to ~2000mm. Swap in a calibrated function when available.
    return max(100.0, 4000.0 / radius_px)
