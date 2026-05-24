"""Robot positions + per-robot derived geometry, keyed by frame version.

`frame.get_yellow_robots(...)` is called once per (team, id) per frame
and the (x, y, theta) tuple is memoized. Iterating both teams or
computing closest-opponent then hits the cache instead of re-walking
the frame's robot list.
"""

import math

from TeamControl.world.transform_cords import world2robot


class RobotCache:
    """Per-frame robot positions and convenience lookups.

    Position tuples are normalized to `(x, y, theta)` floats.
    Missing / non-detected robots resolve to `None`.
    """

    MAX_ROBOT_ID = 16

    def __init__(self):
        self._frame_version = None
        self._frame = None
        self._positions = {}     # (is_yellow, rid) -> (x, y, theta) | None
        self._rel_to_ball = {}   # (is_yellow, rid, ball) -> (rel, dist, ang)

    def update(self, frame, version):
        if version is not None and version == self._frame_version:
            return
        self._frame_version = version
        self._frame = frame
        self._positions.clear()
        self._rel_to_ball.clear()

    def get_position(self, is_yellow, robot_id):
        """Return (x, y, theta) for a robot, or None if not visible."""
        key = (bool(is_yellow), int(robot_id))
        if key in self._positions:
            return self._positions[key]
        pos = self._lookup(is_yellow, robot_id)
        self._positions[key] = pos
        return pos

    def _lookup(self, is_yellow, robot_id):
        if self._frame is None:
            return None
        try:
            r = self._frame.get_yellow_robots(
                isYellow=is_yellow, robot_id=robot_id)
        except Exception:
            return None
        if r is None or isinstance(r, int):
            return None
        try:
            rp = r.position
            return (float(rp[0]), float(rp[1]), float(rp[2]))
        except Exception:
            return None

    def iter_team(self, is_yellow, exclude=None):
        """Yield (robot_id, (x, y, theta)) for each visible robot on a team."""
        for rid in range(self.MAX_ROBOT_ID):
            if exclude is not None and rid == exclude:
                continue
            pos = self.get_position(is_yellow, rid)
            if pos is not None:
                yield rid, pos

    def relative_to_ball(self, is_yellow, robot_id, ball):
        """Return (rel_xy, distance, angle) of ball in robot frame.

        Cached per (team, id, ball) — the ball key changes when the ball
        moves, so this is a *first-time-this-frame* memo, not a stale one.
        """
        if ball is None:
            return None
        key = (bool(is_yellow), int(robot_id), ball)
        cached = self._rel_to_ball.get(key)
        if cached is not None:
            return cached
        rpos = self.get_position(is_yellow, robot_id)
        if rpos is None:
            return None
        rel = world2robot(rpos, ball)
        d = math.hypot(rel[0], rel[1])
        ang = math.atan2(rel[1], rel[0])
        result = (rel, d, ang)
        self._rel_to_ball[key] = result
        return result

    def closest_opponent(self, is_yellow_us, ball):
        """Return (robot_id, (x, y, theta), distance) of nearest opponent
        to the ball, or None if no opponents visible."""
        if ball is None:
            return None
        best = None
        best_d = float("inf")
        for rid, pos in self.iter_team(not is_yellow_us):
            d = math.hypot(pos[0] - ball[0], pos[1] - ball[1])
            if d < best_d:
                best_d = d
                best = (rid, pos, d)
        return best

    @property
    def frame(self):
        return self._frame

    @property
    def version(self):
        return self._frame_version
