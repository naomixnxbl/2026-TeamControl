"""Team-color / side-of-field flags.

These change rarely (only on a referee SWITCH_TEAM event), so the
TeamCache polls `wm` lazily and caches `goal_x` derivatives so robot
loops don't repeat the same `wm.us_positive()` try/except every tick.
"""

from TeamControl.robot.constants import HALF_LEN

# How many ticks between forced refreshes of team flags.
REFRESH_EVERY = 30


class TeamCache:
    """Cached team-side state read from WorldModel."""

    def __init__(self, wm, refresh_every=REFRESH_EVERY):
        self._wm = wm
        self._refresh_every = refresh_every
        self._tick = 0
        self._us_yellow = True
        self._us_positive = True
        self._initialized = False

    def refresh(self):
        self._tick += 1
        if self._initialized and (self._tick % self._refresh_every) != 0:
            return
        try:
            self._us_yellow = bool(self._wm.us_yellow())
        except Exception:
            pass
        try:
            self._us_positive = bool(self._wm.us_positive())
        except Exception:
            pass
        self._initialized = True

    @property
    def us_yellow(self):
        return self._us_yellow

    @property
    def us_positive(self):
        return self._us_positive

    def goal_x(self, is_yellow):
        """Defending-goal x for a given team color, given current side."""
        if is_yellow:
            return -HALF_LEN if self._us_positive else HALF_LEN
        return HALF_LEN if self._us_positive else -HALF_LEN

    def their_goal_x(self, is_yellow):
        return -self.goal_x(is_yellow)

    def force_refresh(self):
        self._initialized = False
        self.refresh()
