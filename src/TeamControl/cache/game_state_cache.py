"""Game-controller state cache.

Updates every few ticks (GC packets arrive at ~1 Hz, so there's no
reason to ask `wm` every 16 ms).
"""

REFRESH_EVERY = 10


class GameStateCache:
    """Cached referee state + active-robot count + ball-left-field loc."""

    def __init__(self, wm, refresh_every=REFRESH_EVERY):
        self._wm = wm
        self._refresh_every = refresh_every
        self._tick = 0
        self._state = None
        self._active = 6
        self._blf = None

    def refresh(self):
        self._tick += 1
        if (self._tick % self._refresh_every) != 0:
            return
        try:
            self._state = self._wm.get_game_state()
        except Exception:
            pass
        try:
            self._active = self._wm.get_active_robots()
        except Exception:
            pass
        try:
            self._blf = self._wm.get_ball_left_field_location()
        except Exception:
            pass

    @property
    def state(self):
        return self._state

    @property
    def active_robots(self):
        return self._active

    @property
    def ball_left_field(self):
        return self._blf
