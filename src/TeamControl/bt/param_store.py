"""ParamStore — live-editable parameter registry for BT tuning.

Tree nodes call ``params.get(key, default)`` instead of reading module-level
constants. The dashboard's param editor calls ``params.set(key, value)`` when
a spinner changes, so parameters take effect on the very next BT tick with no
restart required.

Usage in a tree node:
    from TeamControl.bt.param_store import params
    dist = params.get("attacker.possession_dist", 0.122)

Usage from the dashboard:
    from TeamControl.bt.param_store import params
    params.set("attacker.possession_dist", 0.15)
    params.reset_to_defaults()

The ``DEFAULTS`` dict is the single source of truth for every tunable value.
Add a new entry here and it immediately appears in the dashboard param editor
with the correct default.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

DEFAULTS: dict[str, float] = {
    # ── Attacker ──────────────────────────────────────────────────────
    "attacker.possession_dist":        0.122,
    "attacker.possession_heading_tol": 0.30,
    "attacker.shot_corridor_radius":   0.20,
    # ── Goalie ────────────────────────────────────────────────────────
    "goalie.rush_dist":    1.5,
    "goalie.kick_dist":    0.2,
    "goalie.max_advance":  1.1,
    # ── Defender ──────────────────────────────────────────────────────
    "defender.challenge_dist": 0.6,
    "defender.zone_x_ratio":   0.5,
    # ── Supporter ─────────────────────────────────────────────────────
    "supporter.spacing_m": 1.5,
    # ── Coordinator ───────────────────────────────────────────────────
    "coordinator.stopped_speed": 1.4,
}

# Human-readable labels shown in the dashboard param editor.
LABELS: dict[str, str] = {
    "attacker.possession_dist":        "Possession dist (m)",
    "attacker.possession_heading_tol": "Possession heading tol (rad)",
    "attacker.shot_corridor_radius":   "Shot corridor radius (m)",
    "goalie.rush_dist":                "Rush dist (m)",
    "goalie.kick_dist":                "Kick dist (m)",
    "goalie.max_advance":              "Max advance (m)",
    "defender.challenge_dist":         "Challenge dist (m)",
    "defender.zone_x_ratio":           "Zone X ratio",
    "supporter.spacing_m":             "Spacing (m)",
    "coordinator.stopped_speed":       "Stopped max speed (m/s)",
}


class _ParamStore:
    """Thread-safe key-value parameter store with change notification.

    All public methods are safe to call from any process thread. Observer
    callbacks run on whichever thread calls ``set()``.
    """

    def __init__(self) -> None:
        self._values: dict[str, Any] = dict(DEFAULTS)
        self._lock = threading.RLock()
        self._observers: list[Callable[[str, Any], None]] = []

    # ── Read / write ─────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Return the current value for *key*, falling back to *default*.

        If *key* is not in the store and *default* is not None, *default* is
        stored so the value is visible in the dashboard and stable across
        subsequent calls.
        """
        with self._lock:
            if key in self._values:
                return self._values[key]
            if default is not None:
                self._values[key] = default
            return default

    def set(self, key: str, value: Any) -> None:
        """Update *key* and notify all observers."""
        with self._lock:
            self._values[key] = value
        for obs in list(self._observers):
            try:
                obs(key, value)
            except Exception:
                pass

    def all(self) -> dict[str, Any]:
        """Return a snapshot of all current values."""
        with self._lock:
            return dict(self._values)

    def reset_to_defaults(self) -> None:
        """Restore every key to its default value and notify observers."""
        with self._lock:
            self._values = dict(DEFAULTS)
        for obs in list(self._observers):
            try:
                obs("*", None)
            except Exception:
                pass

    # ── Observer registration ─────────────────────────────────────────

    def subscribe(self, fn: Callable[[str, Any], None]) -> None:
        """Register *fn(key, value)* to be called on every ``set()``."""
        if fn not in self._observers:
            self._observers.append(fn)

    def unsubscribe(self, fn: Callable[[str, Any], None]) -> None:
        try:
            self._observers.remove(fn)
        except ValueError:
            pass


# Module-level singleton — import and use directly:
#   from TeamControl.bt.param_store import params
params = _ParamStore()
