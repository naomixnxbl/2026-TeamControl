"""
Per-process tick cache for changing/dynamic game data.

Each robot process runs in its own Python process, so this is an
in-process cache (no shared memory). One `TickCache(wm)` is created
per robot; call `refresh(now)` once per tick before reading.

Categories live in their own modules and are exposed as attributes
on `TickCache`:

    cache.ball   -> BallCache       (position, history, velocity, prediction)
    cache.robots -> RobotCache      (positions + relative-to-ball vectors)
    cache.team   -> TeamCache       (us_yellow, us_positive, goal_x)
    cache.game   -> GameStateCache  (current GC state)

Invalidation is keyed on `wm.get_version()` — derived values are
recomputed only when the underlying frame advances.
"""

from .base import VersionedCache
from .ball_cache import BallCache
from .robot_cache import RobotCache
from .team_cache import TeamCache
from .game_state_cache import GameStateCache
from .onboard_ball_cache import OnboardBallCache
from .tick_cache import TickCache

__all__ = [
    "VersionedCache",
    "BallCache",
    "RobotCache",
    "TeamCache",
    "GameStateCache",
    "OnboardBallCache",
    "TickCache",
]
