"""Final rule-following guard rails for BT intents.

This module keeps match-safety and rule-compliance rewrites separate from role
selection and tree behavior. Behavior trees can choose tactical intents freely;
this layer makes last-mile adjustments for things like field bounds, defense
areas, and goalie-box restrictions.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.intent import (
    Intent,
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentPass,
)
from TeamControl.bt.contracts.snapshot import GamePhase, Snapshot


@dataclass(frozen=True)
class MovementSafetyConfig:
    """Final movement target guard rails loaded from sim config."""

    keep_robots_in_bounds: bool = True
    keep_goalie_in_goal_box: bool = True
    keep_non_goalies_out_of_goalie_box: bool = True
    avoid_ball_touch_in_opponent_defense_area: bool = True
    field_length: float = 9.0
    field_width: float = 6.0
    field_margin: float = 0.05
    goalie_box_depth: float = 1.0
    goalie_box_width: float = 2.0
    goalie_box_margin: float = 0.05
    goalie_box_avoid_margin: float = 0.15
    goalie_box_exit_margin: float = 0.10
    defense_area_ball_touch_margin: float = 0.18
    defense_area_dribble_kick_margin: float = 0.30

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, bool | float] | None,
    ) -> "MovementSafetyConfig":
        if raw is None:
            return cls()
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        return cls(**values)


def has_rule_following_enabled(config: MovementSafetyConfig) -> bool:
    return (
        config.keep_robots_in_bounds
        or config.keep_goalie_in_goal_box
        or config.keep_non_goalies_out_of_goalie_box
        or config.avoid_ball_touch_in_opponent_defense_area
    )


def apply_rule_following(
    *,
    snapshot: Snapshot,
    robot_id: int,
    robot_pos: tuple[float, float],
    role: RoleType,
    intent: Intent,
    config: MovementSafetyConfig,
    own_goal_line_x: float,
    opponent_goal: tuple[float, float],
) -> Intent:
    """Apply final safety/rule rewrites to a single robot intent."""
    is_goalie = role == RoleType.GOALIE
    intent = _avoid_opponent_defense_area_ball_touch(
        snapshot=snapshot,
        robot_id=robot_id,
        robot_pos=robot_pos,
        is_goalie=is_goalie,
        intent=intent,
        config=config,
        opponent_goal=opponent_goal,
    )
    if config.keep_non_goalies_out_of_goalie_box and not is_goalie:
        intent = _avoid_non_goalie_own_goalie_box_intent(
            snapshot=snapshot,
            robot_pos=robot_pos,
            intent=intent,
            config=config,
            own_goal_line_x=own_goal_line_x,
        )
    if not isinstance(intent, (IntentMove, IntentDribble)):
        return intent

    target = intent.target_pos
    if config.keep_robots_in_bounds:
        target = _clamp_to_field(target, config)
    if config.keep_goalie_in_goal_box and is_goalie:
        target = _clamp_to_goalie_box(target, config, own_goal_line_x)
    if config.keep_non_goalies_out_of_goalie_box and not is_goalie:
        target = _avoid_own_goalie_box(
            robot_pos=robot_pos,
            target=target,
            config=config,
            own_goal_line_x=own_goal_line_x,
        )

    if target == intent.target_pos:
        return intent
    return replace(intent, target_pos=target)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        return value
    return max(minimum, min(maximum, value))


def _distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _distance_to_rect(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> float:
    min_x, max_x, min_y, max_y = bounds
    x, y = point
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return math.hypot(dx, dy)


def _should_kick_dribble_before_defense_area(
    ball: tuple[float, float],
    touch_bounds: tuple[float, float, float, float],
    config: MovementSafetyConfig,
) -> bool:
    return (
        _distance_to_rect(ball, touch_bounds)
        <= max(0.0, config.defense_area_dribble_kick_margin)
    )


def _point_in_rect(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    min_x, max_x, min_y, max_y = bounds
    x, y = point
    return min_x <= x <= max_x and min_y <= y <= max_y


def _segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    if _point_in_rect(start, bounds) or _point_in_rect(end, bounds):
        return True

    min_x, max_x, min_y, max_y = bounds
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    t0 = 0.0
    t1 = 1.0

    for p, q in (
        (-dx, x0 - min_x),
        (dx, max_x - x0),
        (-dy, y0 - min_y),
        (dy, max_y - y0),
    ):
        if abs(p) < 1e-12:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)

    return t0 <= t1


def _is_ball_target(
    target: tuple[float, float],
    ball: tuple[float, float],
    tolerance: float = 0.10,
) -> bool:
    return _distance(target, ball) <= tolerance


def _clamp_to_field(
    target: tuple[float, float],
    config: MovementSafetyConfig,
) -> tuple[float, float]:
    half_length = max(0.0, config.field_length * 0.5 - max(0.0, config.field_margin))
    half_width = max(0.0, config.field_width * 0.5 - max(0.0, config.field_margin))
    return (
        _clamp(float(target[0]), -half_length, half_length),
        _clamp(float(target[1]), -half_width, half_width),
    )


def _clamp_to_goalie_box(
    target: tuple[float, float],
    config: MovementSafetyConfig,
    own_goal_line_x: float,
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = _goalie_box_bounds(config, own_goal_line_x)
    return (
        _clamp(float(target[0]), min_x, max_x),
        _clamp(float(target[1]), min_y, max_y),
    )


def _avoid_opponent_defense_area_ball_touch(
    *,
    snapshot: Snapshot,
    robot_id: int,
    robot_pos: tuple[float, float],
    is_goalie: bool,
    intent: Intent,
    config: MovementSafetyConfig,
    opponent_goal: tuple[float, float],
) -> Intent:
    if (
        not config.avoid_ball_touch_in_opponent_defense_area
        or snapshot.referee_state.game_phase != GamePhase.RUNNING
        or is_goalie
    ):
        return intent

    avoid_margin = max(0.0, config.goalie_box_avoid_margin)
    touch_margin = max(avoid_margin, config.defense_area_ball_touch_margin)
    bounds = _opponent_defense_area_bounds(config, opponent_goal, expand=avoid_margin)
    touch_bounds = _opponent_defense_area_bounds(
        config,
        opponent_goal,
        expand=touch_margin,
    )
    robot_in_area = _point_in_rect(robot_pos, bounds)
    ball_in_area = _point_in_rect(snapshot.ball_position, bounds)
    ball_in_touch_area = _point_in_rect(snapshot.ball_position, touch_bounds)
    ball_exit_bounds = (
        bounds
        if robot_in_area or ball_in_area
        else touch_bounds
    )

    if isinstance(intent, (IntentKick, IntentPass)):
        if robot_in_area or ball_in_area:
            return IntentMove(
                target_pos=_attacker_foul_avoidance_target(
                    robot_pos,
                    snapshot.ball_position,
                    ball_exit_bounds,
                ),
                target_orientation=None,
            )
        return intent

    if isinstance(intent, IntentDribble):
        target = intent.target_pos
        if robot_in_area:
            return IntentMove(
                target_pos=_nearest_box_exit(robot_pos, bounds),
                target_orientation=None,
            )
        if ball_in_area:
            return IntentMove(
                target_pos=_nearest_box_exit(
                    snapshot.ball_position,
                    ball_exit_bounds,
                ),
                target_orientation=None,
            )
        dribble_enters_area = (
            _point_in_rect(target, bounds)
            or _segment_intersects_rect(robot_pos, target, bounds)
        )
        if dribble_enters_area and _should_kick_dribble_before_defense_area(
            snapshot.ball_position,
            touch_bounds,
            config,
        ):
            return IntentKick(target_pos=target)
        if _point_in_rect(target, bounds):
            return replace(intent, target_pos=_nearest_box_exit(target, bounds))
        if _segment_intersects_rect(robot_pos, target, bounds):
            return replace(
                intent,
                target_pos=_box_entry_guard_point(robot_pos, target, bounds),
            )
        return intent

    if isinstance(intent, IntentMove) and _is_ball_target(
        intent.target_pos,
        snapshot.ball_position,
    ):
        if robot_in_area:
            return replace(
                intent,
                target_pos=_nearest_box_exit(robot_pos, bounds),
            )
        if ball_in_touch_area:
            return replace(
                intent,
                target_pos=_nearest_box_exit(
                    snapshot.ball_position,
                    ball_exit_bounds,
                ),
            )

    return intent


def _avoid_own_goalie_box(
    *,
    robot_pos: tuple[float, float],
    target: tuple[float, float],
    config: MovementSafetyConfig,
    own_goal_line_x: float,
) -> tuple[float, float]:
    avoid_margin = max(0.0, config.goalie_box_avoid_margin)
    touch_margin = max(avoid_margin, config.defense_area_ball_touch_margin)
    bounds = _goalie_box_bounds(
        config,
        own_goal_line_x,
        expand=avoid_margin,
    )
    touch_bounds = _goalie_box_bounds(
        config,
        own_goal_line_x,
        expand=touch_margin,
    )
    exit_margin = max(0.02, config.goalie_box_exit_margin)
    if _point_in_rect(robot_pos, bounds):
        return _nearest_box_exit(robot_pos, bounds, pad=exit_margin)
    if _point_in_rect(target, bounds):
        return _safe_box_exit_target(
            robot_pos,
            target,
            bounds,
            pad=exit_margin,
        )
    if _segment_intersects_rect(robot_pos, target, bounds):
        return _box_reroute_waypoint(
            robot_pos,
            target,
            bounds,
            pad=exit_margin,
        )
    return target


def _avoid_non_goalie_own_goalie_box_intent(
    *,
    snapshot: Snapshot,
    robot_pos: tuple[float, float],
    intent: Intent,
    config: MovementSafetyConfig,
    own_goal_line_x: float,
) -> Intent:
    avoid_margin = max(0.0, config.goalie_box_avoid_margin)
    touch_margin = max(avoid_margin, config.defense_area_ball_touch_margin)
    bounds = _goalie_box_bounds(
        config,
        own_goal_line_x,
        expand=avoid_margin,
    )
    touch_bounds = _goalie_box_bounds(
        config,
        own_goal_line_x,
        expand=touch_margin,
    )
    exit_margin = max(0.02, config.goalie_box_exit_margin)

    if _point_in_rect(robot_pos, bounds):
        return IntentMove(
            target_pos=_nearest_box_exit(robot_pos, bounds, pad=exit_margin),
            target_orientation=None,
        )

    ball_in_bounds = _point_in_rect(snapshot.ball_position, bounds)
    ball_in_touch_bounds = _point_in_rect(snapshot.ball_position, touch_bounds)
    ball_exit_bounds = bounds if ball_in_bounds else touch_bounds
    if isinstance(intent, (IntentKick, IntentPass)) and ball_in_touch_bounds:
        return IntentMove(
            target_pos=_safe_box_exit_target(
                robot_pos,
                snapshot.ball_position,
                ball_exit_bounds,
                pad=exit_margin,
            ),
            target_orientation=None,
        )

    if (
        isinstance(intent, IntentMove)
        and ball_in_touch_bounds
        and _is_ball_target(intent.target_pos, snapshot.ball_position)
    ):
        return IntentMove(
            target_pos=_safe_box_exit_target(
                robot_pos,
                snapshot.ball_position,
                ball_exit_bounds,
                pad=exit_margin,
            ),
            target_orientation=None,
            max_speed=intent.max_speed,
        )

    if isinstance(intent, IntentDribble) and ball_in_touch_bounds:
        return IntentMove(
            target_pos=_safe_box_exit_target(
                robot_pos,
                snapshot.ball_position,
                ball_exit_bounds,
                pad=exit_margin,
            ),
            target_orientation=None,
        )

    return intent


def _goalie_box_bounds(
    config: MovementSafetyConfig,
    own_goal_line_x: float,
    expand: float = 0.0,
) -> tuple[float, float, float, float]:
    return _defense_area_bounds(
        config,
        goal_line_x=own_goal_line_x,
        expand=expand,
    )


def _opponent_defense_area_bounds(
    config: MovementSafetyConfig,
    opponent_goal: tuple[float, float],
    expand: float = 0.0,
) -> tuple[float, float, float, float]:
    return _defense_area_bounds(
        config,
        goal_line_x=opponent_goal[0],
        expand=expand,
    )


def _defense_area_bounds(
    config: MovementSafetyConfig,
    goal_line_x: float,
    expand: float = 0.0,
) -> tuple[float, float, float, float]:
    field_half_length = max(0.0, config.field_length * 0.5)
    field_margin = max(0.0, config.field_margin)
    box_margin = max(0.0, config.goalie_box_margin)
    box_depth = max(0.0, config.goalie_box_depth)
    expand = max(0.0, expand)

    outer_abs_x = max(0.0, field_half_length - field_margin)
    inner_abs_x = max(0.0, field_half_length - box_depth + box_margin)
    inner_abs_x = min(inner_abs_x, outer_abs_x)
    protected_inner_abs_x = max(0.0, inner_abs_x - expand)

    side = 1.0 if goal_line_x >= 0.0 else -1.0
    if side > 0.0:
        min_x, max_x = protected_inner_abs_x, outer_abs_x
    else:
        min_x, max_x = -outer_abs_x, -protected_inner_abs_x

    half_box_width = max(0.0, config.goalie_box_width * 0.5 - box_margin)
    min_y = -half_box_width - expand
    max_y = half_box_width + expand
    return (min_x, max_x, min_y, max_y)


def _nearest_box_exit(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
    pad: float = 0.02,
) -> tuple[float, float]:
    candidates = _box_exit_candidates(point, bounds, pad=pad)
    return min(candidates, key=lambda candidate: _distance(point, candidate))


def _safe_box_exit_target(
    robot_pos: tuple[float, float],
    blocked_target: tuple[float, float],
    bounds: tuple[float, float, float, float],
    pad: float = 0.02,
) -> tuple[float, float]:
    candidates = _box_exit_candidates(blocked_target, bounds, pad=pad)
    safe_candidates = [
        candidate
        for candidate in candidates
        if not _segment_intersects_rect(robot_pos, candidate, bounds)
    ]
    if not safe_candidates:
        return _box_reroute_waypoint(
            robot_pos,
            blocked_target,
            bounds,
            pad=pad,
        )
    return min(
        safe_candidates,
        key=lambda candidate: (
            _distance(blocked_target, candidate)
            + 0.25 * _distance(robot_pos, candidate)
        ),
    )


def _box_exit_candidates(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
    pad: float = 0.02,
) -> list[tuple[float, float]]:
    min_x, max_x, min_y, max_y = bounds
    x, y = point
    pad = max(0.0, pad)
    center_x = (min_x + max_x) * 0.5
    if center_x >= 0.0:
        inner_candidate = (min_x - pad, _clamp(y, min_y, max_y))
    else:
        inner_candidate = (max_x + pad, _clamp(y, min_y, max_y))
    return [
        inner_candidate,
        (_clamp(x, min_x, max_x), min_y - pad),
        (_clamp(x, min_x, max_x), max_y + pad),
    ]


def _box_reroute_waypoint(
    robot_pos: tuple[float, float],
    target: tuple[float, float],
    bounds: tuple[float, float, float, float],
    pad: float = 0.02,
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    pad = max(0.0, pad)
    center_x = (min_x + max_x) * 0.5
    if center_x >= 0.0:
        inner_x = min_x - pad
    else:
        inner_x = max_x + pad
    candidates = [
        (inner_x, min_y - pad),
        (inner_x, max_y + pad),
    ]
    return min(
        candidates,
        key=lambda candidate: _distance(robot_pos, candidate)
        + _distance(candidate, target),
    )


def _box_entry_guard_point(
    robot_pos: tuple[float, float],
    target: tuple[float, float],
    bounds: tuple[float, float, float, float],
    pad: float = 0.02,
) -> tuple[float, float]:
    """Return a point just before a segment enters a protected box."""
    min_x, max_x, min_y, max_y = bounds
    sx, sy = robot_pos
    tx, ty = target
    dx = tx - sx
    dy = ty - sy
    pad = max(0.0, pad)
    candidates: list[tuple[float, float, float]] = []

    if abs(dx) > 1e-9:
        for x in (min_x, max_x):
            t = (x - sx) / dx
            if 0.0 <= t <= 1.0:
                y = sy + dy * t
                if min_y <= y <= max_y:
                    candidates.append((t, x, y))

    if abs(dy) > 1e-9:
        for y in (min_y, max_y):
            t = (y - sy) / dy
            if 0.0 <= t <= 1.0:
                x = sx + dx * t
                if min_x <= x <= max_x:
                    candidates.append((t, x, y))

    if not candidates:
        return _box_reroute_waypoint(robot_pos, target, bounds)

    _, entry_x, entry_y = min(candidates, key=lambda candidate: candidate[0])
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return _nearest_box_exit(robot_pos, bounds)
    return (
        entry_x - (dx / distance) * pad,
        entry_y - (dy / distance) * pad,
    )


def _attacker_foul_avoidance_target(
    robot_pos: tuple[float, float],
    ball_pos: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    if _point_in_rect(robot_pos, bounds):
        return _nearest_box_exit(robot_pos, bounds)
    return _nearest_box_exit(ball_pos, bounds)
