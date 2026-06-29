"""Line-of-sight helpers for tactics-level decisions.

The helpers here operate on simple ``(x, y)`` points and snapshot robot-like
objects. They do not depend on py_trees or coordinator state, so attacker,
supporter, passing, and future role assignment logic can all reuse them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

Point = tuple[float, float]


@dataclass(frozen=True)
class LineBlocker:
    """Object close enough to block a line segment."""

    robot_id: int | None
    position: Point
    distance_to_line: float


@dataclass(frozen=True)
class LineOfSightResult:
    """Detailed result for a line-of-sight check."""

    start: Point
    end: Point
    clearance: float
    blockers: tuple[LineBlocker, ...]

    @property
    def is_clear(self) -> bool:
        return len(self.blockers) == 0


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""

    return math.hypot(a[0] - b[0], a[1] - b[1])


def face_angle(start: Point, target: Point) -> float:
    """Heading angle from ``start`` toward ``target``."""

    return math.atan2(target[1] - start[1], target[0] - start[0])


def angle_error(target: float, current: float) -> float:
    """Signed smallest angular difference, wrapped to ``[-pi, pi]``."""

    return (target - current + math.pi) % (2 * math.pi) - math.pi


def point_to_segment_dist(point: Point, seg_a: Point, seg_b: Point) -> float:
    """Shortest distance from ``point`` to line segment ``seg_a -> seg_b``."""

    ax, ay = seg_a
    bx, by = seg_b
    px, py = point

    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        return distance(point, seg_a)

    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + t * abx, ay + t * aby)
    return distance(point, closest)


def evaluate_line_of_sight(
    start: Point,
    end: Point,
    obstacles: Iterable[object],
    *,
    clearance: float = 0.18,
    ignore_robot_ids: Sequence[int] = (),
) -> LineOfSightResult:
    """Return whether the path from ``start`` to ``end`` is blocked.

    ``obstacles`` may contain ``RobotState``-like objects with ``robot_id`` and
    ``position`` attributes, or raw ``(x, y)`` points.
    """

    ignored = set(ignore_robot_ids)
    blockers: list[LineBlocker] = []

    for obstacle in obstacles:
        robot_id = _robot_id(obstacle)
        if robot_id is not None and robot_id in ignored:
            continue

        pos = _position(obstacle)
        if pos is None:
            continue

        dist = point_to_segment_dist(pos, start, end)
        if dist <= clearance:
            blockers.append(
                LineBlocker(
                    robot_id=robot_id,
                    position=pos,
                    distance_to_line=dist,
                )
            )

    blockers.sort(key=lambda b: b.distance_to_line)
    return LineOfSightResult(
        start=start,
        end=end,
        clearance=clearance,
        blockers=tuple(blockers),
    )


def line_of_sight_clear(
    start: Point,
    end: Point,
    obstacles: Iterable[object],
    *,
    clearance: float = 0.18,
    ignore_robot_ids: Sequence[int] = (),
) -> bool:
    """Return ``True`` when no obstacle blocks the line segment."""

    return evaluate_line_of_sight(
        start,
        end,
        obstacles,
        clearance=clearance,
        ignore_robot_ids=ignore_robot_ids,
    ).is_clear


def pass_lane_clear(
    passer_pos: Point,
    receiver_pos: Point,
    obstacles: Iterable[object],
    *,
    clearance: float = 0.18,
    ignore_robot_ids: Sequence[int] = (),
) -> bool:
    """Alias for pass-specific readability."""

    return line_of_sight_clear(
        passer_pos,
        receiver_pos,
        obstacles,
        clearance=clearance,
        ignore_robot_ids=ignore_robot_ids,
    )


def _position(obstacle: object) -> Point | None:
    pos = getattr(obstacle, "position", obstacle)
    if not isinstance(pos, tuple) and not isinstance(pos, list):
        return None
    if len(pos) < 2:
        return None
    return (float(pos[0]), float(pos[1]))


def _robot_id(obstacle: object) -> int | None:
    rid = getattr(obstacle, "robot_id", None)
    return int(rid) if rid is not None else None

