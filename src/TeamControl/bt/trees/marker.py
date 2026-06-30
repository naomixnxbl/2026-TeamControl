"""Marker behaviour tree — man-marking role for the GegenPressing strategy.

Topology::

    MarkingSequenceNode (Sequence, memory=False)
    ├── LookAtBall          → stashes angle_to_ball on the tree (no intent)
    └── MarkFallback (Selector — OR logic)
        ├── ShadowOpponent  → IntentMove ball-side of the assigned opponent,
        │                      denying the pass lane; SUCCESS when a man is set
        └── ZoneCover        → IntentMove to a defensive zone slot when this
                               marker has no man (its man left the danger area)

Which opponent a marker shadows is decided **team-level** by the Coordinator
(``_apply_marker_assignment``) and handed to the tree on the blackboard
(``RobotBlackboard.mark_target_id``). Keeping the assignment in the Coordinator
is what stops two markers grabbing the same opponent and lets the assignment
stay stable across ticks (hysteresis). The tree itself is stateless beyond the
``look_angle`` scratch value, exactly like the defender tree.

Rule notes (RoboCup SSL §8.4):
  * The marker sits *ball-side* of its man at a fixed standoff and never drives
    through the opponent, so it cannot commit a *pushing* foul (§8.4.1).
  * Approach speed is capped once inside ``mark_crash_radius`` of the marked
    opponent so the relative speed at any contact stays well under the
    *crashing* threshold (§8.4.2).
  * Markers only ever run during RUNNING — every set-piece / stop phase is
    handled by the Coordinator, which enforces the 0.5 m / 1.5 m/s rules — so a
    marker can never break the defender-too-close or stop-speed rules.

Design notes (shared with the other trees):
  * Snapshot is injected via ``set_snapshot()`` before each ``tick()``; nodes
    read world state through ``self._tree._snapshot`` (read-only).
  * Blackboard is injected via ``tick(blackboard)`` using the ``_blackboard_ref``
    one-element-list protocol. Nodes write ``_blackboard_ref[0].current_intent``.
  * No raw motor commands are produced anywhere in this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from pathlib import Path
import py_trees
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from TeamControl.bt.contracts.blackboard import RobotBlackboard
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot

BT_TUNING_FILENAME = "bt_tuning.yaml"
LEGACY_HEURISTIC_WEIGHT_FILENAME = "heuristic_weight.yaml"

# -----------------------------------------------------------------------
# Tuneable constants
# -----------------------------------------------------------------------

FIELD_HALF_X: float = 4.5
FIELD_HALF_Y: float = 3.0
FIELD_MARGIN: float = 0.2

# Distance the marker holds ball-side of its man (m). Small enough to genuinely
# deny the pass lane, large enough to avoid contact with the marked opponent.
MARK_STANDOFF: float = 0.35
# Within this distance of the marked opponent the approach speed is capped, so
# any contact happens at low relative speed (anti-crash, §8.4.2).
MARK_CRASH_RADIUS: float = 0.5
# Capped approach speed (m/s) used inside MARK_CRASH_RADIUS.
MARK_APPROACH_SPEED: float = 0.9
# How far from our own goal the zone-cover line sits, toward midfield (m).
ZONE_DEPTH: float = 2.0


@dataclass(frozen=True)
class MarkerPositioningConfig:
    """Configurable marker positioning values."""

    mark_standoff: float = MARK_STANDOFF
    mark_crash_radius: float = MARK_CRASH_RADIUS
    mark_approach_speed: float = MARK_APPROACH_SPEED
    zone_depth: float = ZONE_DEPTH
    field_margin: float = FIELD_MARGIN


def load_marker_positioning_config(
    config_filename: str | Path = BT_TUNING_FILENAME,
) -> MarkerPositioningConfig:
    """Load marker-positioning config from yaml, preserving defaults."""

    path = _resolve_utils_config_path(config_filename)

    if not path.exists():
        return MarkerPositioningConfig()

    with open(path, "r") as f:
        raw = yaml.load(f, Loader) or {}
    if not isinstance(raw, Mapping):
        return MarkerPositioningConfig()

    section = _nested_mapping(raw, ("behavior_tree", "marker", "positioning"))
    if section is None:
        section = raw.get("marker_positioning")
    if not isinstance(section, Mapping):
        return MarkerPositioningConfig()

    defaults = MarkerPositioningConfig()
    values: dict[str, float] = {}
    for item in fields(MarkerPositioningConfig):
        if item.name in section:
            values[item.name] = float(section[item.name])
        else:
            values[item.name] = getattr(defaults, item.name)
    return MarkerPositioningConfig(**values)


def _resolve_utils_config_path(config_filename: str | Path) -> Path:
    path = Path(config_filename)
    if path.is_absolute():
        return path

    utils_dir = Path(__file__).resolve().parents[2] / "utils"
    path = utils_dir / path
    if path.exists() or path.name != BT_TUNING_FILENAME:
        return path

    legacy_path = utils_dir / LEGACY_HEURISTIC_WEIGHT_FILENAME
    return legacy_path if legacy_path.exists() else path


def _nested_mapping(raw: Mapping, keys: tuple[str, ...]) -> Mapping | None:
    current: object = raw
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


# -----------------------------------------------------------------------
# Condition / action nodes
# -----------------------------------------------------------------------

class LookAtBall(py_trees.behaviour.Behaviour):
    """Stash the angle from marker to ball on the tree (no intent).

    Mirrors the defender tree: writing an orientation here as an intent would
    clobber the downstream movement intent, so the angle is stashed and the
    movement nodes apply it as their ``target_orientation``.
    """

    def __init__(self, tree_ref: MarkerTree) -> None:
        super().__init__("LookAtBall")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        self._tree.look_angle = math.atan2(
            snap.ball_position[1] - robot.position[1],
            snap.ball_position[0] - robot.position[0],
        )
        return py_trees.common.Status.SUCCESS


class ShadowOpponent(py_trees.behaviour.Behaviour):
    """Mark the assigned opponent ball-side, denying the pass lane to him.

    FAILURE (→ ZoneCover) when this marker has no assigned man this tick or the
    assigned opponent is not in the snapshot. Otherwise writes an IntentMove to
    a point ``mark_standoff`` from the opponent toward the ball, facing the
    ball, with the approach speed capped near the opponent.
    """

    def __init__(self, tree_ref: MarkerTree) -> None:
        super().__init__("ShadowOpponent")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        opp = _find_enemy(snap, bb.mark_target_id)
        if opp is None:
            return py_trees.common.Status.FAILURE

        config = self._tree.positioning_config
        ball = snap.ball_position

        # Stand ball-side of the man at a fixed standoff: this is on the segment
        # between the ball and the opponent, denying the direct pass to him.
        dx = ball[0] - opp.position[0]
        dy = ball[1] - opp.position[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            # Ball is on top of the man (he is essentially contesting the ball);
            # fall back to shielding goal-side toward our own goal.
            gx, gy = self._tree.own_goal_position
            dx = gx - opp.position[0]
            dy = gy - opp.position[1]
            dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        target = (
            opp.position[0] + ux * config.mark_standoff,
            opp.position[1] + uy * config.mark_standoff,
        )
        target = _clamp_field(target, config.field_margin)

        # Anti-crash / anti-push: cap speed once close to the marked opponent so
        # contact, if any, happens at low relative speed (§8.4.2).
        max_speed = None
        if _distance(robot.position, opp.position) <= config.mark_crash_radius:
            max_speed = config.mark_approach_speed

        bb.current_intent = IntentMove(
            target_pos=target,
            target_orientation=self._tree.look_angle,
            max_speed=max_speed,
        )
        bb.intent_source = "ShadowOpponent"
        return py_trees.common.Status.SUCCESS


class ZoneCover(py_trees.behaviour.Behaviour):
    """Hold a defensive zone slot when this marker has no man to shadow.

    The slot sits ``zone_depth`` in front of our own goal, tracking the ball's
    y so the marker stays goal-side of play, with a small per-robot lateral
    band so several zone-covering markers do not stack on the same point.
    """

    def __init__(self, tree_ref: MarkerTree) -> None:
        super().__init__("ZoneCover")
        self._tree = tree_ref

    def update(self) -> py_trees.common.Status:
        snap = self._tree._snapshot
        bb = self._tree._blackboard_ref[0]
        if snap is None or bb is None:
            return py_trees.common.Status.FAILURE

        robot = _find_robot(snap, bb.robot_id)
        if robot is None:
            return py_trees.common.Status.FAILURE

        config = self._tree.positioning_config
        own_goal_x = self._tree.own_goal_position[0]
        # zone_depth in front of our goal, toward midfield.
        zone_x = own_goal_x + self._tree.attack_sign * config.zone_depth
        # Spread zone-covering markers laterally by a stable per-robot band.
        band = ((bb.robot_id % 3) - 1) * 1.0
        zone_y = snap.ball_position[1] + band
        target = _clamp_field((zone_x, zone_y), config.field_margin)

        bb.current_intent = IntentMove(
            target_pos=target,
            target_orientation=self._tree.look_angle,
        )
        bb.intent_source = "ZoneCover"
        return py_trees.common.Status.SUCCESS


# -----------------------------------------------------------------------
# MarkerTree
# -----------------------------------------------------------------------

class MarkerTree:
    """Wrapper around the Marker py_trees topology.

    Usage::

        tree = MarkerTree()
        tree.set_snapshot(snapshot)   # inject world state
        tree.tick(blackboard)          # run tree; writes Intent to blackboard
        intent = blackboard.current_intent
    """

    def __init__(
        self,
        us_positive: bool = True,
        positioning_config: MarkerPositioningConfig | None = None,
        positioning_config_file: str = BT_TUNING_FILENAME,
    ) -> None:
        self._snapshot: Snapshot | None = None
        self._blackboard_ref: list = [None]
        self.look_angle: float = 0.0
        self.positioning_config = (
            positioning_config
            if positioning_config is not None
            else load_marker_positioning_config(positioning_config_file)
        )
        # Convention (matches the other trees): us_positive=True means we are on
        # +x, our own goal is at +x and we attack toward -x.
        self.us_positive = us_positive
        self.own_goal_position: tuple[float, float] = (
            (FIELD_HALF_X, 0.0) if us_positive else (-FIELD_HALF_X, 0.0)
        )
        self.attack_sign: float = -1.0 if us_positive else 1.0
        self.root = self._build_tree()

    # ------------------------------------------------------------------

    def set_snapshot(self, snapshot: Snapshot) -> None:
        """Inject the current world-state snapshot before ticking."""
        self._snapshot = snapshot

    def tick(self, blackboard: RobotBlackboard) -> None:
        """Tick the tree with the given per-robot blackboard."""
        self._blackboard_ref[0] = blackboard
        self.root.tick_once()

    # ------------------------------------------------------------------

    def _build_tree(self) -> py_trees.composites.Sequence:
        mark_fallback = py_trees.composites.Selector(
            name="MarkFallback", memory=False
        )
        mark_fallback.add_children([
            ShadowOpponent(self),
            ZoneCover(self),
        ])

        root = py_trees.composites.Sequence(
            name="MarkingSequenceNode", memory=False
        )
        root.add_children([
            LookAtBall(self),
            mark_fallback,
        ])
        return root


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _find_robot(snap: Snapshot, robot_id: int) -> RobotState | None:
    for r in snap.own_robots:
        if r.robot_id == robot_id:
            return r
    return None


def _find_enemy(snap: Snapshot, robot_id: int | None) -> RobotState | None:
    if robot_id is None:
        return None
    for r in snap.enemy_robots:
        if r.robot_id == robot_id:
            return r
    return None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp_field(
    target: tuple[float, float], margin: float
) -> tuple[float, float]:
    x = max(-FIELD_HALF_X + margin, min(FIELD_HALF_X - margin, target[0]))
    y = max(-FIELD_HALF_Y + margin, min(FIELD_HALF_Y - margin, target[1]))
    return (x, y)
