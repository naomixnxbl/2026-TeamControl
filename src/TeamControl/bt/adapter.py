"""Adapter layer — bridges 2026 TeamControl runtime with the TurtleRabbitBT
behaviour tree pipeline.

The TurtleRabbitBT pipeline is:

    WorldModel  ──(this module)──> Snapshot
                                       │
                                Coordinator.tick()
                                       │
                                 list[Intent]
                                       │
                          ──(this module)──> RobotCommand → dispatcher_q

Two responsibilities live here so the BT core stays free of any 2026-specific
types:

* ``build_snapshot_from_world_model`` — read the latest vision frame from the
  shared ``WorldModel`` and produce a frozen ``Snapshot``.
* ``intent_to_robot_command`` — resolve an ``Intent`` into a ``RobotCommand``
  using the existing stateless skill functions.

Intent kinds not yet wired to physical kicker/dribbler hardware fall through
to a zero-velocity command with a TODO marker, so the surrounding pipeline
keeps running while higher fidelity is added.
"""
from __future__ import annotations

import math
import time
from typing import Iterable

from TeamControl.bt.contracts.intent import (
    Intent,
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentOrient,
    IntentPass,
    IntentReceive,
)
from TeamControl.bt.contracts.motion_target import MotionTarget
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.skills.kick_at import kick_at
from TeamControl.bt.skills.move_to import move_to
from TeamControl.network.robot_command import RobotCommand
from TeamControl.SSL.game_controller.common import GameState
from TeamControl.planner import PlannerAPI, PlannerInput
from TeamControl.robot.constants import CRUISE_SPEED
from TeamControl.robot.motion import get_motion_controller
from TeamControl.world.field_config import (
    FIELD_X_MAX,
    FIELD_X_MIN,
    FIELD_Y_MAX,
    FIELD_Y_MIN,
    VORONOI_ARRIVAL_DEADZONE_MM,
    VORONOI_CHASE_SPEED_SCALE,
    VORONOI_DENSITY_PERCENT,
    VORONOI_HORIZON_MS,
    VORONOI_MAX_DENSITY_NODES,
    VORONOI_WAYPOINT_REACHED_MM,
)

# Runtime world state comes in as raw SSL/grSim millimetres.
_MM_TO_M = 0.001
_M_TO_MM = 1000.0
_VORONOI_NAV_SPEED = CRUISE_SPEED * VORONOI_CHASE_SPEED_SCALE


# Map every GC-produced GameState into the BT's GamePhase.
# The GC FSM already resolves ours-vs-theirs before storing GameState,
# so no extra colour check is needed here.
_PHASE_MAP = {
    GameState.HALTED:          GamePhase.HALTED,
    GameState.HALF_TIME:       GamePhase.HALF_TIME,
    GameState.STOPPED:         GamePhase.STOPPED,
    GameState.OUR_PREPARE_KICKOFF: GamePhase.PREPARE_KICKOFF,
    GameState.ENEMY_KICKOFF:    GamePhase.ENEMY_KICKOFF,
    GameState.OUR_KICKOFF:         GamePhase.KICKOFF,
    GameState.OUR_FREE_KICK:       GamePhase.FREE_KICK,
    GameState.ENEMY_FREE_KICK:   GamePhase.ENEMY_FREE_KICK,
    GameState.OUR_BALL_PLACEMENT:  GamePhase.BALL_PLACEMENT,
    GameState.ENEMY_BALL_PLACEMENT:  GamePhase.BALL_PLACEMENT,
    GameState.OUR_PREPARE_PENALTY:     GamePhase.PREPARE_PENALTY,
    GameState.ENEMY_PREPARE_PENALTY: GamePhase.PREPARE_PENALTY_OPP,
    GameState.OUR_PENALTY_SHOOTOUT:       GamePhase.PENALTY_SHOOT,
    GameState.ENEMY_PENALTY_SHOOTOUT:      GamePhase.PENALTY_DEFEND,
    GameState.RUNNING:         GamePhase.RUNNING,
}


def _phase_from_state(state) -> GamePhase:
    if state is None:
        return GamePhase.RUNNING  # sandbox / no-GC default
    return _PHASE_MAP.get(state, GamePhase.RUNNING)


def _mm_to_m(value: float) -> float:
    return float(value) * _MM_TO_M


def _ball_pos_vel(frame) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (ball_position, ball_velocity) tuples from the latest frame.

    Velocity is not currently estimated by ``Frame``/``Ball``; emit (0,0) and
    leave the proper estimator hookup as a follow-up.
    """
    ball = frame.ball if frame is not None else None
    if ball is None:
        return (0.0, 0.0), (0.0, 0.0)
    # Vision protocol sends positions in mm; BT uses metres.
    return (_mm_to_m(ball.x), _mm_to_m(ball.y)), (0.0, 0.0)


def _team_to_states(team) -> tuple[RobotState, ...]:
    out: list[RobotState] = []
    for robot in team:  # iteration only yields active Robot instances
        out.append(
            RobotState(
                robot_id=int(robot.id),
                # Vision protocol sends positions in mm; BT uses metres.
                position=(_mm_to_m(robot.x), _mm_to_m(robot.y)),
                orientation=float(robot.o),
            )
        )
    return tuple(out)


def build_snapshot_from_world_model(wm, is_yellow: bool | None = None) -> Snapshot | None:
    """Build a frozen ``Snapshot`` from the latest data in ``WorldModel``.

    ``is_yellow`` selects whose perspective the snapshot is built from. Pass
    it explicitly when more than one BT process shares the same WorldModel
    (e.g. 6v6 simulation). When omitted, falls back to ``wm.us_yellow()``.

    Returns ``None`` when no vision frame has been received yet — callers
    should skip the tick in that case.
    """
    frame = wm.get_latest_frame()
    if frame is None:
        return None

    ball_pos, ball_vel = _ball_pos_vel(frame)

    if is_yellow is None:
        is_yellow = bool(wm.us_yellow())
    own_team = frame.robots_yellow if is_yellow else frame.robots_blue
    enemy_team = frame.robots_blue if is_yellow else frame.robots_yellow

    raw_placement = wm.get_ball_placement_pos()
    placement_pos = (float(raw_placement[0]), float(raw_placement[1])) if raw_placement else None

    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        own_robots=_team_to_states(own_team),
        enemy_robots=_team_to_states(enemy_team),
        referee_state=RefereeState(
            game_phase=_phase_from_state(wm.get_game_state()),
            score=(0, 0),  # TODO: read from wm.ref_data once exposed
            ball_placement_pos=placement_pos,
        ),
    )


def _robot_id_of(intent: Intent, fallback: int) -> int:
    # Most intents do not carry a robot id; the caller already knows which
    # blackboard produced them. The Coordinator could be extended to carry
    # robot_id alongside the intent — for now the runner supplies it.
    return fallback


def intent_to_motion_target(
    intent: Intent, robot_id: int, snapshot: Snapshot
) -> MotionTarget | None:
    """Resolve an ``Intent`` to a ``MotionTarget`` via the skill layer.

    Returns ``None`` when the robot is absent from the snapshot or when the
    intent kind has no skill mapping yet.
    """
    try:
        if isinstance(intent, IntentMove):
            return move_to(
                snapshot,
                robot_id,
                intent.target_pos,
                intent.target_orientation,
                intent.max_speed,
            )
        if isinstance(intent, IntentKick):
            return kick_at(snapshot, robot_id, intent.target_pos)
        if isinstance(intent, IntentDribble):
            # No dribble skill yet — drive toward the target like move_to,
            # but compute target_orientation so the robot faces the dribble
            # target. Without this, move_to defaults orientation to 0.0 and
            # the robot rotates east while trying to dribble west, leaving
            # the ball behind.
            robot = next(
                (r for r in snapshot.own_robots if r.robot_id == robot_id), None
            )
            if robot is not None:
                angle = math.atan2(
                    intent.target_pos[1] - robot.position[1],
                    intent.target_pos[0] - robot.position[0],
                )
            else:
                angle = None
            return move_to(snapshot, robot_id, intent.target_pos, angle)
        if isinstance(intent, IntentPass):
            return kick_at(snapshot, robot_id, intent.target_pos)
        if isinstance(intent, IntentOrient):
            return MotionTarget(
                target_velocity=(0.0, 0.0),
                target_orientation=float(intent.target_orientation),
                arrival_mode="precision",
            )
        if isinstance(intent, IntentReceive):
            return MotionTarget(
                target_velocity=(0.0, 0.0),
                target_orientation=0.0,
                arrival_mode="normal",
            )
    except ValueError:
        # Robot absent from snapshot — skip the tick for it.
        return None
    return None


def _angular_velocity_to_target(
    current_orientation: float, target_orientation: float, gain: float = 4.0
) -> float:
    """Wrap angle error to [-pi, pi] and apply a proportional gain."""
    err = (target_orientation - current_orientation + math.pi) % (2 * math.pi) - math.pi
    return float(max(-6.0, min(6.0, err * gain)))


def intent_to_robot_command(
    intent: Intent,
    robot_id: int,
    snapshot: Snapshot,
    is_yellow: bool,
) -> RobotCommand | None:
    """End-to-end: Intent → MotionTarget → RobotCommand wire packet."""
    target = intent_to_motion_target(intent, robot_id, snapshot)
    if target is None:
        return None

    # Pull the current orientation so we can compute an angular velocity AND
    # rotate the velocity from world frame into the robot's body frame.
    robot = next(
        (r for r in snapshot.own_robots if r.robot_id == robot_id), None
    )
    current_o = robot.orientation if robot is not None else 0.0
    w = _angular_velocity_to_target(current_o, target.target_orientation)

    # Skills produce target_velocity in WORLD frame. grSim's RobotCommand
    # expects body-frame velocities (veltangent = forward, velnormal = left).
    # Without this rotation the robot drives correctly only when its heading
    # happens to be ~0 — for any other heading the motion direction is
    # rotated by -orientation, producing curved / circular paths.
    vx_world, vy_world = target.target_velocity
    cos_o = math.cos(current_o)
    sin_o = math.sin(current_o)
    vt = vx_world * cos_o + vy_world * sin_o    # forward along heading
    vn = -vx_world * sin_o + vy_world * cos_o   # left perpendicular

    kick = 1 if isinstance(intent, (IntentKick, IntentPass)) else 0
    dribble = 1 if isinstance(intent, IntentDribble) else 0

    return RobotCommand(
        robot_id=robot_id,
        vx=float(vt),
        vy=float(vn),
        w=float(w),
        kick=kick,
        dribble=dribble,
        isYellow=bool(is_yellow),
    )


class VoronoiRouter:
    """Voronoi/Dijkstra planner state for one BT process.

    One instance covers all robots in the process; the underlying
    ``VoronoiWaypointManager`` separates state by ``(is_yellow, robot_id)``
    so sharing is safe.
    """

    def __init__(self) -> None:
        self._planner = PlannerAPI(
            density_percent=VORONOI_DENSITY_PERCENT,
            max_density_nodes=VORONOI_MAX_DENSITY_NODES,
        )
        self._active_targets: dict[int, tuple[float, float, float]] = {}

    def get_waypoint_velocity(
        self,
        robot_id: int,
        is_yellow: bool,
        current_pos_m: tuple[float, float],
        orientation: float,
        target_pos_m: tuple[float, float],
        wm,
        now_s: float,
        stay_in_field: bool = True,
    ) -> tuple[float, float]:
        """Return body-frame (vx, vy) in m/s toward the next Voronoi waypoint.

        Converts meter-scale snapshot positions to mm for the planner, then
        maps the resulting waypoint to a proportional velocity and rotates it
        into robot body-frame. Returns (0.0, 0.0) on any planning failure.

        When *stay_in_field* is True (default), returns (0, 0) if the target
        is outside the configured field boundary so robots don't chase the ball
        out of bounds.
        """
        rx = current_pos_m[0] * _M_TO_MM
        ry = current_pos_m[1] * _M_TO_MM
        tx = target_pos_m[0] * _M_TO_MM
        ty = target_pos_m[1] * _M_TO_MM

        # Stop if the target is outside the field (e.g. ball rolled out).
        if stay_in_field and not (FIELD_X_MIN <= tx <= FIELD_X_MAX and FIELD_Y_MIN <= ty <= FIELD_Y_MAX):
            return 0.0, 0.0

        # Within arrival deadzone — drive straight to target, skip the planner.
        robot_to_target_mm = math.hypot(tx - rx, ty - ry)
        if robot_to_target_mm < VORONOI_ARRIVAL_DEADZONE_MM:
            if robot_to_target_mm < 1.0:
                return 0.0, 0.0
            speed = min(robot_to_target_mm * _MM_TO_M, _VORONOI_NAV_SPEED)
            vx_w = (tx - rx) / robot_to_target_mm * speed
            vy_w = (ty - ry) / robot_to_target_mm * speed
            cos_o, sin_o = math.cos(orientation), math.sin(orientation)
            return vx_w * cos_o + vy_w * sin_o, -vx_w * sin_o + vy_w * cos_o

        active = self._active_targets.get(robot_id)
        reached = (
            active is not None
            and math.hypot(active[0] - rx, active[1] - ry) <= VORONOI_WAYPOINT_REACHED_MM
        )
        try:
            obstacles = wm.get_planning_obstacles(
                now_s=now_s,
                horizon_ms=VORONOI_HORIZON_MS,
                ignore_robots=((bool(is_yellow), int(robot_id)),),
            )
            plan_out = self._planner.plan(PlannerInput(
                robot_id=robot_id,
                is_yellow=is_yellow,
                current_pose=(rx, ry, orientation),
                target_pose=(tx, ty, 0.0),
                obstacles=obstacles,
                clearance_mm=0.0,
                robot_reached_current_waypoint=reached,
                now_s=now_s,
            ))
        except Exception:
            return 0.0, 0.0

        self._active_targets[robot_id] = plan_out.active_target_pose
        wx, wy = plan_out.active_target_pose[0], plan_out.active_target_pose[1]
        dx, dy = wx - rx, wy - ry
        dist_mm = math.hypot(dx, dy)
        if dist_mm < 1.0:
            return 0.0, 0.0
        speed = min(dist_mm * _MM_TO_M, _VORONOI_NAV_SPEED)
        vx_w = dx / dist_mm * speed
        vy_w = dy / dist_mm * speed
        cos_o, sin_o = math.cos(orientation), math.sin(orientation)
        return vx_w * cos_o + vy_w * sin_o, -vx_w * sin_o + vy_w * cos_o


class PDRouter:
    """Voronoi/Dijkstra planner + RobotMotionController for one BT process.

    Identical interface to :class:`VoronoiRouter`.  The only difference is the
    velocity calculation: instead of a simple proportional vector,
    ``get_motion_controller(...).translational_motion()`` runs the full PD
    control loop (accel limiting, hardware gains, field clamping).
    ``translational_motion`` already returns body-frame velocities, so no
    world-to-body rotation is needed here.
    """

    def __init__(self) -> None:
        self._planner = PlannerAPI(
            density_percent=VORONOI_DENSITY_PERCENT,
            max_density_nodes=VORONOI_MAX_DENSITY_NODES,
        )
        self._active_targets: dict[int, tuple[float, float, float]] = {}

    def get_waypoint_velocity(
        self,
        robot_id: int,
        is_yellow: bool,
        current_pos_m: tuple[float, float],
        orientation: float,
        target_pos_m: tuple[float, float],
        wm,
        now_s: float,
        stay_in_field: bool = True,
    ) -> tuple[float, float]:
        """Return body-frame (vx, vy) in m/s toward the next Voronoi waypoint.

        Uses RobotMotionController.translational_motion for full PD control
        (accel limiting, hardware gains, field clamping). Pose units are metres
        from the snapshot; the planner works in mm internally.

        When *stay_in_field* is True (default), returns (0, 0) if the target
        is outside the configured field boundary so robots don't chase the ball
        out of bounds.
        """
        rx = current_pos_m[0] * _M_TO_MM
        ry = current_pos_m[1] * _M_TO_MM
        tx = target_pos_m[0] * _M_TO_MM
        ty = target_pos_m[1] * _M_TO_MM

        # Stop if the target is outside the field (e.g. ball rolled out).
        if stay_in_field and not (FIELD_X_MIN <= tx <= FIELD_X_MAX and FIELD_Y_MIN <= ty <= FIELD_Y_MAX):
            return 0.0, 0.0

        # Within arrival deadzone — hand off directly to PD controller, skip the planner.
        robot_to_target_mm = math.hypot(tx - rx, ty - ry)
        if robot_to_target_mm < VORONOI_ARRIVAL_DEADZONE_MM:
            motion_ctrl = get_motion_controller(robot_id, is_yellow)
            deadline = time.monotonic() + 0.5
            return motion_ctrl.translational_motion(
                (rx, ry, orientation), (tx, ty), deadline, stay_in_field=True
            )

        active = self._active_targets.get(robot_id)
        reached = (
            active is not None
            and math.hypot(active[0] - rx, active[1] - ry) <= VORONOI_WAYPOINT_REACHED_MM
        )
        try:
            obstacles = wm.get_planning_obstacles(
                now_s=now_s,
                horizon_ms=VORONOI_HORIZON_MS,
                ignore_robots=((bool(is_yellow), int(robot_id)),),
            )
            plan_out = self._planner.plan(PlannerInput(
                robot_id=robot_id,
                is_yellow=is_yellow,
                current_pose=(rx, ry, orientation),
                target_pose=(tx, ty, 0.0),
                obstacles=obstacles,
                clearance_mm=0.0,
                robot_reached_current_waypoint=reached,
                now_s=now_s,
            ))
        except Exception:
            return 0.0, 0.0

        self._active_targets[robot_id] = plan_out.active_target_pose
        wx, wy = plan_out.active_target_pose[0], plan_out.active_target_pose[1]
        motion_ctrl = get_motion_controller(robot_id, is_yellow)
        deadline = time.monotonic() + 0.5
        return motion_ctrl.translational_motion(
            (rx, ry, orientation), (wx, wy), deadline, stay_in_field=True
        )


def dispatch_coordinator_output(
    coordinator,
    robot_ids: Iterable[int],
    snapshot: Snapshot,
    is_yellow: bool,
    dispatcher_q,
    run_time: float = 1.0,
    wm=None,
    router: "VoronoiRouter | PDRouter | None" = None,
) -> int:
    """Walk each robot's blackboard after a ``Coordinator.tick`` and emit a
    ``RobotCommand`` per non-empty intent.

    When *router* and *wm* are both supplied, ``IntentMove`` and
    ``IntentDribble`` are routed through the Voronoi/Dijkstra planner for
    obstacle-aware navigation. All other intents (kick, pass, orient, receive)
    keep the original snapshot-based skill implementations.

    Reading the intent off the per-robot blackboard is more robust than
    aligning the ``Coordinator.tick`` list with ``robot_ids``: skipped robots
    (absent from snapshot, or which produced no intent) leave gaps in the
    returned list that are easy to misalign.
    """
    use_voronoi = router is not None and wm is not None
    now_s = time.time() if use_voronoi else 0.0
    sent = 0
    for rid in robot_ids:
        bb = coordinator.blackboards.get(rid)
        if bb is None or bb.current_intent is None:
            continue
        intent = bb.current_intent

        if use_voronoi and isinstance(intent, (IntentMove, IntentDribble)):
            robot = next((r for r in snapshot.own_robots if r.robot_id == rid), None)
            if robot is None:
                continue
            vx_b, vy_b = router.get_waypoint_velocity(
                rid, is_yellow,
                robot.position, robot.orientation,
                intent.target_pos, wm, now_s,
            )
            if isinstance(intent, IntentDribble):
                target_o = math.atan2(
                    intent.target_pos[1] - robot.position[1],
                    intent.target_pos[0] - robot.position[0],
                )
                dribble = 1
            else:
                target_o = (
                    intent.target_orientation
                    if intent.target_orientation is not None
                    else robot.orientation
                )
                dribble = 0
            w = _angular_velocity_to_target(robot.orientation, target_o)
            cmd = RobotCommand(
                robot_id=rid,
                vx=float(vx_b),
                vy=float(vy_b),
                w=float(w),
                kick=0,
                dribble=dribble,
                isYellow=bool(is_yellow),
            )
        else:
            cmd = intent_to_robot_command(intent, rid, snapshot, is_yellow)

        if cmd is None:
            continue
        if not dispatcher_q.full():
            dispatcher_q.put([cmd, run_time])
            sent += 1
    return sent
