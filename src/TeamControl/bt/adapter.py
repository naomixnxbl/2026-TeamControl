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
from TeamControl.bt.skills.receive_ball import receive_ball
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.pd_controller import PDController
from TeamControl.SSL.game_controller.common import GameState

# Runtime world state comes in as raw SSL/grSim millimetres.
_MM_TO_M = 0.001
DRIBBLE_DISTANCE_LIMIT_M = 1.0  # SSL §8.4.2: robots may not dribble > 1 m continuously

BALL_TARGET_EPSILON = 0.08
BALL_APPROACH_STOP_DISTANCE = 0.35
BALL_APPROACH_HEADING_TOL = 0.45
BALL_APPROACH_SLOW_SPEED = 0.45

DRIBBLE_CONTROL_DISTANCE = 0.16
DRIBBLE_HEADING_TOL = 0.45
DRIBBLE_APPROACH_SPEED = 0.55
DRIBBLE_CARRY_SPEED = 1.0

KICK_APPROACH_OFFSET = 0.22
KICK_APPROACH_TOL = 0.08
KICK_CONTACT_DISTANCE = 0.18
# Heading error (rad) between the robot's facing and the ball→target line
# allowed before the kick fires. 0.22 rad (~13°) keeps shots/passes on target
# (at 2 m a 13° error is ~0.45 m — inside the goal mouth) while staying loose
# enough that the robot still fires; the approach logic rotates it under this
# before driving through the ball.
KICK_ALIGN_TOL = 0.22
KICK_BALL_FRONT_TOL = 0.45
KICK_APPROACH_SPEED = 1.0
KICK_CONTACT_SPEED = 1.5


# Map every GC-produced GameState into the BT's GamePhase.
# The GC FSM already resolves ours-vs-theirs before storing GameState,
# so no extra colour check is needed here.
_PHASE_MAP = {
    GameState.HALTED:          GamePhase.HALTED,
    GameState.HALF_TIME:       GamePhase.HALF_TIME,
    GameState.STOPPED:         GamePhase.STOPPED,
    GameState.OUR_PREPARE_KICKOFF:   GamePhase.PREPARE_KICKOFF,
    GameState.ENEMY_PREPARE_KICKOFF: GamePhase.ENEMY_KICKOFF,
    GameState.ENEMY_KICKOFF:         GamePhase.ENEMY_KICKOFF,
    GameState.OUR_KICKOFF:         GamePhase.KICKOFF,
    GameState.OUR_FREE_KICK:       GamePhase.FREE_KICK,
    GameState.ENEMY_FREE_KICK:   GamePhase.ENEMY_FREE_KICK,
    GameState.OUR_BALL_PLACEMENT:  GamePhase.BALL_PLACEMENT,
    GameState.ENEMY_BALL_PLACEMENT:  GamePhase.BALL_PLACEMENT,
    GameState.OUR_PREPARE_PENALTY: GamePhase.PREPARE_PENALTY,
    GameState.ENEMY_PREPARE_PENALTY: GamePhase.PREPARE_PENALTY_OPP,
    GameState.OUR_PENALTY_SHOOTOUT: GamePhase.PENALTY_SHOOT,
    GameState.ENEMY_PENALTY_SHOOTOUT: GamePhase.PENALTY_DEFEND,
    GameState.RUNNING:         GamePhase.RUNNING,
}


def _phase_from_state(state) -> GamePhase:
    if state is None:
        return GamePhase.RUNNING  # sandbox / no-GC default
    return _PHASE_MAP.get(state, GamePhase.RUNNING)


_ENEMY_PHASE_MAP = {
    GamePhase.KICKOFF: GamePhase.ENEMY_KICKOFF,
    GamePhase.ENEMY_KICKOFF: GamePhase.KICKOFF,
    GamePhase.FREE_KICK: GamePhase.ENEMY_FREE_KICK,
    GamePhase.ENEMY_FREE_KICK: GamePhase.FREE_KICK,
    GamePhase.PREPARE_PENALTY: GamePhase.PREPARE_PENALTY_OPP,
    GamePhase.PREPARE_PENALTY_OPP: GamePhase.PREPARE_PENALTY,
    GamePhase.PENALTY_SHOOT: GamePhase.PENALTY_DEFEND,
    GamePhase.PENALTY_DEFEND: GamePhase.PENALTY_SHOOT,
}


def _phase_for_perspective(wm, is_yellow: bool) -> GamePhase:
    """Return GamePhase from the requested team's perspective."""
    phase = _phase_from_state(wm.get_game_state())
    wm_yellow = bool(wm.us_yellow())
    if is_yellow == wm_yellow:
        return phase
    return _ENEMY_PHASE_MAP.get(phase, phase)


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


def build_snapshot_from_world_model(
    wm,
    is_yellow: bool | None = None,
    active_robot_ids: Iterable[int] | None = None,
) -> Snapshot | None:
    """Build a frozen ``Snapshot`` from the latest data in ``WorldModel``.

    ``is_yellow`` selects whose perspective the snapshot is built from. Pass
    it explicitly when more than one BT process shares the same WorldModel
    (e.g. 6v6 simulation). When omitted, falls back to ``wm.us_yellow()``.

    ``active_robot_ids`` limits which same-colour robots are treated as
    controllable teammates. Same-colour robots outside this set are still
    included as obstacles by moving them into ``enemy_robots``.

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
    own_states = _team_to_states(own_team)
    enemy_states = _team_to_states(enemy_team)
    if active_robot_ids is not None:
        active_ids = {int(robot_id) for robot_id in active_robot_ids}
        active_own_states = tuple(
            robot for robot in own_states if robot.robot_id in active_ids
        )
        inactive_own_states = tuple(
            robot for robot in own_states if robot.robot_id not in active_ids
        )
        own_states = active_own_states
        enemy_states = enemy_states + inactive_own_states

    get_placement = getattr(wm, "get_ball_placement_pos", None)
    raw_placement = get_placement() if get_placement is not None else None
    placement_pos = (float(raw_placement[0]), float(raw_placement[1])) if raw_placement else None

    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        own_robots=own_states,
        enemy_robots=enemy_states,
        referee_state=RefereeState(
            game_phase=_phase_for_perspective(wm, is_yellow),
            score=(0, 0),  # TODO: read from wm.ref_data once exposed
            ball_placement_pos=placement_pos,
        ),
    )


def _robot_id_of(intent: Intent, fallback: int) -> int:
    # Most intents do not carry a robot id; the caller already knows which
    # blackboard produced them. The Coordinator could be extended to carry
    # robot_id alongside the intent — for now the runner supplies it.
    return fallback


def _get_robot(snapshot: Snapshot, robot_id: int) -> RobotState:
    for robot in snapshot.own_robots:
        if robot.robot_id == robot_id:
            return robot
    raise ValueError(f"Robot {robot_id} not found in snapshot")


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _face_angle(
    source: tuple[float, float],
    target: tuple[float, float],
) -> float:
    return math.atan2(target[1] - source[1], target[0] - source[0])


def _angle_error(target: float, current: float) -> float:
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def _limit_velocity(
    velocity: tuple[float, float],
    max_speed: float,
) -> tuple[float, float]:
    vx, vy = velocity
    speed = math.hypot(vx, vy)
    if speed < 1e-9 or speed <= max_speed:
        return velocity
    scale = max_speed / speed
    return (vx * scale, vy * scale)


def _is_ball_target(
    target_pos: tuple[float, float],
    ball_pos: tuple[float, float],
) -> bool:
    return _distance(target_pos, ball_pos) <= BALL_TARGET_EPSILON


def _ball_is_in_front(robot: RobotState, ball_pos: tuple[float, float]) -> bool:
    angle_to_ball = _face_angle(robot.position, ball_pos)
    return abs(_angle_error(angle_to_ball, robot.orientation)) <= DRIBBLE_HEADING_TOL


def _kick_angle(ball_pos: tuple[float, float], target_pos: tuple[float, float]) -> float:
    return _face_angle(ball_pos, target_pos)


def _kick_pose_ready(
    snapshot: Snapshot,
    robot_id: int,
    target_pos: tuple[float, float],
) -> bool:
    robot = _get_robot(snapshot, robot_id)
    ball = snapshot.ball_position
    angle_ball_to_target = _kick_angle(ball, target_pos)
    angle_robot_to_ball = _face_angle(robot.position, ball)
    dist_to_ball = _distance(robot.position, ball)

    return (
        dist_to_ball <= KICK_CONTACT_DISTANCE
        and abs(_angle_error(angle_robot_to_ball, robot.orientation)) <= KICK_BALL_FRONT_TOL
        and abs(_angle_error(angle_ball_to_target, robot.orientation)) <= KICK_ALIGN_TOL
    )


def _guard_ball_approach(
    snapshot: Snapshot,
    robot_id: int,
    target: MotionTarget,
) -> MotionTarget:
    robot = _get_robot(snapshot, robot_id)
    angle_to_ball = _face_angle(robot.position, snapshot.ball_position)
    heading_err = abs(_angle_error(angle_to_ball, robot.orientation))
    dist_to_ball = _distance(robot.position, snapshot.ball_position)

    if (
        dist_to_ball <= BALL_APPROACH_STOP_DISTANCE
        and heading_err > BALL_APPROACH_HEADING_TOL
    ):
        return MotionTarget(
            target_velocity=(0.0, 0.0),
            target_orientation=angle_to_ball,
            arrival_mode="precision",
        )

    velocity = target.target_velocity
    if heading_err > BALL_APPROACH_HEADING_TOL:
        velocity = _limit_velocity(velocity, BALL_APPROACH_SLOW_SPEED)

    return MotionTarget(
        target_velocity=velocity,
        target_orientation=angle_to_ball,
        arrival_mode=target.arrival_mode,
    )


def _dribble_motion_target(
    snapshot: Snapshot,
    robot_id: int,
    target_pos: tuple[float, float],
) -> MotionTarget:
    robot = _get_robot(snapshot, robot_id)
    ball = snapshot.ball_position
    angle_to_ball = _face_angle(robot.position, ball)
    heading_err = abs(_angle_error(angle_to_ball, robot.orientation))
    dist_to_ball = _distance(robot.position, ball)

    if dist_to_ball > DRIBBLE_CONTROL_DISTANCE or not _ball_is_in_front(robot, ball):
        if (
            dist_to_ball <= BALL_APPROACH_STOP_DISTANCE
            and heading_err > DRIBBLE_HEADING_TOL
        ):
            return MotionTarget(
                target_velocity=(0.0, 0.0),
                target_orientation=angle_to_ball,
                arrival_mode="precision",
            )
        return move_to(
            snapshot,
            robot_id,
            ball,
            angle_to_ball,
            DRIBBLE_APPROACH_SPEED,
        )

    target_angle = _face_angle(robot.position, target_pos)
    return MotionTarget(
        target_velocity=(0.0, 0.0),
        target_orientation=target_angle,
        arrival_mode="precision",
    )


def _kick_motion_target(
    snapshot: Snapshot,
    robot_id: int,
    target_pos: tuple[float, float],
) -> MotionTarget:
    robot = _get_robot(snapshot, robot_id)
    ball = snapshot.ball_position
    angle = _kick_angle(ball, target_pos)
    ux = math.cos(angle)
    uy = math.sin(angle)
    approach = (
        ball[0] - ux * KICK_APPROACH_OFFSET,
        ball[1] - uy * KICK_APPROACH_OFFSET,
    )

    if _kick_pose_ready(snapshot, robot_id, target_pos):
        return kick_at(snapshot, robot_id, target_pos)

    dist_to_ball = _distance(robot.position, ball)

    # Already contact-close: rotate in place to face the kick angle.
    # Moving backward to the approach position (below) would break possession
    # because the approach is computed 0.22 m behind the ball — farther than
    # KICK_APPROACH_TOL (0.08 m) from a robot that already has the ball at ~0.11 m.
    if dist_to_ball <= KICK_CONTACT_DISTANCE:
        return MotionTarget(
            target_velocity=(0.0, 0.0),
            target_orientation=angle,
            arrival_mode="precision",
        )

    heading_err = abs(_angle_error(angle, robot.orientation))
    angle_to_ball = _face_angle(robot.position, ball)
    ball_front_err = abs(_angle_error(angle_to_ball, robot.orientation))
    behind_ball = (
        (robot.position[0] - ball[0]) * ux
        + (robot.position[1] - ball[1]) * uy
    ) < -0.04

    if not behind_ball or _distance(robot.position, approach) > KICK_APPROACH_TOL:
        return move_to(
            snapshot,
            robot_id,
            approach,
            angle,
            KICK_APPROACH_SPEED,
        )

    if (
        dist_to_ball <= BALL_APPROACH_STOP_DISTANCE
        and (
            heading_err > KICK_ALIGN_TOL
            or ball_front_err > KICK_BALL_FRONT_TOL
        )
    ):
        return MotionTarget(
            target_velocity=(0.0, 0.0),
            target_orientation=angle,
            arrival_mode="precision",
        )

    return move_to(
        snapshot,
        robot_id,
        ball,
        angle,
        KICK_CONTACT_SPEED,
    )


class DribbleLimitTracker:
    """Tracks continuous dribble distance per robot to enforce SSL §8.4.2.

    A robot may not dribble the ball more than 1 metre without releasing it.
    The limit is measured as straight-line displacement from the ball position
    when dribbling began. Releasing the ball (kick, pass, or losing possession)
    resets the counter.
    """

    def __init__(self, max_dribble_distance_m: float = DRIBBLE_DISTANCE_LIMIT_M) -> None:
        self.max_dribble_distance_m = float(max_dribble_distance_m)
        self._dribble_start_pos: dict[int, tuple[float, float]] = {}

    def should_enable_dribbler(
        self,
        robot_id: int,
        wants_dribble: bool,
        *,
        ball_pos: tuple[float, float] | None = None,
    ) -> bool:
        if not wants_dribble:
            self._dribble_start_pos.pop(robot_id, None)
            return False

        if ball_pos is None:
            return True

        if robot_id not in self._dribble_start_pos:
            self._dribble_start_pos[robot_id] = ball_pos
            return True

        start = self._dribble_start_pos[robot_id]
        dist = math.hypot(ball_pos[0] - start[0], ball_pos[1] - start[1])
        return dist < self.max_dribble_distance_m


def load_dribble_distance_limit(config_filename: str = "bt_tuning.yaml") -> float:
    """Read rule_limits.dribble_distance_limit_m from bt_tuning.yaml."""
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        path = _Path(__file__).resolve().parents[1] / "utils" / config_filename
        with open(path) as f:
            raw = _yaml.safe_load(f) or {}
        return float(raw.get("rule_limits", {}).get("dribble_distance_limit_m", DRIBBLE_DISTANCE_LIMIT_M))
    except Exception:
        return DRIBBLE_DISTANCE_LIMIT_M


def intent_to_motion_target(
    intent: Intent, robot_id: int, snapshot: Snapshot
) -> MotionTarget | None:
    """Resolve an ``Intent`` to a ``MotionTarget`` via the skill layer.

    Returns ``None`` when the robot is absent from the snapshot or when the
    intent kind has no skill mapping yet.
    """
    try:
        if isinstance(intent, IntentMove):
            target = move_to(
                snapshot,
                robot_id,
                intent.target_pos,
                intent.target_orientation,
                intent.max_speed,
                gain=intent.speed_gain,
            )
            if _is_ball_target(intent.target_pos, snapshot.ball_position):
                return _guard_ball_approach(snapshot, robot_id, target)
            return target
        if isinstance(intent, IntentKick):
            return _kick_motion_target(snapshot, robot_id, intent.target_pos)
        if isinstance(intent, IntentDribble):
            return _dribble_motion_target(snapshot, robot_id, intent.target_pos)
        if isinstance(intent, IntentPass):
            return _kick_motion_target(snapshot, robot_id, intent.target_pos)
        if isinstance(intent, IntentOrient):
            return MotionTarget(
                target_velocity=(0.0, 0.0),
                target_orientation=float(intent.target_orientation),
                arrival_mode="precision",
            )
        if isinstance(intent, IntentReceive):
            # Present the dribbler to the incoming ball: hold position but face
            # the ball (receive_ball orients toward it) instead of staring at
            # heading 0.0, which left the ball arriving behind/beside the kicker.
            return receive_ball(snapshot, robot_id)
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
    try:
        robot = _get_robot(snapshot, robot_id)
    except ValueError:
        return None
    current_o = robot.orientation
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

    wants_kick = isinstance(intent, (IntentKick, IntentPass))
    kick = (
        1
        if wants_kick and _kick_pose_ready(snapshot, robot_id, intent.target_pos)
        else 0
    )
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


class _RobotMovement:
    """Per-robot motion state. Holds the heading PD controller so each robot
    keeps its own derivative history (no shared state between robots)."""

    def __init__(self, angular_kp: float, angular_kd: float, angular_limit: float) -> None:
        self.angular_pd = PDController(angular_kp, angular_kd, angular_limit)
        # Phase for the left↔right dribble nudge (keeps the ball on the dribbler).
        self.dribble_phase: int = 0


class MotionExecutor:
    """Stateful PD-backed motion layer — the unified Intent → RobotCommand path.

    Resolves each Intent the same way as :func:`intent_to_robot_command` (same
    skill layer, ball-approach guard, kick/dribble flags, world→body rotation),
    but replaces the memoryless proportional heading gain with a **per-robot PD
    controller**. The derivative term brakes the turn as the heading error
    shrinks, damping the overshoot/oscillation the plain-P gain produced. Linear
    velocity still comes from the skill layer so the guards and kick contact
    speeds are preserved untouched.

    One :class:`PDController` per robot, created lazily, so robots never share
    derivative state. Drop-in for the legacy path: ``run_bt_v2_process`` builds
    one executor and routes every robot's command through it.
    """

    ANGULAR_KP: float = 4.0   # matches the legacy proportional heading gain
    ANGULAR_KD: float = 0.35  # derivative braking (damps turn overshoot)
    ANGULAR_LIMIT: float = 6.0
    # Dribble nudge: while carrying the ball, weave the body left↔right a touch
    # so the ball stays worked against the dribbler instead of rolling off.
    DRIBBLE_NUDGE_AMP: float = 0.25    # lateral velocity amplitude (m/s)
    DRIBBLE_NUDGE_FREQ: float = 0.35   # rad/tick (≈0.18 s period @100 Hz)
    DRIBBLE_NUDGE_RANGE: float = 0.20  # only nudge while the ball is this close

    def __init__(
        self,
        angular_kp: float = ANGULAR_KP,
        angular_kd: float = ANGULAR_KD,
        angular_limit: float = ANGULAR_LIMIT,
    ) -> None:
        self._angular_kp = angular_kp
        self._angular_kd = angular_kd
        self._angular_limit = angular_limit
        self._movements: dict[int, _RobotMovement] = {}

    def _get_movement(self, robot_id: int) -> _RobotMovement:
        movement = self._movements.get(robot_id)
        if movement is None:
            movement = _RobotMovement(
                self._angular_kp, self._angular_kd, self._angular_limit
            )
            self._movements[robot_id] = movement
        return movement

    def resolve_command(
        self,
        intent: Intent,
        robot_id: int,
        snapshot: Snapshot,
        is_yellow: bool,
    ) -> RobotCommand | None:
        """Intent → RobotCommand with PD heading control. None if robot absent."""
        try:
            robot = _get_robot(snapshot, robot_id)
        except ValueError:
            return None

        # IntentReceive = hold still and ready; no linear or angular motion.
        if isinstance(intent, IntentReceive):
            return RobotCommand(
                robot_id=robot_id, vx=0.0, vy=0.0, w=0.0,
                kick=0, dribble=0, isYellow=bool(is_yellow),
            )

        target = intent_to_motion_target(intent, robot_id, snapshot)
        if target is None:
            return None

        current_o = robot.orientation
        movement = self._get_movement(robot_id)
        # PD heading control. Error wrapped to [-pi, pi]; a None target
        # orientation resolves (via the skill layer) to the current heading, so
        # the error is 0 and w is 0.
        err = (target.target_orientation - current_o + math.pi) % (2 * math.pi) - math.pi
        w = float(movement.angular_pd.update(err))

        # Skills produce target_velocity in WORLD frame; grSim wants body frame.
        vx_world, vy_world = target.target_velocity
        cos_o = math.cos(current_o)
        sin_o = math.sin(current_o)
        vt = vx_world * cos_o + vy_world * sin_o    # forward along heading
        vn = -vx_world * sin_o + vy_world * cos_o   # left perpendicular

        # Left↔right dribble nudge: while we're carrying the ball, add a small
        # oscillating lateral velocity so the ball stays worked on the dribbler.
        if isinstance(intent, IntentDribble):
            bx, by = snapshot.ball_position
            if math.hypot(robot.position[0] - bx, robot.position[1] - by) <= self.DRIBBLE_NUDGE_RANGE:
                movement.dribble_phase += 1
                vn += self.DRIBBLE_NUDGE_AMP * math.sin(
                    movement.dribble_phase * self.DRIBBLE_NUDGE_FREQ
                )
            else:
                movement.dribble_phase = 0

        wants_kick = isinstance(intent, (IntentKick, IntentPass))
        kick = (
            1
            if wants_kick and _kick_pose_ready(snapshot, robot_id, intent.target_pos)
            else 0
        )
        dribble = 1 if isinstance(intent, IntentDribble) else 0

        return RobotCommand(
            robot_id=robot_id,
            vx=float(vt),
            vy=float(vn),
            w=w,
            kick=kick,
            dribble=dribble,
            isYellow=bool(is_yellow),
        )


def dispatch_coordinator_output(
    coordinator,
    robot_ids: Iterable[int],
    snapshot: Snapshot,
    is_yellow: bool,
    dispatcher_q,
    run_time: float = 1.0,
    dribble_tracker: DribbleLimitTracker | None = None,
    executor: "MotionExecutor | None" = None,
) -> int:
    """Walk each robot's blackboard after a ``Coordinator.tick`` and emit a
    ``RobotCommand`` per non-empty intent.

    Reading the intent off the per-robot blackboard is more robust than
    aligning the ``Coordinator.tick`` list with ``robot_ids``: skipped robots
    (absent from snapshot, or which produced no intent) leave gaps in the
    returned list that are easy to misalign.
    """
    sent = 0
    for rid in robot_ids:
        bb = coordinator.blackboards.get(rid)
        if bb is None or bb.current_intent is None:
            if dribble_tracker is not None:
                dribble_tracker.should_enable_dribbler(rid, False, ball_pos=snapshot.ball_position)
            continue
        intent = bb.current_intent
        wants_dribble = isinstance(intent, IntentDribble)
        dribble_allowed = True
        if dribble_tracker is not None:
            dribble_allowed = dribble_tracker.should_enable_dribbler(
                rid,
                wants_dribble,
                ball_pos=snapshot.ball_position,
            )
        if wants_dribble and not dribble_allowed:
            # SSL §8.4.2: 1-metre limit reached — force a kick toward the dribble
            # target instead of continuing to carry.
            intent = IntentKick(target_pos=intent.target_pos)
        # Unified motion: route through the PD executor when provided, else the
        # legacy memoryless proportional path (kept for backward compatibility).
        if executor is not None:
            cmd = executor.resolve_command(intent, rid, snapshot, is_yellow)
        else:
            cmd = intent_to_robot_command(intent, rid, snapshot, is_yellow)
        if cmd is None:
            continue
        if not dispatcher_q.full():
            dispatcher_q.put([cmd, run_time])
            sent += 1
    return sent
