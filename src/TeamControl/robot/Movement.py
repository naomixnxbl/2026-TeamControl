# HISTORICAL MODULE — used by the behaviour tree, voronoi planner, and sandbox.
# Active game behaviour (striker, goalie, navigator, team) uses ball_nav.move_toward() instead.
#
# Known issues in this file:
#   - PDController class below is a standalone copy, kept as-is. Do not add new callers.
#   - calculateBallVelocity() speed levels (0.02–0.10) are 10x too low; actual m/s values
#     should be in the 0.2–1.0 range.  Do not rely on it for real robot speeds.
#   - For new movement code, use ball_nav.move_toward() (see docs/motion-strategy.md).
#
# Un-frozen for one specific purpose: RobotMovement.velocity_to_target() now
# calls the shared motion rules in ball_nav.py (FieldGeometryCache,
# clamp_for_role, apply_boundary_braking) via an opt-in stay_in_field
# parameter, same as RobotMotionController and ball_nav.move_toward -- so
# every motion controller in the repo follows the same rule set. The
# embedded PDController class itself is untouched/still "do not add new
# callers" -- this is only about adopting the shared geometric rules.
import math
from dataclasses import dataclass
from typing import Tuple, Optional

from TeamControl.world.transform_cords import world2robot
from TeamControl.robot import constants as C
from TeamControl.robot.ball_nav import (
    FieldGeometryCache,
    apply_boundary_braking,
    clamp_for_role,
    regulate_speed_to_target,
)
from TeamControl.robot.pd_controller import PDController
from TeamControl.robot.arrival import is_close, is_facing_direction


@dataclass
class Intent:
    type: str                            # what the BT wants, e.g. "move_to"
    target: Tuple[float, float, float]   # (x, y, θ) in world frame (mm, mm, rad)


@dataclass
class RobotState:
    x: float      # world-frame position (mm)
    y: float
    theta: float  # heading (rad)



def _wrap_angle(a: float) -> float:
    """
    Wraps any angle (radians) into (-π, π] — always the shortest rotation.
    - Without this, a 3.5 rad error rotates the long way instead of -2.78 rad the short way.
    - Formula: (a + π) % (2π) − π, then bump −π → +π.
    """
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


class RobotMovement:
    """
    Per-robot movement controller — one linear PD and one angular PD.
    - Keep ONE instance per robot; PD needs state across ticks for the D term.
    - Gains default to constants.py; pass overrides for per-robot tuning.
    """

    def __init__(self,
                 turn_kp: Optional[float] = None,
                 turn_kd: Optional[float] = None,
                 linear_kp: Optional[float] = None,
                 linear_kd: Optional[float] = None):
        self.angular_pd = PDController(
            kp=C.TURN_KP if turn_kp is None else turn_kp,
            kd=C.TURN_KD if turn_kd is None else turn_kd,
            out_limit=C.MAX_W,
        )
        self.linear_pd = PDController(
            kp=C.LINEAR_KP if linear_kp is None else linear_kp,
            kd=C.LINEAR_KD if linear_kd is None else linear_kd,
            out_limit=C.MAX_SPEED,
        )
        # Shared motion rules (ball_nav.py) -- field-size cache and role,
        # same as RobotMotionController. Opt-in via stay_in_field below.
        self._field_cache = FieldGeometryCache()
        self.is_goalie = False

    def reset(self) -> None:
        """Clear PD history. Call when target changes abruptly or robot stops."""
        self.angular_pd.reset()
        self.linear_pd.reset()

    def step(self, intent: Intent, state: RobotState,
             threshold_xy: float = 50.0,
             threshold_theta: float = 0.05) -> Optional[Tuple[float, float, float]]:
        """
        Main entry point called every tick by the Behaviour Tree.

        Returns (vx, vy, w) to drive the robot, or None when it has arrived
        (close enough in position AND facing the right way).
        """
        target_xy = (intent.target[0], intent.target[1])
        target_theta = intent.target[2]

        # Arrived? Both conditions must pass before we stop commanding.
        if is_close(target_xy, (state.x, state.y), threshold_xy) and \
           is_facing_direction(target_theta, state.theta, threshold_theta):
            self.reset()  # clear history so next move starts clean
            return None

        # Linear velocity: convert target to robot's local frame first,
        # because motor commands are in robot-frame (forward/sideways).
        local_target = world2robot((state.x, state.y, state.theta), target_xy)
        vx, vy = self.go_to_target(local_target)


        # Angular velocity: shortest rotation to target heading.
        angle_err = _wrap_angle(target_theta - state.theta)
        if abs(angle_err) < C.ANGLE_EPSILON:
            self.angular_pd.reset()
            w = 0.0
        else:
            w = self.angular_pd.update(angle_err)

        return vx, vy, w

    def set_role(self, is_goalie: bool) -> None:
        """Set whether this robot is the goalie (used by stay_in_field's
        penalty-box clamp: goalie stays in, non-goalie stays out)."""
        self.is_goalie = bool(is_goalie)

    def step(self, intent: Intent, state: RobotState,
             threshold_xy: float = 50.0,
             threshold_theta: float = 0.05) -> Optional[Tuple[float, float, float]]:
        """
        Main entry point called every tick by the Behaviour Tree.

        Returns (vx, vy, w) to drive the robot, or None when it has arrived
        (close enough in position AND facing the right way).
        """
        target_xy = (intent.target[0], intent.target[1])
        target_theta = intent.target[2]

        # Arrived? Both conditions must pass before we stop commanding.
        if is_close(target_xy, (state.x, state.y), threshold_xy) and \
           is_facing_direction(target_theta, state.theta, threshold_theta):
            self.reset()  # clear history so next move starts clean
            return None

        # Linear velocity: convert target to robot's local frame first,
        # because motor commands are in robot-frame (forward/sideways).
        local_target = world2robot((state.x, state.y, state.theta), target_xy)
        vx, vy = self.go_to_target(local_target)


        # Angular velocity: shortest rotation to target heading.
        angle_err = _wrap_angle(target_theta - state.theta)
        if abs(angle_err) < C.ANGLE_EPSILON:
            self.angular_pd.reset()
            w = 0.0
        else:
            w = self.angular_pd.update(angle_err)

        return vx, vy, w

    def velocity_to_target(self,
                           robot_pos: Tuple[float, float, float],
                           target: Tuple[float, float],
                           turning_target: Optional[Tuple[float, float]] = None,
                           speed: Optional[float] = None,
                           stop_threshold: float = 150.0,
                           stay_in_field: bool = False,
                           ) -> Tuple[float, float, float]:
        """
        Top-level driver: given where the robot is in the world, compute the
        (vx, vy, w) command that drives it toward `target` while facing
        `turning_target`.

        Math / why:
          - Our robot is holonomic (can translate in any direction AND spin
            at the same time), so we solve the linear and angular problems
            INDEPENDENTLY and hand back both answers.
          - world2robot() converts a world-frame point into the robot's own
            local frame. We do this because the robot's motors only know
            "forward/sideways/spin" — they have no idea what "world +X"
            means. Once everything is in the robot frame, +x is "in front
            of me", +y is "to my left", so vx/vy/w map straight to motor
            commands.

        stay_in_field=True applies the same shared rules as
        RobotMotionController/ball_nav.move_toward: field-size-change
        awareness, goalie/non-goalie penalty-box clamping, and field-
        boundary dynamic braking (including the goal-post no-go zone).
        """
        if robot_pos is None or target is None:
            raise ValueError("Robot pos or Target is None")

        # Rule: aware of any change in field geometry.
        self._field_cache.refresh()

        if stay_in_field:
            target = clamp_for_role(target, self.is_goalie)

        # Where is the target relative to me, right now? (mm in robot frame)
        trans_target = world2robot(robot_pos, target)
        vx, vy = self.go_to_target(trans_target, speed=speed, stop_threshold=stop_threshold)

        if turning_target is None:
            w = 0.0
        else:
            trans_turn = world2robot(robot_pos, turning_target)
            w = self.turn_to_target(trans_turn)

        if stay_in_field:
            vx, vy = apply_boundary_braking(robot_pos, vx, vy)

        return vx, vy, w

    def turn_to_target(self,
                       target: Optional[Tuple[float, float]] = None,
                       epsilon: Optional[float] = None
                       ) -> float:
        """
        Return ω (rad/s) to rotate the kicker toward `target` (robot frame, mm).
        - Kicker faces +x, so the angle error is atan2(ty, tx).
        - Wrapped to (-π, π] to always take the shortest rotation.
        - Below epsilon: stop and reset PD to avoid jitter.
        """
        if target is None:
            self.angular_pd.reset()
            return 0.0

        eps = C.ANGLE_EPSILON if epsilon is None else epsilon
        angle = _wrap_angle(math.atan2(target[1], target[0]))

        if abs(angle) < eps:
            self.angular_pd.reset()
            return 0.0

        return self.angular_pd.update(angle)

    def go_to_target(self,
                     target_pos: Optional[Tuple[float, float]],
                     speed: Optional[float] = None,
                     stop_threshold: Optional[float] = None
                     ) -> Tuple[float, float]:
        """
        Return (vx, vy) in m/s toward `target_pos` (robot frame, mm).
        - Error = target position in robot frame (robot is always at its own origin).
        - kp ~0.002: 500 mm error → 1.0 m/s, saturates at MAX_SPEED beyond that.
        - Speed capped by zone: 0 in kicker zone, 20% in dribble zone, full elsewhere.
        - stop_threshold: stop early to avoid chasing sub-threshold noise.
        """
        if target_pos is None:
            return 0.0, 0.0

        max_speed = C.MAX_SPEED if speed is None else speed
        stop = 0.0 if stop_threshold is None else stop_threshold

        dist = math.hypot(target_pos[0], target_pos[1])
        if dist <= stop:
            self.linear_pd.reset()
            return 0.0, 0.0

        zone_cap = self.threshold_zone(dist, max_speed)
        if zone_cap <= 0.0:
            self.linear_pd.reset()
            return 0.0, 0.0

        vx, vy = self.linear_pd.update((target_pos[0], target_pos[1]))

        # Scale down to zone cap while preserving direction.
        mag = math.hypot(vx, vy)
        if mag > zone_cap and mag > 0:
            s = zone_cap / mag
            vx *= s
            vy *= s

        # Rule: never overshoot -- stateless sqrt(2*a*d) cap (see
        # ball_nav.regulate_speed_to_target), shared with every other
        # motion controller. Wins over the zone cap above if tighter.
        mag = math.hypot(vx, vy)
        regulated = regulate_speed_to_target(dist, mag, C.LINEAR_AMAX)
        if mag > 0.0 and regulated < mag:
            s = regulated / mag
            vx *= s
            vy *= s
        return vx, vy

    # Backward-compat alias for legacy callers.
    go_To_Target = go_to_target

    # ─── pure geometry helpers (no controller state) ──────────────────

    @staticmethod
    def threshold_zone(distance: float, max_speed: float) -> float:
        """
        Speed cap by distance (mm): 0 in kicker zone (<70), 20% in dribble zone (<400), full elsewhere.
        - Safety net on top of PD — prevents slamming the ball even if gains are mistuned.
        """
        if distance < C.KICKER_ZONE:
            return 0.0
        if distance < C.DRIBBLE_ZONE:
            return max_speed * C.DRIBBLE_SPEED_FRAC
        return max_speed

    @staticmethod
    def shooting_pos(ball_pos, shootingTarget, robot_offset: float = 200.0):
        """Standoff point `robot_offset` mm behind the ball on the ball→shootingTarget line."""
        dx = float(shootingTarget[0]) - float(ball_pos[0])
        dy = float(shootingTarget[1]) - float(ball_pos[1])
        norm = math.hypot(dx, dy)
        if norm == 0:
            return (float(ball_pos[0]), float(ball_pos[1]))
        dx /= norm
        dy /= norm
        return (float(ball_pos[0]) - robot_offset * dx,
                float(ball_pos[1]) - robot_offset * dy)



# Utility classes

_MOVEMENT_BY_ROBOT: dict = {}


def get_movement(robot_id, is_yellow: bool = True) -> RobotMovement:
    """
    Return the persistent RobotMovement for this robot, creating it on first call.
    - Use from free functions that can't hold an instance across ticks.
    - Long-lived classes should hold self.movement = RobotMovement() directly.
    """
    key = (bool(is_yellow), robot_id)
    rm = _MOVEMENT_BY_ROBOT.get(key)
    if rm is None:
        rm = RobotMovement()
        _MOVEMENT_BY_ROBOT[key] = rm
    return rm


class Follow_path:
    def __init__(self):
        self.path = None

    def update_path(self, path: list):
        """
        Adds a path to follow
        Prams --> path as a list[x position , y position]
        """
        self.path = path

    def get_point(self, robot_pos: tuple[float, float]):
        '''
        Gets the fist point of a given path, will remove the first point once reached
        Prams --> the robot position [x position , y position]
        '''
        if self.path == None:
            print("Please update the path before you call this function")
        else:
            diff = math.hypot(self.path[0][0] - robot_pos[0],
                              self.path[0][1] - robot_pos[1])

            if len(self.path) == 1:
                return self.path
            elif diff < 0.5:
                del self.path[0]
                return self.path[0]
            else:
                return self.path[0]


class calculateBallVelocity:
    """
    step() returns a 2-tuple:
      (distance, speed)
    where:
      - distance : float            # world-frame distance to the ball
      - speed    : Optional[float]  # chosen speed (m/s), or None if unreachable
    """

    def __init__(self, time_threshold: float = 1.5):
        self.time_threshold = time_threshold
        self.speed_levels = [0.02, 0.04, 0.06, 0.08, 0.10]

    def _pick_speed(self, distance: float) -> Optional[float]:
        best = None
        for v in self.speed_levels:
            if distance / v <= self.time_threshold:
                if best is None or v < best:
                    best = v
        return best

    def step(
        self,
        robot_pose: Tuple[float, float, float],
        ball_pos:   Tuple[float, float]
    ) -> Tuple[float, Optional[float]]:
        dx = ball_pos[0] - robot_pose[0]
        dy = ball_pos[1] - robot_pose[1]
        distance = math.hypot(dx, dy)
        speed = self._pick_speed(distance)
        return distance, speed
