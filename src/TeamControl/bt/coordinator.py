"""Coordinator — role assignment, game-state dispatch, and per-robot tree dispatch.

Pipeline:
    Snapshot → Coordinator.tick() → list[Intent]

Game-state handling
-------------------
Before running the normal role trees the Coordinator checks the current
GamePhase from the Snapshot's RefereeState and may override behaviour:

    HALTED / HALF_TIME  → no intents produced (robots coast to a stop)
    STOPPED             → every robot holds position; any robot within
                          STOP_BALL_CLEARANCE of the ball is nudged away
    PREPARE_KICKOFF     → robots move to pre-kickoff positions
    KICKOFF             → attacker goes to ball at centre, others to support spots
    FREE_KICK           → attacker goes to ball, others hold positions
    CORNER_KICK         → free kick near opp goal line: kicker crosses, supporters
                          pack the box (a free kick refined by ball position; §5.3)
    GOAL_KICK           → free kick near own goal line: kicker clears upfield,
                          supporters push to midfield outlets
    BALL_PLACEMENT      → all robots keep STOP_BALL_CLEARANCE from ball
    PENALTY_SHOOT       → attacker to penalty spot, others behind ball line
    PENALTY_DEFEND      → goalie tracks ball on goal line, others hold
    RUNNING             → normal role-tree dispatch (existing behaviour)
"""
from __future__ import annotations

import math
import time
from typing import Any

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import Intent, IntentMove
from TeamControl.bt.contracts.snapshot import GamePhase, Snapshot
from TeamControl.bt.tactics.heuristic_role_swap import (
    RoleHeuristicWeights,
    assign_roles_heuristically,
    load_role_heuristic_weights,
)
from TeamControl.bt.tactics.rule_following import (
    MovementSafetyConfig,
    apply_rule_following,
    has_rule_following_enabled,
)

# ---------------------------------------------------------------------------
# Role assignment, fixed by robot ID unless heuristic role swapping is enabled.
# ---------------------------------------------------------------------------
# index 0 -> GOALIE, 1 -> ATTACKER, 2-5 -> SUPPORTER
ROLE_ASSIGNMENT: dict[int, RoleType] = {
    0: RoleType.GOALIE,
    1: RoleType.ATTACKER, # this should be the attacking tree
    2: RoleType.SUPPORTER,
    3: RoleType.SUPPORTER,
    4: RoleType.SUPPORTER,
    5: RoleType.SUPPORTER,
}

# ---------------------------------------------------------------------------
# Field constants (metres — matches existing BT skill constants)
# ---------------------------------------------------------------------------
# Legal SSL threshold is 0.5m. We use 0.55m as a safety buffer so robots
# are never right on the limit. 0.5m is what the rules enforce, not 0.55m.
STOP_BALL_CLEARANCE: float = 0.55
LEGAL_BALL_CLEARANCE: float = 0.50   # actual SSL rule threshold

OWN_GOAL: tuple[float, float] = (-4.5, 0.0)
OPP_GOAL: tuple[float, float] = (4.5, 0.0)
OWN_GOAL_LINE_X: float = -4.5

# Div B field (9m × 6m): goal line at x=±4.5m.
# §2.1.3: penalty mark is 6m from opponent goal center → 4.5 - 6 = -1.5m from center.
# For us_positive=False (opp goal at +4.5): penalty mark at (-1.5, 0).
# Mirrored to (+1.5, 0) for us_positive=True.
PENALTY_SPOT: tuple[float, float] = (-1.5, 0.0)

# Home positions robots return to during STOPPED — spread across own half (negative-x for us_positive=False).
# Mirrored in Coordinator.__init__ when us_positive=True.
STOPPED_HOME_POSITIONS: dict[int, tuple[float, float]] = {
    0: (-4.2,  0.0),   # goalie near own goal
    1: (-2.5, -1.0),
    2: (-2.5,  1.0),
    3: (-1.5,  0.0),
    4: (-1.5, -1.5),
    5: (-1.5,  1.5),
}

# Defensive positions for when the OPPONENT has the kickoff.
# All in own half (-x for us_positive=False), all outside center circle (r=0.5m).
# Shape: goalie on line, two defenders wide, two midfielders covering channels, one at centre-half.
OPP_KICKOFF_POSITIONS: dict[int, tuple[float, float]] = {
    0: (-4.2,  0.0),   # goalie on goal line
    1: (-3.0, -1.5),   # defender left
    2: (-3.0,  1.5),   # defender right
    3: (-2.0,  0.0),   # centre back
    4: (-1.5, -1.2),   # mid left
    5: (-1.5,  1.2),   # mid right
}

# FREE_KICK support offsets — relative to ball position, at legal distance (>0.5m from ball).
# Kicker is assigned dynamically; these are fallback slots for non-kicker, non-goalie robots.
FREE_KICK_SUPPORT_OFFSETS: list[tuple[float, float]] = [
    (0.0, -1.2),   # slot 0 — left wing
    (0.0,  1.2),   # slot 1 — right wing
    (-0.8,  0.0),  # slot 2 — back centre
    (-0.8, -0.8),  # slot 3 — back left
    (-0.8,  0.8),  # slot 4 — back right
]

# ---------------------------------------------------------------------------
# Corner / goal kick classification.
# SSL emits only DIRECT_/INDIRECT_FREE — there is no "corner" or "goal kick"
# command (§5.3). A free kick whose ball sits within FREE_KICK_GOAL_LINE_BAND
# of a goal line is treated as a corner (near the *opponent* goal, attacking)
# or a goal kick (near *our* goal, clearing). The kind is locked once per
# free-kick episode so it can't flap as the ball moves after the kick.
# ---------------------------------------------------------------------------
FREE_KICK_GOAL_LINE_BAND: float = 1.5  # m from a goal line to count as corner/goal kick

# Attacking box slots for OUR corner kick — (depth in front of opp goal, y).
# depth is measured back from the opponent goal line toward midfield; kept ≥1.3m
# so robots stay clear of the opponent defense area during the set piece (§8.4).
CORNER_BOX_SLOTS: list[tuple[float, float]] = [
    (1.4, -0.6),
    (1.4,  0.6),
    (1.6,  0.0),
    (2.0, -1.4),
    (2.0,  1.4),
]
# Where the corner taker aims — a dangerous central point in front of the goal.
CORNER_KICK_TARGET_DEPTH: float = 1.4  # m in front of opp goal line

# Outlet slots for OUR goal kick (defensive clear) — (depth up from own goal, y).
GOAL_KICK_OUTLET_SLOTS: list[tuple[float, float]] = [
    (3.0, -2.0),
    (3.0,  2.0),
    (2.0, -1.0),
    (2.0,  1.0),
    (4.0,  0.0),
]
# Where the goal-kick taker clears to — straight upfield from the ball, toward
# the opponent half. Aiming along the ball's current y keeps the clearance from
# ever crossing our own goal mouth.
GOAL_KICK_CLEAR_DEPTH: float = 3.0   # m upfield from ball, toward opponent half

# Defensive screen slots for an ENEMY corner kick — (depth in front of own goal, y).
OPP_CORNER_DEFEND_SLOTS: list[tuple[float, float]] = [
    (0.6,  0.0),   # central, just ahead of keeper
    (0.6, -0.7),
    (0.6,  0.7),
    (1.0, -1.3),
    (1.0,  1.3),
]

# PREPARE_KICKOFF positions — all robots in own half (negative-x when us_positive=True).
# Rule §5.3.2: one attacker is allowed ANYWHERE inside the centre circle (radius=0.5m).
# We place attacker at (0, 0) — centre of the circle — to be as close to ball as allowed.
# Ball is at centre, placed by human referee before kickoff command. Robots must NOT touch it.
KICKOFF_POSITIONS: dict[int, tuple[float, float]] = {
    0: (-4.0, 0.0),    # goalie — in front of own goal
    1: (-2.0, -1.5),   # defender left
    2: (-2.0,  1.5),   # defender right
    3: (-0.3,  0.0),   # kicker — inside center circle, 30 cm behind ball on attack axis
    4: (-1.0,  1.0),   # supporter right
    5: (-1.0, -1.0),   # supporter left
}

# PENALTY_SHOOT: attacker at penalty spot, ALL others ≥ 1m behind ball (x ≤ 2.5).
# Our goalie is NOT the defending keeper — opponent's keeper defends their goal.
# All our non-attacker robots just need to be 1m behind the ball.
PENALTY_SHOOT_POSITIONS: dict[int, tuple[float, float]] = {
    0: (2.0,  0.5),    # our goalie — behind ball
    1: (2.0, -1.5),
    2: (2.0,  1.5),
    3: (2.0, -0.5),
    4: (2.0,  0.5),
    5: PENALTY_SPOT,   # attacker shoots
}

# Positions for penalty defend: goalie on goal line, others in own half
PENALTY_DEFEND_POSITIONS: dict[int, tuple[float, float]] = {
    0: (-4.5, 0.0),    # goalie on goal line (tracks ball y dynamically)
    1: (-2.0, -1.0),
    2: (-2.0,  1.0),
    3: (-1.5, -0.5),
    4: (-1.5,  0.5),
    5: (-1.0,  0.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_robot(snapshot: Snapshot, robot_id: int):
    for r in snapshot.own_robots:
        if r.robot_id == robot_id:
            return r
    return None


def _nudge_away_from_ball(
    robot_pos: tuple[float, float],
    ball_pos: tuple[float, float],
    clearance: float,
) -> tuple[float, float]:
    """Return a target position *clearance* metres from the ball, away from robot."""
    dx = robot_pos[0] - ball_pos[0]
    dy = robot_pos[1] - ball_pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (robot_pos[0] - clearance, robot_pos[1])
    scale = clearance / dist
    return (ball_pos[0] + dx * scale, ball_pos[1] + dy * scale)


def _dist_to_segment(
    point: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> float:
    """Return the shortest distance from *point* to line segment A→B."""
    ax, ay = seg_a
    bx, by = seg_b
    px, py = point
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    # Project point onto the line, clamped to [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def _nudge_away_from_segment(
    robot_pos: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
    clearance: float,
) -> tuple[float, float]:
    """Push *robot_pos* perpendicularly away from segment A→B to *clearance* distance."""
    ax, ay = seg_a
    bx, by = seg_b
    px, py = robot_pos
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        return _nudge_away_from_ball(robot_pos, seg_a, clearance)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    # Direction from closest point on segment to robot
    dx = px - closest_x
    dy = py - closest_y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        # Robot is exactly on the line — nudge perpendicular to the segment
        perp_x, perp_y = -aby, abx
        perp_len = math.hypot(perp_x, perp_y)
        return (closest_x + perp_x / perp_len * clearance,
                closest_y + perp_y / perp_len * clearance)
    return (closest_x + dx / dist * clearance, closest_y + dy / dist * clearance)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _mirror(pos: tuple[float, float]) -> tuple[float, float]:
    """Flip a position to the other half of the field (negate x)."""
    return (-pos[0], pos[1])


def _mirror_positions(
    positions: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    return {k: _mirror(v) for k, v in positions.items()}


class Coordinator:
    """Assigns roles, checks game phase, and dispatches role trees each tick.

    Parameters
    ----------
    trees:
        Mapping of RoleType → tree object.
    us_positive:
        True if WE ARE on the positive-x half (our own goal is at +x, we attack toward -x).
        False if we are on the negative-x half (our own goal is at -x, we attack toward +x).
    """

    def __init__(
        self,
        trees: dict[RoleType, Any],
        us_positive: bool = True,
        role_assignment: dict[int, RoleType] | None = None,
        heuristic_role_swap: bool = False,
        heuristic_weights: RoleHeuristicWeights | None = None,
        heuristic_weights_file: str = "bt_tuning.yaml",
        movement_safety: MovementSafetyConfig | dict[str, bool | float] | None = None,
    ) -> None:
        self.trees = trees
        self.role_assignment = dict(role_assignment or ROLE_ASSIGNMENT)
        self.heuristic_role_swap = bool(heuristic_role_swap)
        self.heuristic_weights = (
            heuristic_weights
            if heuristic_weights is not None
            else load_role_heuristic_weights(heuristic_weights_file)
        )
        self.movement_safety = (
            movement_safety
            if isinstance(movement_safety, MovementSafetyConfig)
            else MovementSafetyConfig.from_mapping(movement_safety)
        )
        self.blackboards: dict[int, RobotBlackboard] = {}
        self._role_swap_last_changed_at: dict[int, float] = {}
        self._free_kick_kicker_id: int | None = None
        self._free_kick_support_slots: dict[int, int] = {}
        self._free_kick_kicker_ready: bool = False
        self._free_kick_has_kicked: bool = False
        self._free_kick_ball_ref: tuple[float, float] | None = None
        self._free_kick_kicker_hold: tuple[float, float] | None = None
        self._free_kick_double_touch_cleared: bool = False
        self._kickoff_kicker_id: int | None = None
        self._kickoff_kicker_ready: bool = False
        self._kickoff_has_kicked: bool = False
        self._kickoff_ball_ref: tuple[float, float] | None = None
        self._kickoff_kicker_hold: tuple[float, float] | None = None
        self._kickoff_double_touch_cleared: bool = False
        self._kickoff_needs_slot: set[int] = set()  # robots that must reach their slot
        self._opp_kickoff_carry: bool = False
        self._penalty_shoot_carry: bool = False
        self._penalty_shoot_done: bool = False
        self._penalty_shoot_ball_ref: tuple[float, float] | None = None
        self._penalty_defend_carry: bool = False
        self._penalty_defend_ball_ref: tuple[float, float] | None = None
        self._enemy_free_kick_ball_ref: tuple[float, float] | None = None
        self._last_phase: GamePhase | None = None
        self._pre_halt_phase: GamePhase | None = None  # phase before HALTED
        if us_positive:
            # We are on +x half → own goal at +x, opponent goal at -x, attack toward -x.
            self._kickoff_pos = _mirror_positions(KICKOFF_POSITIONS)
            self._opp_kickoff_pos = _mirror_positions(OPP_KICKOFF_POSITIONS)
            self._penalty_shoot_pos = _mirror_positions(PENALTY_SHOOT_POSITIONS)
            self._penalty_defend_pos = _mirror_positions(PENALTY_DEFEND_POSITIONS)
            self._stopped_home = _mirror_positions(STOPPED_HOME_POSITIONS)
            self._opp_goal: tuple[float, float] = OWN_GOAL   # (-4.5, 0)
            self._own_goal_line_x: float = -OWN_GOAL_LINE_X  # +4.5
            self._attack_sign: float = -1.0
        else:
            # We are on -x half → own goal at -x, opponent goal at +x, attack toward +x.
            self._kickoff_pos = dict(KICKOFF_POSITIONS)
            self._penalty_shoot_pos = PENALTY_SHOOT_POSITIONS
            self._penalty_defend_pos = PENALTY_DEFEND_POSITIONS
            self._opp_goal = OPP_GOAL                         # (4.5, 0)
            self._own_goal_line_x = OWN_GOAL_LINE_X           # -4.5
            self._attack_sign: float = 1.0
            self._opp_kickoff_pos = dict(OPP_KICKOFF_POSITIONS)
            self._stopped_home = dict(STOPPED_HOME_POSITIONS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """Tick all robots present in the snapshot.

        Checks the current GamePhase first. For halted/set-piece states a
        dedicated handler writes intents directly onto each robot's blackboard
        instead of running the normal role trees.
        """
        self._ensure_blackboards(snapshot, robot_ids)

        phase = snapshot.referee_state.game_phase

        if phase != self._last_phase:
            self._free_kick_kicker_id = None
            self._free_kick_support_slots = {}
            self._free_kick_kicker_ready = False
            self._free_kick_has_kicked = False
            self._free_kick_ball_ref = None
            self._free_kick_kicker_hold = None
            self._free_kick_double_touch_cleared = False
            self._enemy_free_kick_ball_ref = None
            # Preserve kickoff state when GC skips straight from KICKOFF/PREPARE_KICKOFF
            # to RUNNING before the kicker has fired — we finish the kick in RUNNING.
            kickoff_carry = (
                phase == GamePhase.RUNNING
                and self._kickoff_kicker_id is not None
                and not self._kickoff_kicker_ready
            )
            if not kickoff_carry:
                self._kickoff_kicker_id = None
                self._kickoff_kicker_ready = False
            self._kickoff_has_kicked = False
            self._kickoff_ball_ref = None
            self._kickoff_kicker_hold = None
            self._kickoff_double_touch_cleared = False
            self._kickoff_needs_slot = set()
            # Track pre-HALTED phase so carries can look through HALTED.
            if self._last_phase not in (GamePhase.HALTED, GamePhase.HALF_TIME):
                self._pre_halt_phase = self._last_phase
            # Carry opp kickoff positioning into RUNNING — works through HALTED.
            prior = self._pre_halt_phase
            self._opp_kickoff_carry = (
                prior == GamePhase.ENEMY_KICKOFF
                and phase == GamePhase.RUNNING
            )
            # Carry penalty shoot into RUNNING until kicker fires.
            if phase in (GamePhase.PREPARE_PENALTY, GamePhase.PENALTY_SHOOT):
                self._penalty_shoot_carry = True
                self._penalty_shoot_done = False
                self._penalty_shoot_ball_ref = None  # fresh ref each penalty attempt
            elif phase == GamePhase.RUNNING and self._penalty_shoot_carry and not self._penalty_shoot_done:
                pass  # keep carry active
            else:
                self._penalty_shoot_carry = False
                self._penalty_shoot_done = False

            # Carry penalty defend into RUNNING until ball moves (kick detected).
            if phase in (GamePhase.PREPARE_PENALTY_OPP, GamePhase.PENALTY_DEFEND):
                self._penalty_defend_carry = True
                self._penalty_defend_ball_ref = None  # set on first RUNNING tick
            elif phase == GamePhase.RUNNING and self._penalty_defend_carry:
                pass  # keep carry active, cleared below when ball moves
            else:
                self._penalty_defend_carry = False
                self._penalty_defend_ball_ref = None
            self._last_phase = phase

        if phase in (GamePhase.HALTED, GamePhase.HALF_TIME):
            # Robots must not move — produce no intents so the dispatcher
            # lets existing commands time out and robots coast to zero.
            return []

        if phase == GamePhase.STOPPED:
            self._handle_stopped(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.BALL_PLACEMENT:
            self._handle_ball_placement(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.PREPARE_KICKOFF:
            # Lock in the kicker now so it carries into KICKOFF/RUNNING.
            self._lock_kickoff_kicker(snapshot, robot_ids)
            self._handle_prepare_kickoff(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.ENEMY_KICKOFF:
            self._handle_opp_kickoff(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.KICKOFF:
            self._handle_kickoff(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.FREE_KICK:
            # Lock the corner/goal/plain classification once for the episode.
            if self._free_kick_kind is None:
                self._free_kick_kind = self._classify_free_kick(snapshot, enemy=False)
            if self._free_kick_kind == GamePhase.CORNER_KICK:
                self._handle_corner_kick(snapshot, robot_ids)
            elif self._free_kick_kind == GamePhase.GOAL_KICK:
                self._handle_goal_kick(snapshot, robot_ids)
            else:
                self._handle_free_kick(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.ENEMY_FREE_KICK:
            if self._free_kick_kind is None:
                self._free_kick_kind = self._classify_free_kick(snapshot, enemy=True)
            if self._free_kick_kind == GamePhase.ENEMY_CORNER_KICK:
                self._handle_opp_corner_kick(snapshot, robot_ids)
            else:
                # ENEMY_GOAL_KICK and plain ENEMY_FREE_KICK share the defensive
                # spread — in SSL they are the same restart, only the location
                # differs, and the generic handler already keeps legal clearance.
                self._handle_opp_free_kick(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.PREPARE_PENALTY:
            self._handle_prepare_penalty(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.PREPARE_PENALTY_OPP:
            self._handle_prepare_penalty_opp(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.PENALTY_SHOOT:
            self._handle_penalty_shoot(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        if phase == GamePhase.PENALTY_DEFEND:
            self._handle_penalty_defend(snapshot, robot_ids)
            return self._finalize_intents(snapshot, robot_ids)

        # RUNNING — finish enemy kickoff positioning if carry is active.
        if self._enemy_kickoff_carry:
            result = self._handle_enemy_kickoff(snapshot, robot_ids)
            # Clear carry once all robots are within 0.2m of their slots.
            all_at_slot = True
            for rid in robot_ids:
                robot = _find_robot(snapshot, rid)
                if robot is None:
                    continue
                slot = self._opp_kickoff_pos.get(rid, robot.position)
                if math.hypot(robot.position[0] - slot[0], robot.position[1] - slot[1]) > 0.2:
                    all_at_slot = False
                    break
            if all_at_slot:
                self._opp_kickoff_carry = False
            return self._finalize_intents(snapshot, robot_ids)

        # RUNNING — finish kickoff: approach, kick, then hold for double-touch rule.
        if self._kickoff_kicker_id is not None:
            result = self._handle_kickoff(snapshot, robot_ids)
            # Only end carry once double-touch restriction is lifted (teammate touched ball
            # or ball moved enough that carry purpose is complete).
            if self._kickoff_double_touch_cleared:
                self._kickoff_kicker_id = None
                self._kickoff_kicker_ready = False
            return self._finalize_intents(snapshot, robot_ids)

        # RUNNING — carry penalty shoot until kicker fires OR ball moves.
        if self._penalty_shoot_carry and not self._penalty_shoot_done:
            bx, by = snapshot.ball_position
            if self._penalty_shoot_ball_ref is None:
                self._penalty_shoot_ball_ref = (bx, by)
            ref_x, ref_y = self._penalty_shoot_ball_ref
            if math.hypot(bx - ref_x, by - ref_y) > 0.2:
                self._penalty_shoot_carry = False
                self._penalty_shoot_done = True
                self._penalty_shoot_ball_ref = None

        # RUNNING — carry penalty defend until ball moves (kick detected).
        if self._penalty_defend_carry:
            bx, by = snapshot.ball_position
            if self._penalty_defend_ball_ref is None:
                self._penalty_defend_ball_ref = (bx, by)
            ref_x, ref_y = self._penalty_defend_ball_ref
            ball_moved = math.hypot(bx - ref_x, by - ref_y) > 0.30
            if ball_moved:
                self._penalty_defend_carry = False
                self._penalty_defend_ball_ref = None
            else:
                self._handle_penalty_defend(snapshot, robot_ids)
                return self._finalize_intents(snapshot, robot_ids)

        # RUNNING — carry penalty shoot until kicker fires.
        if self._penalty_shoot_carry and not self._penalty_shoot_done:
            self._handle_penalty_shoot(snapshot, robot_ids)
            # Check if kicker just issued a kick intent.
            from TeamControl.bt.contracts.intent import IntentKick
            kicker_bb = self.blackboards.get(5)
            if kicker_bb is not None and isinstance(kicker_bb.current_intent, IntentKick):
                self._penalty_shoot_done = True
                self._penalty_shoot_carry = False
                self._penalty_shoot_ball_ref = None  # clear so next penalty starts fresh
            return result
            return self._finalize_intents(snapshot, robot_ids)

        self._apply_heuristic_roles(snapshot, robot_ids)
        self._normal_tick(snapshot, robot_ids)
        return self._finalize_intents(snapshot, robot_ids)

    def _finalize_intents(
        self,
        snapshot: Snapshot,
        robot_ids: list[int],
    ) -> list[Intent]:
        """Apply final movement guard rails and return current intents."""
        if has_rule_following_enabled(self.movement_safety):
            self._apply_rule_following(snapshot, robot_ids)

        present_ids = {robot.robot_id for robot in snapshot.own_robots}
        intents: list[Intent] = []
        for robot_id in robot_ids:
            if robot_id not in present_ids:
                continue
            bb = self.blackboards.get(robot_id)
            if bb is not None and bb.current_intent is not None:
                intents.append(bb.current_intent)
        return intents

    def _apply_rule_following(
        self,
        snapshot: Snapshot,
        robot_ids: list[int],
    ) -> None:
        robot_by_id = {robot.robot_id: robot for robot in snapshot.own_robots}
        for robot_id in robot_ids:
            robot = robot_by_id.get(robot_id)
            if robot is None:
                continue
            bb = self.blackboards.get(robot_id)
            if bb is None or bb.current_intent is None:
                continue
            bb.current_intent = apply_rule_following(
                snapshot=snapshot,
                robot_id=robot_id,
                robot_pos=robot.position,
                role=self._role_of(robot_id),
                intent=bb.current_intent,
                config=self.movement_safety,
                own_goal_line_x=self._own_goal_line_x,
                opponent_goal=self._opp_goal,
            )

    # ------------------------------------------------------------------
    # Game-phase handlers
    # ------------------------------------------------------------------

    def _handle_stopped(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """STOPPED / BALL_PLACEMENT: every robot holds position unless it is
        within STOP_BALL_CLEARANCE of the ball, in which case it backs away."""
        intents: list[Intent] = []
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            dist = math.hypot(
                robot.position[0] - snapshot.ball_position[0],
                robot.position[1] - snapshot.ball_position[1],
            )
            if dist < STOP_BALL_CLEARANCE:
                # Too close — nudge away to the clearance boundary.
                target = _nudge_away_from_ball(
                    robot.position, snapshot.ball_position, STOP_BALL_CLEARANCE
                )
            else:
                # Already clear — hold current position.
                target = robot.position
            # max_speed=1.4 enforces the SSL < 1.5 m/s rule during STOPPED
            bb.current_intent = IntentMove(target_pos=target, target_orientation=None, max_speed=1.4)
            intents.append(bb.current_intent)
        return intents

    def _handle_fixed_positions(
        self,
        snapshot: Snapshot,
        robot_ids: list[int],
        positions: dict[int, tuple[float, float]],
    ) -> list[Intent]:
        """Move every robot to its designated position from *positions* map."""
        intents: list[Intent] = []
        for robot_id in robot_ids:
            if _find_robot(snapshot, robot_id) is None:
                continue
            bb = self.blackboards[robot_id]
            target = positions.get(robot_id, KICKOFF_POSITIONS.get(robot_id, (0.0, 0.0)))
            bb.current_intent = IntentMove(target_pos=target, target_orientation=None)
            intents.append(bb.current_intent)
        return intents

    def _handle_opp_kickoff(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """OPP_KICKOFF: opponent has the kickoff — all our robots must be in own half,
        outside the center circle. No kicker exception for us."""
        CENTER_CIRCLE_R = 0.5
        intents: list[Intent] = []
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            rx, ry = robot.position
            in_own_half = rx * self._attack_sign < 0
            outside_circle = math.hypot(rx, ry) > CENTER_CIRCLE_R
            slot = self._opp_kickoff_pos.get(robot_id, robot.position)
            dist_to_slot = math.hypot(rx - slot[0], ry - slot[1])
            if in_own_half and outside_circle and dist_to_slot < 0.15:
                target = robot.position  # already at slot — hold
            else:
                target = slot  # always move to defensive slot
            bb.current_intent = IntentMove(target_pos=target, target_orientation=None, max_speed=1.4)
            intents.append(bb.current_intent)
        return intents

    def _handle_prepare_kickoff(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """PREPARE_KICKOFF: kicker moves to center-circle prep spot; others stay in own half.

        Rule §5.3.2: all robots in own half excluding center circle, except one
        kicker who may be anywhere inside the center circle. No robot touches ball.
        Non-kicker robots hold their current position if already valid; otherwise
        they move just inside own half at their current y.
        """
        CENTER_CIRCLE_R = 0.5  # m
        kicker_id = self._kickoff_kicker_id  # already locked by caller
        kicker_prep_x = -0.3 * self._attack_sign  # 30 cm behind ball on attack axis

        intents: list[Intent] = []
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]

            if robot_id == kicker_id:
                bb.current_intent = IntentMove(
                    target_pos=(kicker_prep_x, 0.0), target_orientation=None
                )
            else:
                rx, ry = robot.position
                in_own_half = rx * self._attack_sign < 0
                dist_center = math.hypot(rx, ry)
                outside_circle = dist_center > CENTER_CIRCLE_R
                slot = self._kickoff_pos.get(robot_id)
                # Mark robot as needing to reach its slot if it starts in opponent half.
                if not (in_own_half and outside_circle):
                    self._kickoff_needs_slot.add(robot_id)
                # Clear the flag once the robot arrives at its slot (within 0.15m).
                if robot_id in self._kickoff_needs_slot and slot is not None:
                    dist_to_slot = math.hypot(rx - slot[0], ry - slot[1])
                    if dist_to_slot < 0.15:
                        self._kickoff_needs_slot.discard(robot_id)
                if robot_id in self._kickoff_needs_slot and slot is not None:
                    target = slot
                elif in_own_half and outside_circle:
                    target = robot.position  # already valid — hold
                else:
                    target = slot if slot is not None else robot.position
                bb.current_intent = IntentMove(target_pos=target, target_orientation=None)

            intents.append(bb.current_intent)
        return intents

    def _handle_kickoff(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """KICKOFF: closest non-goalie robot kicks; others hold kickoff positions.

        Same two-stage approach as FREE_KICK: robot goes to approach_target first
        (behind ball on attack axis), then drives into ball once aligned.
        """
        self._lock_kickoff_kicker(snapshot, robot_ids)
        kicker_id = self._kickoff_kicker_id
        bx, by = snapshot.ball_position

        if self._kickoff_ball_ref is None:
            self._kickoff_ball_ref = (bx, by)

        ball_in_play = (
            self._kickoff_has_kicked
            and math.hypot(bx - self._kickoff_ball_ref[0], by - self._kickoff_ball_ref[1]) > 0.5
        )

        # Clear double-touch restriction once a non-kicker own robot touches the ball.
        if ball_in_play and not self._kickoff_double_touch_cleared:
            for rid in robot_ids:
                if rid == kicker_id:
                    continue
                r = _find_robot(snapshot, rid)
                if r is not None and math.hypot(r.position[0] - bx, r.position[1] - by) < 0.15:
                    self._kickoff_double_touch_cleared = True
                    break

        from TeamControl.bt.contracts.intent import IntentKick
        intents: list[Intent] = []

        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]

            if robot_id == kicker_id:
                dist_to_ball = math.hypot(robot.position[0] - bx, robot.position[1] - by)
                approach_x = bx - 0.25 * self._attack_sign
                dist_to_approach = math.hypot(robot.position[0] - approach_x, robot.position[1] - by)
                on_correct_side = (robot.position[0] - bx) * self._attack_sign < -0.05
                if dist_to_ball < 0.15 and on_correct_side:
                    self._kickoff_kicker_ready = True

                if ball_in_play and self._kickoff_double_touch_cleared:
                    # Teammate touched ball — double-touch restriction lifted, run normal BT.
                    bb.last_intent = bb.current_intent
                    bb.current_intent = None
                    tree = self.trees[bb.current_role]
                    if hasattr(tree, "set_snapshot") and hasattr(tree, "tick"):
                        tree.set_snapshot(snapshot)
                        tree.tick(bb)
                    else:
                        if hasattr(tree, "_blackboard_ref"):
                            tree._blackboard_ref[0] = bb
                        tree.tick_once()
                    if bb.current_intent is not None:
                        intents.append(bb.current_intent)
                    continue
                elif ball_in_play:
                    # Ball in play but no teammate touched yet — hold (double-touch rule).
                    if self._kickoff_kicker_hold is None:
                        self._kickoff_kicker_hold = (robot.position[0], robot.position[1])
                    bb.current_intent = IntentMove(
                        target_pos=self._kickoff_kicker_hold, target_orientation=None
                    )
                elif self._kickoff_kicker_ready:
                    if not self._kickoff_has_kicked:
                        self._kickoff_has_kicked = True
                        self._kickoff_ball_ref = (bx, by)
                    bb.current_intent = IntentKick(target_pos=self._enemy_goal)
                elif dist_to_approach < 0.15:
                    bb.current_intent = IntentMove(target_pos=(bx, by), target_orientation=None)
                else:
                    bb.current_intent = IntentMove(target_pos=(approach_x, by), target_orientation=None)

            elif ROLE_ASSIGNMENT.get(robot_id) == RoleType.GOALIE:
                target = (self._own_goal_line_x, max(-_GOAL_HW_M, min(_GOAL_HW_M, by)))
                bb.current_intent = IntentMove(target_pos=target, target_orientation=None)
                if self._kickoff_kicker_ready:
                    bb.current_intent = IntentKick(target_pos=self._opp_goal)
                elif dist_to_approach < 0.10:
                    bb.current_intent = IntentMove(target_pos=(bx, by), target_orientation=None)
                else:
                    bb.current_intent = IntentMove(
                        target_pos=(approach_x, by), target_orientation=None
                    )
            elif self._is_goalie(robot_id):
                by = snapshot.ball_position[1]
                target = (self._own_goal_line_x, max(-1.0, min(1.0, by)))
                bb.current_intent = IntentMove(
                    target_pos=target, target_orientation=None
                )
            else:
                if ball_in_play:
                    # Ball in play — run normal BT so robots can chase the ball and
                    # naturally clear the kicker's double-touch restriction.
                    bb.last_intent = bb.current_intent
                    bb.current_intent = None
                    tree = self.trees[bb.current_role]
                    if hasattr(tree, "set_snapshot") and hasattr(tree, "tick"):
                        tree.set_snapshot(snapshot)
                        tree.tick(bb)
                    else:
                        if hasattr(tree, "_blackboard_ref"):
                            tree._blackboard_ref[0] = bb
                        tree.tick_once()
                    if bb.current_intent is not None:
                        intents.append(bb.current_intent)
                    continue
                else:
                    # Keep moving to own half if not there yet; otherwise hold.
                    CENTER_CIRCLE_R = 0.5
                    rx, ry = robot.position
                    in_own_half = rx * self._attack_sign < 0
                    outside_circle = math.hypot(rx, ry) > CENTER_CIRCLE_R
                    if in_own_half and outside_circle:
                        target = robot.position
                    else:
                        safe_x = -self._attack_sign * max(abs(rx), CENTER_CIRCLE_R + 0.1)
                        target = (safe_x, ry)
                    bb.current_intent = IntentMove(target_pos=target, target_orientation=None)

            intents.append(bb.current_intent)
        return intents

    def _handle_ball_placement(
        self, snapshot: Snapshot, robot_ids: list[int]
    ) -> list[Intent]:
        """BALL_PLACEMENT: attacker moves ball to designated target.

        Placer (attacker, robot 5):
          - If far from ball     → IntentMove to ball
          - If at ball           → IntentDribble to placement target
          - If ball at target    → IntentMove away (§5.2: placer must clear
                                   ≥0.05m after placing for our next free kick)

        All others:
          - Must stay ≥ 0.5 m from the ball AND from the line between
            ball and target (§8.4.3). Nudged away if too close to either.
          - Speed capped at 1.4 m/s.
        """
        target = snapshot.referee_state.ball_placement_pos
        if target is None:
            return self._handle_stopped(snapshot, robot_ids)

        ball = snapshot.ball_position
        # Check if ball has reached the target (within 0.15m — SSL success threshold).
        ball_at_target = math.hypot(ball[0] - target[0], ball[1] - target[1]) < 0.15
        intents: list[Intent] = []

        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]

            if robot_id == self.KICKOFF_KICKER_ID:
                dist_to_ball = math.hypot(
                    robot.position[0] - ball[0],
                    robot.position[1] - ball[1],
                )
                if ball_at_target:
                    # Ball placed — clear to ≥0.05m (use LEGAL_BALL_CLEARANCE
                    # to ensure we don't interfere with the next free kick).
                    clear_target = _nudge_away_from_ball(
                        robot.position, ball, LEGAL_BALL_CLEARANCE
                    )
                    bb.current_intent = IntentMove(
                        target_pos=clear_target, target_orientation=None
                    )
                elif dist_to_ball > 0.25:
                    bb.current_intent = IntentMove(
                        target_pos=ball, target_orientation=None
                    )
                else:
                    from TeamControl.bt.contracts.intent import IntentDribble
                    bb.current_intent = IntentDribble(target_pos=target)
            else:
                # Non-placer: keep clear of ball and the ball→target line.
                pos = robot.position
                dist_ball = math.hypot(pos[0] - ball[0], pos[1] - ball[1])
                dist_line = _dist_to_segment(pos, ball, target)

                if dist_ball < STOP_BALL_CLEARANCE:
                    move_target = _nudge_away_from_ball(pos, ball, STOP_BALL_CLEARANCE)
                elif dist_line < STOP_BALL_CLEARANCE:
                    move_target = _nudge_away_from_segment(pos, ball, target, STOP_BALL_CLEARANCE)
                else:
                    move_target = pos  # already clear — hold position

                bb.current_intent = IntentMove(
                    target_pos=move_target, target_orientation=None, max_speed=1.4
                )

            intents.append(bb.current_intent)
        return intents

    def _handle_free_kick(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """FREE_KICK: closest non-goalie robot kicks; others run BT."""
        # Lock in the kicker once for the duration of the FREE_KICK phase.
        if self._free_kick_kicker_id is None:
            best_dist = float("inf")
            for robot_id in robot_ids:
                if self._is_goalie(robot_id):
                    continue
                robot = _find_robot(snapshot, robot_id)
                if robot is None:
                    continue
                d = math.hypot(
                    robot.position[0] - snapshot.ball_position[0],
                    robot.position[1] - snapshot.ball_position[1],
                )
                if d < best_dist:
                    best_dist = d
                    self._free_kick_kicker_id = robot_id
        kicker_id = self._free_kick_kicker_id

        # Record ball position on the first tick so ball_in_play can be detected
        # even if the kicker never reaches approach position.
        bx, by = snapshot.ball_position

        if self._free_kick_ball_ref is None:
            self._free_kick_ball_ref = (bx, by)

        # ball_in_play only counts after the kicker has actually fired — prevents
        # early-exit if the ball drifts >0.5m before the kick happens.
        ball_in_play = (
            self._free_kick_has_kicked
            and math.hypot(bx - self._free_kick_ball_ref[0], by - self._free_kick_ball_ref[1]) > 0.5
        )

        # Once ball is in play, check if a non-kicker own robot has touched the ball
        # (within 0.15 m) — that clears the double-touch restriction on the kicker.
        if ball_in_play and not self._free_kick_double_touch_cleared:
            for rid in robot_ids:
                if rid == kicker_id:
                    continue
                r = _find_robot(snapshot, rid)
                if r is not None and math.hypot(r.position[0] - bx, r.position[1] - by) < 0.15:
                    self._free_kick_double_touch_cleared = True
                    break

        from TeamControl.bt.contracts.intent import IntentKick
        intents: list[Intent] = []
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            if robot_id == kicker_id:
                dist_to_ball = math.hypot(robot.position[0] - bx, robot.position[1] - by)
                approach_x = bx - 0.25 * self._attack_sign
                dist_to_approach = math.hypot(robot.position[0] - approach_x, robot.position[1] - by)
                on_correct_side = (robot.position[0] - bx) * self._attack_sign < -0.05
                if dist_to_ball < 0.15 and on_correct_side:
                    self._free_kick_kicker_ready = True

                if ball_in_play and self._free_kick_double_touch_cleared:
                    # Teammate touched ball — double-touch restriction lifted, run normal BT.
                    bb.last_intent = bb.current_intent
                    bb.current_intent = None
                    tree = self.trees[bb.current_role]
                    if hasattr(tree, "set_snapshot") and hasattr(tree, "tick"):
                        tree.set_snapshot(snapshot)
                        tree.tick(bb)
                    else:
                        if hasattr(tree, "_blackboard_ref"):
                            tree._blackboard_ref[0] = bb
                        tree.tick_once()
                    if bb.current_intent is not None:
                        intents.append(bb.current_intent)
                elif ball_in_play:
                    # Ball in play but no teammate touched yet — hold position (double-touch rule).
                    if self._free_kick_kicker_hold is None:
                        self._free_kick_kicker_hold = (robot.position[0], robot.position[1])
                if self._free_kick_kicker_ready:
                    bb.current_intent = IntentKick(target_pos=self._opp_goal)
                elif dist_to_approach < 0.10:
                    # Reached approach position — now drive straight into the ball
                    bb.current_intent = IntentMove(
                        target_pos=self._free_kick_kicker_hold, target_orientation=None
                    )
                    intents.append(bb.current_intent)
                else:
                    # Ball not yet in play — approach and kick.
                    if self._free_kick_kicker_ready:
                        if not self._free_kick_has_kicked:
                            self._free_kick_has_kicked = True
                            self._free_kick_ball_ref = (bx, by)  # ref from actual kick position
                        bb.current_intent = IntentKick(target_pos=self._enemy_goal)
                    elif dist_to_approach < 0.15:
                        bb.current_intent = IntentMove(target_pos=(bx, by), target_orientation=None)
                    else:
                        bb.current_intent = IntentMove(target_pos=(approach_x, by), target_orientation=None)
                    intents.append(bb.current_intent)
            elif ROLE_ASSIGNMENT.get(robot_id) == RoleType.GOALIE:
                on_line = abs(robot.position[0] - self._own_goal_line_x) < 0.30
                if on_line:
                    by = snapshot.ball_position[1]
                    target = (self._own_goal_line_x, max(-_GOAL_HW_M, min(_GOAL_HW_M, by)))
                else:
                    target = (self._own_goal_line_x, 0.0)
                bb.current_intent = IntentMove(
                    target_pos=target, target_orientation=None
                )
                intents.append(bb.current_intent)
            else:
                if ball_in_play:
                    bb.last_intent = bb.current_intent
                    bb.current_intent = None
                    tree = self.trees[bb.current_role]
                    if hasattr(tree, "set_snapshot") and hasattr(tree, "tick"):
                        tree.set_snapshot(snapshot)
                        tree.tick(bb)
                    else:
                        if hasattr(tree, "_blackboard_ref"):
                            tree._blackboard_ref[0] = bb
                        tree.tick_once()
                    if bb.current_intent is not None:
                        intents.append(bb.current_intent)
                else:
                    slot = self._free_kick_support_slots.get(robot_id)
                    if slot is None:
                        slot = len(self._free_kick_support_slots)
                        self._free_kick_support_slots[robot_id] = slot
                    offset = FREE_KICK_SUPPORT_OFFSETS[slot % len(FREE_KICK_SUPPORT_OFFSETS)]
                    raw = (bx + offset[0], by + offset[1])
                    support_pos = (
                        max(-_HALF_LEN_M + 0.2, min(_HALF_LEN_M - 0.2, raw[0])),
                        max(-_HALF_WID_M + 0.2, min(_HALF_WID_M - 0.2, raw[1])),
                    )
                    bb.current_intent = IntentMove(
                        target_pos=support_pos, target_orientation=None
                    )
                    intents.append(bb.current_intent)
        return intents

    def _handle_opp_free_kick(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """OPP_FREE_KICK: opponent has the free kick — all our robots stay ≥0.5m from ball.

        Goalie tracks ball on own goal line (y only). Non-goalie robots move to
        defensive spread positions offset from the ball, all at legal clearance distance.
        """
        bx, by = snapshot.ball_position
        intents: list[Intent] = []

        # Record ball position when enemy free kick starts (first tick of this phase).
        if self._enemy_free_kick_ball_ref is None:
            self._enemy_free_kick_ball_ref = (bx, by)

        ball_in_play = math.hypot(
            bx - self._enemy_free_kick_ball_ref[0],
            by - self._enemy_free_kick_ball_ref[1],
        ) > 0.5

        if ball_in_play:
            # Enemy kicked — restrictions lifted, run normal BT for all robots.
            for robot_id in robot_ids:
                robot = _find_robot(snapshot, robot_id)
                if robot is None:
                    continue
                bb = self.blackboards[robot_id]
                bb.last_intent = bb.current_intent
                bb.current_intent = None
                tree = self.trees[bb.current_role]
                if hasattr(tree, "set_snapshot") and hasattr(tree, "tick"):
                    tree.set_snapshot(snapshot)
                    tree.tick(bb)
                else:
                    if hasattr(tree, "_blackboard_ref"):
                        tree._blackboard_ref[0] = bb
                    tree.tick_once()
                if bb.current_intent is not None:
                    intents.append(bb.current_intent)
            return intents

        # Ball not yet in play — hold wall formation at 0.55m clearance.
        CLEARANCE = STOP_BALL_CLEARANCE  # 0.55m (above the 0.5m SSL rule)
        wall_x = bx - CLEARANCE * self._attack_sign  # step toward own goal from ball
        spread_y = [0.0, -0.6, 0.6, -1.2, 1.2]

        slot_idx = 0
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]

            if ROLE_ASSIGNMENT.get(robot_id) == RoleType.GOALIE:
                target = (self._own_goal_line_x, max(-_GOAL_HW_M, min(_GOAL_HW_M, by)))
            else:
                sy = spread_y[slot_idx % len(spread_y)]
                slot_idx += 1
                target = (wall_x, by + sy)
                dist = math.hypot(target[0] - bx, target[1] - by)
                if dist < CLEARANCE:
                    scale = CLEARANCE / max(dist, 1e-6)
                    target = (bx + (target[0] - bx) * scale, by + (target[1] - by) * scale)

            bb.current_intent = IntentMove(target_pos=target, target_orientation=None, max_speed=1.4)
            intents.append(bb.current_intent)
        return intents

    # ------------------------------------------------------------------
    # Corner / goal kick — refined free-kick variants (§5.3)
    # ------------------------------------------------------------------

    def _classify_free_kick(self, snapshot: Snapshot, enemy: bool) -> GamePhase:
        """Classify a free kick into corner / goal kick / plain by ball location.

        SSL awards only a direct/indirect free kick when the ball leaves over a
        goal line; the *kind* depends on which goal line it sits near.

        For OUR free kick:   near opp goal ⇒ CORNER_KICK, near our goal ⇒ GOAL_KICK.
        For ENEMY free kick: near our goal ⇒ ENEMY_CORNER_KICK (they attack),
                             near opp goal ⇒ ENEMY_GOAL_KICK (they restart deep).
        Anything mid-field stays a plain (ENEMY_)FREE_KICK.
        """
        bx, _by = snapshot.ball_position
        opp_goal_line_x = self._opp_goal[0]
        own_goal_line_x = self._own_goal_line_x
        near_opp = abs(bx - opp_goal_line_x) <= FREE_KICK_GOAL_LINE_BAND
        near_own = abs(bx - own_goal_line_x) <= FREE_KICK_GOAL_LINE_BAND
        if enemy:
            if near_own:
                return GamePhase.ENEMY_CORNER_KICK
            if near_opp:
                return GamePhase.ENEMY_GOAL_KICK
            return GamePhase.ENEMY_FREE_KICK
        if near_opp:
            return GamePhase.CORNER_KICK
        if near_own:
            return GamePhase.GOAL_KICK
        return GamePhase.FREE_KICK

    def _lock_free_kick_kicker(self, snapshot: Snapshot, robot_ids: list[int]) -> None:
        """Pick the closest non-goalie robot as the kicker (idempotent)."""
        if self._free_kick_kicker_id is not None:
            return
        best_dist = float("inf")
        for robot_id in robot_ids:
            if self._is_goalie(robot_id):
                continue
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            d = math.hypot(
                robot.position[0] - snapshot.ball_position[0],
                robot.position[1] - snapshot.ball_position[1],
            )
            if d < best_dist:
                best_dist = d
                self._free_kick_kicker_id = robot_id

    def _kicker_take_kick(self, robot, bb, snapshot, kick_target) -> None:
        """Drive the locked kicker behind the ball and fire toward *kick_target*.

        Geometry is target-relative (approach from the side of the ball opposite
        the target) so it works for any kick direction — corner crosses and goal
        clearances included. The adapter's kick skill does the fine alignment;
        ``self._free_kick_kicker_ready`` latches once the robot is in contact.
        """
        from TeamControl.bt.contracts.intent import IntentKick
        bx, by = snapshot.ball_position
        tx, ty = kick_target
        dirx, diry = tx - bx, ty - by
        dlen = math.hypot(dirx, diry) or 1.0
        ux, uy = dirx / dlen, diry / dlen
        approach = (bx - ux * 0.25, by - uy * 0.25)
        dist_to_ball = math.hypot(robot.position[0] - bx, robot.position[1] - by)
        dist_to_approach = math.hypot(
            robot.position[0] - approach[0], robot.position[1] - approach[1]
        )
        behind_ball = (
            (robot.position[0] - bx) * ux + (robot.position[1] - by) * uy
        ) < -0.04
        if dist_to_ball < 0.15 and behind_ball:
            self._free_kick_kicker_ready = True
        if self._free_kick_kicker_ready:
            bb.current_intent = IntentKick(target_pos=kick_target)
        elif dist_to_approach < 0.10:
            bb.current_intent = IntentMove(target_pos=(bx, by), target_orientation=None)
        else:
            bb.current_intent = IntentMove(target_pos=approach, target_orientation=None)

    def _handle_corner_kick(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """CORNER_KICK: our attacking free kick near the opponent goal line.

        Kicker crosses the ball to a dangerous central point in front of the
        opponent goal; supporters occupy box slots to follow up; goalie holds
        the own goal line.
        """
        self._lock_free_kick_kicker(snapshot, robot_ids)
        kicker_id = self._free_kick_kicker_id
        opp_goal_x = self._opp_goal[0]
        # Cross target: in front of the opponent goal, just outside its area.
        kick_target = (opp_goal_x - self._attack_sign * CORNER_KICK_TARGET_DEPTH, 0.0)

        intents: list[Intent] = []
        slot = 0
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            if robot_id == kicker_id:
                self._kicker_take_kick(robot, bb, snapshot, kick_target)
            elif self._is_goalie(robot_id):
                by = max(-1.0, min(1.0, snapshot.ball_position[1]))
                bb.current_intent = IntentMove(
                    target_pos=(self._own_goal_line_x, by), target_orientation=None
                )
            else:
                depth, sy = CORNER_BOX_SLOTS[slot % len(CORNER_BOX_SLOTS)]
                slot += 1
                target = (opp_goal_x - self._attack_sign * depth, sy)
                bb.current_intent = IntentMove(target_pos=target, target_orientation=None)
            intents.append(bb.current_intent)
        return intents

    def _handle_goal_kick(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """GOAL_KICK: our defensive free kick near our own goal line.

        Kicker clears the ball upfield and wide of our goal; supporters push up
        to outlet positions to receive; goalie holds the line.
        """
        self._lock_free_kick_kicker(snapshot, robot_ids)
        kicker_id = self._free_kick_kicker_id
        bx, by = snapshot.ball_position
        own_goal_x = self._own_goal_line_x
        # Clear straight upfield (toward the opponent half) along the ball's y.
        kick_target = (bx + self._attack_sign * GOAL_KICK_CLEAR_DEPTH, by)

        intents: list[Intent] = []
        slot = 0
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            if robot_id == kicker_id:
                self._kicker_take_kick(robot, bb, snapshot, kick_target)
            elif self._is_goalie(robot_id):
                by_clamped = max(-1.0, min(1.0, snapshot.ball_position[1]))
                bb.current_intent = IntentMove(
                    target_pos=(own_goal_x, by_clamped), target_orientation=None
                )
            else:
                depth, sy = GOAL_KICK_OUTLET_SLOTS[slot % len(GOAL_KICK_OUTLET_SLOTS)]
                slot += 1
                target = (own_goal_x + self._attack_sign * depth, sy)
                bb.current_intent = IntentMove(target_pos=target, target_orientation=None)
            intents.append(bb.current_intent)
        return intents

    def _handle_opp_corner_kick(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """ENEMY_CORNER_KICK: opponent attacks near our goal line.

        Goalie tracks the ball on the line; the rest pack a defensive screen
        across the goal mouth between ball and goal, each kept ≥ STOP_BALL_CLEARANCE
        from the ball so we never give away a defender-too-close foul (§5.4).
        """
        bx, by = snapshot.ball_position
        own_goal_x = self._own_goal_line_x
        intents: list[Intent] = []
        slot = 0
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            if self._is_goalie(robot_id):
                by_clamped = max(-1.0, min(1.0, by))
                target = (own_goal_x, by_clamped)
            else:
                depth, sy = OPP_CORNER_DEFEND_SLOTS[slot % len(OPP_CORNER_DEFEND_SLOTS)]
                slot += 1
                target = (own_goal_x + self._attack_sign * depth, sy)
                # Never sit closer than the legal clearance to the ball.
                if math.hypot(target[0] - bx, target[1] - by) < STOP_BALL_CLEARANCE:
                    target = _nudge_away_from_ball(target, (bx, by), STOP_BALL_CLEARANCE)
            bb.current_intent = IntentMove(
                target_pos=target, target_orientation=None, max_speed=1.4
            )
            intents.append(bb.current_intent)
        return intents

    def _handle_prepare_penalty(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """PREPARE_PENALTY: position our robots before we shoot.

        Kicker (robot 5) approaches ball without touching it.
        All others move to 1m behind ball (own-goal side). Goalie to own goal line.
        """
        bx, by = snapshot.ball_position
        wait_x = bx - self._attack_sign * 1.1
        spread_y = [0.0, -0.7, 0.7, -1.4]
        intents: list[Intent] = []
        slot = 0
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            if robot_id == 5:
                # Approach but do NOT kick yet.
                approach_x = bx - 0.25 * self._attack_sign
                bb.current_intent = IntentMove(target_pos=(approach_x, by), target_orientation=None)
            elif self._is_goalie(robot_id):
                by_clamped = max(-1.0, min(1.0, by))
                bb.current_intent = IntentMove(
                    target_pos=(self._own_goal_line_x, by_clamped), target_orientation=None
                )
            else:
                sy = spread_y[slot % len(spread_y)]
                slot += 1
                final_target = (wait_x, by + sy)
                needs_cross = (robot.position[0] - bx) * self._attack_sign > 0
                clear_of_ball_y = abs(robot.position[1] - by) > 0.6
                if needs_cross and not clear_of_ball_y:
                    detour_y = by + (1.5 if slot % 2 == 0 else -1.5)
                    bb.current_intent = IntentMove(
                        target_pos=(robot.position[0], detour_y), target_orientation=None
                    )
                else:
                    bb.current_intent = IntentMove(target_pos=final_target, target_orientation=None)
            intents.append(bb.current_intent)
        return intents

    def _handle_prepare_penalty_opp(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """PREPARE_PENALTY_OPP: position our robots before opponent shoots.

        Goalie to own goal line. All others ≥1m behind ball (toward centre — away
        from our goal, same side as the kicker's run-up).
        """
        bx, by = snapshot.ball_position
        # During enemy penalty, "behind the ball" means TOWARD centre, not toward our goal.
        wait_x = bx + self._attack_sign * 1.1
        # Cluster all non-goalie robots together on the positive-y side, lined up in a row.
        cluster_y = [0.5, 1.0, 1.5, 2.0, 2.5]
        intents: list[Intent] = []
        slot = 0
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            if self._is_goalie(robot_id):
                by_clamped = max(-1.0, min(1.0, by))
                bb.current_intent = IntentMove(
                    target_pos=(self._own_goal_line_x, by_clamped), target_orientation=None
                )
            else:
                target_y = cluster_y[slot % len(cluster_y)]
                slot += 1
                bb.current_intent = IntentMove(target_pos=(wait_x, target_y), target_orientation=None)
            intents.append(bb.current_intent)
        return intents

    def _handle_penalty_shoot(
        self, snapshot: Snapshot, robot_ids: list[int]
    ) -> list[Intent]:
        """PENALTY_SHOOT: kicker approaches and shoots; all others stay ≥1m behind ball.

        SSL rule: all robots except the kicker and defending keeper must stay ≥1m
        behind the ball (on the own-goal side) throughout the penalty procedure.
        """
        from TeamControl.bt.contracts.intent import IntentKick
        PENALTY_KICKER_ID = 5
        bx, by = snapshot.ball_position
        # 1.1m behind ball = toward own goal from ball
        wait_x = bx - self._attack_sign * 1.1
        spread_y = [0.0, -0.7, 0.7, -1.4]

        # Kicker only shoots once all supporters are ≥1m behind ball.
        supporters_ready = all(
            (bx - robot.position[0]) * self._attack_sign > 0.9
            for rid in robot_ids
            if rid != PENALTY_KICKER_ID
            and not self._is_goalie(rid)
            and (robot := _find_robot(snapshot, rid)) is not None
        )

        intents: list[Intent] = []
        slot = 0
        for robot_id in robot_ids:
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            bb = self.blackboards[robot_id]
            if robot_id == PENALTY_KICKER_ID:
                approach_x = bx - 0.25 * self._attack_sign
                if not supporters_ready:
                    bb.current_intent = IntentMove(target_pos=(approach_x, by), target_orientation=None)
                else:
                    dist_to_ball = math.hypot(robot.position[0] - bx, robot.position[1] - by)
                    dist_to_approach = math.hypot(robot.position[0] - approach_x, robot.position[1] - by)
                    on_correct_side = (robot.position[0] - bx) * self._attack_sign < -0.05
                    if dist_to_ball < 0.15 and on_correct_side:
                        bb.current_intent = IntentKick(target_pos=self._enemy_goal)
                    elif dist_to_approach < 0.15:
                        bb.current_intent = IntentMove(target_pos=(bx, by), target_orientation=None)
                    else:
                        bb.current_intent = IntentMove(target_pos=(approach_x, by), target_orientation=None)
            elif self._is_goalie(robot_id):
                by_clamped = max(-1.0, min(1.0, by))
                bb.current_intent = IntentMove(
                    target_pos=(self._own_goal_line_x, by_clamped), target_orientation=None
                )
            else:
                sy = spread_y[slot % len(spread_y)]
                slot += 1
                bb.current_intent = IntentMove(
                    target_pos=(wait_x, by + sy), target_orientation=None
                )
            intents.append(bb.current_intent)
        return intents

    def _handle_penalty_defend(
        self, snapshot: Snapshot, robot_ids: list[int]
    ) -> list[Intent]:
        """PENALTY_DEFEND: goalie tracks ball on goal line; others hold.
        
        Non-goalies hold 1m behind ball on own-goal side, with safety clamping
        to ensure they never drift past the goal line.
        """
        intents: list[Intent] = []
        for robot_id in robot_ids:
            if _find_robot(snapshot, robot_id) is None:
                continue
            bb = self.blackboards[robot_id]
            if self._is_goalie(robot_id):
                # Stay on goal line, track ball's y — clamped to goal mouth.
                by = max(-1.0, min(1.0, snapshot.ball_position[1]))
                target = (self._own_goal_line_x, by)
            else:
                # Non-goalie defenders: ≥1m behind ball toward centre, clustered on +y side.
                bx, by = snapshot.ball_position
                wait_x = bx + self._attack_sign * 1.1
                cluster_y = [0.5, 1.0, 1.5, 2.0, 2.5]
                sy = cluster_y[robot_id % len(cluster_y)]
                target = (wait_x, sy)
            bb.current_intent = IntentMove(target_pos=target, target_orientation=None)
            intents.append(bb.current_intent)
        return intents

    # ------------------------------------------------------------------
    # Normal role-tree dispatch (RUNNING state)
    # ------------------------------------------------------------------

    def _normal_tick(self, snapshot: Snapshot, robot_ids: list[int]) -> list[Intent]:
        """Run each robot's role tree — the original coordinator behaviour."""
        snapshot_ids: set[int] = {r.robot_id for r in snapshot.own_robots}
        intents: list[Intent] = []

        for robot_id in robot_ids:
            if robot_id not in snapshot_ids:
                continue

            bb = self.blackboards[robot_id]
            bb.last_intent = bb.current_intent
            bb.current_intent = None

            tree = self.trees[bb.current_role]
            if hasattr(tree, "set_snapshot") and hasattr(tree, "tick"):
                tree.set_snapshot(snapshot)
                tree.tick(bb)
            else:
                if hasattr(tree, "_blackboard_ref"):
                    tree._blackboard_ref[0] = bb
                tree.tick_once()

            if bb.current_intent is not None:
                intents.append(bb.current_intent)

        return intents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Robot designated to take kickoffs — pre-positioned at center circle during PREPARE_KICKOFF.
    KICKOFF_KICKER_ID: int = 3

    def _lock_kickoff_kicker(self, snapshot: Snapshot, robot_ids: list[int]) -> None:
        """Lock in the designated kickoff kicker (idempotent).

        Robot 3 is always the kicker — it is pre-positioned inside the center
        circle during PREPARE_KICKOFF and is the only robot allowed there per §5.3.2.
        Falls back to closest non-goalie if robot 5 is absent from the snapshot.
        """
        if self._kickoff_kicker_id is not None:
            return
        if self.KICKOFF_KICKER_ID in robot_ids and _find_robot(snapshot, self.KICKOFF_KICKER_ID):
            self._kickoff_kicker_id = self.KICKOFF_KICKER_ID
            return
        # Fallback: robot 5 absent — pick closest non-goalie.
        best_dist = float("inf")
        for robot_id in robot_ids:
            if self._is_goalie(robot_id):
                continue
            robot = _find_robot(snapshot, robot_id)
            if robot is None:
                continue
            d = math.hypot(
                robot.position[0] - snapshot.ball_position[0],
                robot.position[1] - snapshot.ball_position[1],
            )
            if d < best_dist:
                best_dist = d
                self._kickoff_kicker_id = robot_id

    def _ensure_blackboards(self, snapshot: Snapshot, robot_ids: list[int]) -> None:
        """Create blackboards only for robots present in both robot_ids and snapshot."""
        snapshot_ids = {r.robot_id for r in snapshot.own_robots}
        for robot_id in robot_ids:
            if robot_id not in snapshot_ids:
                continue
            if robot_id not in self.blackboards:
                role = self._role_of(robot_id)
                self.blackboards[robot_id] = RobotBlackboard(
                    robot_id=robot_id,
                    current_role=role,
                )
            else:
                static_role = self._role_of(robot_id)
                if not self.heuristic_role_swap or static_role == RoleType.GOALIE:
                    self.blackboards[robot_id].current_role = static_role

    def _apply_heuristic_roles(self, snapshot: Snapshot, robot_ids: list[int]) -> None:
        """Update non-goalie roles when dynamic role swapping is enabled."""
        if not self.heuristic_role_swap:
            return

        present_ids = {
            robot.robot_id
            for robot in snapshot.own_robots
            if robot.robot_id in robot_ids and robot.robot_id in self.blackboards
        }
        if not present_ids:
            return

        now = time.monotonic()
        current_roles = {
            robot_id: self.blackboards[robot_id].current_role
            for robot_id in present_ids
        }
        time_since_last_swap = {
            robot_id: (
                now - self._role_swap_last_changed_at[robot_id]
                if robot_id in self._role_swap_last_changed_at
                else math.inf
            )
            for robot_id in present_ids
        }

        result = assign_roles_heuristically(
            snapshot,
            robot_ids,
            current_roles,
            base_roles=self.role_assignment,
            time_since_last_swap=time_since_last_swap,
            attack_goal=self._opp_goal,
            own_goal=(self._own_goal_line_x, 0.0),
            heuristic_weights=self.heuristic_weights,
        )

        for robot_id, new_role in result.roles.items():
            bb = self.blackboards.get(robot_id)
            if bb is None:
                continue

            old_role = bb.current_role
            bb.current_role = new_role
            if old_role != new_role:
                self._role_swap_last_changed_at[robot_id] = now

    def _role_of(self, robot_id: int) -> RoleType:
        return self.role_assignment.get(robot_id, RoleType.SUPPORTER)

    def _is_goalie(self, robot_id: int) -> bool:
        return self._role_of(robot_id) == RoleType.GOALIE
