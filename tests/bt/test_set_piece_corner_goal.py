"""Corner-kick / goal-kick free-kick refinement tests (§5.3).

SSL has no dedicated "corner" or "goal kick" referee command — both are
direct/indirect free kicks. The Coordinator classifies a free kick by where the
ball left the field:

    near opponent goal line  -> CORNER_KICK        (we attack)
    near our own goal line    -> GOAL_KICK          (we clear)
    enemy free kick near our goal  -> ENEMY_CORNER_KICK (we defend the mouth)
    enemy free kick near their goal -> ENEMY_GOAL_KICK  (defensive spread)

Geometry note: us_positive=False ⇒ own goal at x=-4.5, opponent goal at x=+4.5,
we attack toward +x. us_positive=True mirrors everything.
"""
from __future__ import annotations

import math

from TeamControl.bt.contracts.intent import IntentKick, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.coordinator import Coordinator, STOP_BALL_CLEARANCE

_ALL_IDS = [0, 1, 2, 3, 4, 5]
_DEFAULT_POS = {
    0: (-4.0, 0.0),   # goalie
    1: (0.0, 0.0),
    2: (0.5, 1.0),
    3: (0.5, -1.0),
    4: (-1.0, 1.5),
    5: (-1.0, -1.5),
}
# Disable the movement-safety post-processing so we test the raw set-piece
# positioning the handlers produce, not the field/defense-area clamps applied
# on top of them.
_NO_SAFETY = {
    "keep_robots_in_bounds": False,
    "keep_goalie_in_goal_box": False,
    "keep_non_goalies_out_of_goalie_box": False,
    "avoid_ball_touch_in_opponent_defense_area": False,
}


def _snapshot(ball_pos, phase=GamePhase.FREE_KICK, positions=None):
    positions = positions or _DEFAULT_POS
    own = [
        RobotState(robot_id=i, position=positions[i], orientation=0.0)
        for i in _ALL_IDS
    ]
    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=(0.0, 0.0),
        own_robots=own,
        enemy_robots=[],
        referee_state=RefereeState(game_phase=phase, score=(0, 0)),
    )


def _coord(us_positive=False):
    return Coordinator(trees={}, us_positive=us_positive, movement_safety=_NO_SAFETY)


def _intents_by_id(coord):
    return {rid: bb.current_intent for rid, bb in coord.blackboards.items()}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Classification — which goal line the ball sits near decides the kind
# ---------------------------------------------------------------------------

def test_classify_our_free_kick_near_opp_goal_is_corner():
    coord = _coord(us_positive=False)  # opp goal at +4.5
    snap = _snapshot((4.0, 1.0))
    assert coord._classify_free_kick(snap, enemy=False) == GamePhase.CORNER_KICK


def test_classify_our_free_kick_near_own_goal_is_goal_kick():
    coord = _coord(us_positive=False)  # own goal at -4.5
    snap = _snapshot((-4.0, 1.0))
    assert coord._classify_free_kick(snap, enemy=False) == GamePhase.GOAL_KICK


def test_classify_midfield_free_kick_stays_plain():
    coord = _coord(us_positive=False)
    snap = _snapshot((0.0, 0.0))
    assert coord._classify_free_kick(snap, enemy=False) == GamePhase.FREE_KICK
    assert coord._classify_free_kick(snap, enemy=True) == GamePhase.ENEMY_FREE_KICK


def test_classify_enemy_free_kick_near_our_goal_is_enemy_corner():
    coord = _coord(us_positive=False)  # own goal at -4.5
    snap = _snapshot((-4.0, 1.0))
    assert coord._classify_free_kick(snap, enemy=True) == GamePhase.ENEMY_CORNER_KICK


def test_classify_enemy_free_kick_near_their_goal_is_enemy_goal_kick():
    coord = _coord(us_positive=False)  # opp goal at +4.5
    snap = _snapshot((4.0, 1.0))
    assert coord._classify_free_kick(snap, enemy=True) == GamePhase.ENEMY_GOAL_KICK


def test_classification_mirrors_for_us_positive_true():
    coord = _coord(us_positive=True)  # opp goal at -4.5, own goal at +4.5
    assert coord._classify_free_kick(_snapshot((-4.0, 1.0)), enemy=False) == GamePhase.CORNER_KICK
    assert coord._classify_free_kick(_snapshot((4.0, 1.0)), enemy=False) == GamePhase.GOAL_KICK


# ---------------------------------------------------------------------------
# Dispatch + positioning through tick()
# ---------------------------------------------------------------------------

def test_corner_kick_advances_supporters_and_keeps_goalie_home():
    coord = _coord(us_positive=False)
    coord.tick(_snapshot((4.0, 1.5), phase=GamePhase.FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.CORNER_KICK

    intents = _intents_by_id(coord)
    # Goalie holds our goal line (x=-4.5), ball-y clamped to the goal mouth.
    assert isinstance(intents[0], IntentMove)
    assert intents[0].target_pos[0] == -4.5

    kicker_id = coord._free_kick_kicker_id
    assert kicker_id is not None and kicker_id != 0

    # Every non-goalie, non-kicker supporter pushes up near the opponent goal.
    for rid in _ALL_IDS:
        if rid in (0, kicker_id):
            continue
        assert isinstance(intents[rid], IntentMove)
        assert intents[rid].target_pos[0] > 2.0, f"supporter {rid} not advanced"


def test_goal_kick_pushes_supporters_upfield_to_outlets():
    coord = _coord(us_positive=False)
    coord.tick(_snapshot((-4.0, 1.0), phase=GamePhase.FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.GOAL_KICK

    intents = _intents_by_id(coord)
    kicker_id = coord._free_kick_kicker_id
    for rid in _ALL_IDS:
        if rid in (0, kicker_id):
            continue
        # Outlet slots sit at least 2 m upfield of our own goal line (-4.5).
        assert intents[rid].target_pos[0] >= -2.6, f"supporter {rid} not upfield"


def test_enemy_corner_keeps_all_non_goalies_legal_distance_from_ball():
    coord = _coord(us_positive=False)
    ball = (-4.0, 1.5)
    coord.tick(_snapshot(ball, phase=GamePhase.ENEMY_FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.ENEMY_CORNER_KICK

    intents = _intents_by_id(coord)
    # Goalie defends the mouth on the goal line.
    assert intents[0].target_pos[0] == -4.5
    # No outfield robot is parked inside the 0.5 m (we use 0.55 m) ball exclusion.
    for rid in (1, 2, 3, 4, 5):
        assert _dist(intents[rid].target_pos, ball) >= STOP_BALL_CLEARANCE - 1e-6


def test_enemy_goal_kick_uses_generic_defensive_spread():
    coord = _coord(us_positive=False)
    ball = (4.0, 0.0)
    coord.tick(_snapshot(ball, phase=GamePhase.ENEMY_FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.ENEMY_GOAL_KICK
    intents = _intents_by_id(coord)
    for rid in (1, 2, 3, 4, 5):
        assert _dist(intents[rid].target_pos, ball) >= STOP_BALL_CLEARANCE - 1e-6


def test_kicker_kicks_once_in_contact_with_ball():
    coord = _coord(us_positive=False)
    # Ball on the centre line in front of the opp goal -> cross target is straight
    # back along -x, so "behind the ball" is simply the +x side. Park robot 1 in
    # contact on that side; it should be the (closest) kicker and fire at once.
    positions = dict(_DEFAULT_POS)
    positions[1] = (4.1, 0.0)  # 0.1 m behind ball at (4.0, 0.0), in contact
    coord.tick(
        _snapshot((4.0, 0.0), phase=GamePhase.FREE_KICK, positions=positions),
        _ALL_IDS,
    )
    kicker_id = coord._free_kick_kicker_id
    assert kicker_id == 1
    assert isinstance(coord.blackboards[1].current_intent, IntentKick)


# ---------------------------------------------------------------------------
# Kind is locked per episode and reset between episodes
# ---------------------------------------------------------------------------

def test_kind_locked_when_ball_moves_after_kick():
    coord = _coord(us_positive=False)
    coord.tick(_snapshot((4.0, 1.0), phase=GamePhase.FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.CORNER_KICK
    # Ball flies to midfield after the kick but the phase is still FREE_KICK:
    # the classification must NOT flap back to a plain free kick.
    coord.tick(_snapshot((0.0, 0.0), phase=GamePhase.FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.CORNER_KICK


def test_kind_resets_on_new_episode():
    coord = _coord(us_positive=False)
    coord.tick(_snapshot((4.0, 1.0), phase=GamePhase.FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.CORNER_KICK
    # STOP between restarts clears the lock...
    coord.tick(_snapshot((0.0, 0.0), phase=GamePhase.STOPPED), _ALL_IDS)
    assert coord._free_kick_kind is None
    # ...so the next midfield free kick is classified fresh as plain.
    coord.tick(_snapshot((0.0, 0.0), phase=GamePhase.FREE_KICK), _ALL_IDS)
    assert coord._free_kick_kind == GamePhase.FREE_KICK
