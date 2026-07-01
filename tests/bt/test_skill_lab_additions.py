"""Tests for the Skill Lab additions and the counter-attack/side-aware refactor:
receive_pass, shoot_open_goal, kick_at_goal, move_then_attack, pass_to_outlet,
the shared side-aware helpers, the reachable kick trigger, and the adapter's
tracked ball-velocity feed.
"""
from __future__ import annotations

import pytest

from TeamControl.bt.adapter import _ball_velocity_mps
from TeamControl.bt.contracts.intent import (
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentOrient,
)
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.skills import _shared
from TeamControl.skills._shared import (
    best_goal_target,
    forward_outlet,
    in_own_half,
    opp_goal,
    own_goal,
    set_attack_sign,
)
from TeamControl.skills.chase_ball import CHASE_BALL_SPEED_GAIN, chase_ball
from TeamControl.skills.kick_at_goal import kick_at_goal
from TeamControl.skills.move_then_attack import move_then_attack
from TeamControl.skills.pass_to_outlet import pass_to_outlet
from TeamControl.skills.receive_pass import receive_pass
from TeamControl.skills.skills import BEHAVIOURS_BY_ID, reset_robot_state


@pytest.fixture(autouse=True)
def _reset_skill_state():
    """Isolate the module-level attack sign + kick-phase caches between tests."""
    set_attack_sign(1.0)
    _shared.approach_cache.clear()
    _shared.kick_phase.clear()
    _shared.seq_phase.clear()
    yield
    set_attack_sign(1.0)


def _snap(own, enemy=(), ball=(0.0, 0.0), ball_vel=(0.0, 0.0)) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=ball_vel,
        own_robots=list(own),
        enemy_robots=list(enemy),
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _r(rid, pos, o=0.0) -> RobotState:
    return RobotState(robot_id=rid, position=pos, orientation=o)


# ── adapter ball velocity ─────────────────────────────────────────────────────

class _WMTraj:
    def get_ball_trajectory(self, horizon_ms=None):
        # ((pred_x, pred_y), (vx, vy)) in mm / mm-per-s.
        return ((1500.0, 0.0), (2000.0, -1000.0))


def test_ball_velocity_reads_tracked_map_and_converts_to_mps():
    assert _ball_velocity_mps(_WMTraj()) == (2.0, -1.0)


def test_ball_velocity_defaults_to_zero_without_trajectory():
    class _WMNoTraj:
        pass

    assert _ball_velocity_mps(_WMNoTraj()) == (0.0, 0.0)


def test_ball_velocity_zero_when_ball_untracked():
    class _WMNone:
        def get_ball_trajectory(self, horizon_ms=None):
            return None

    assert _ball_velocity_mps(_WMNone()) == (0.0, 0.0)


# ── side-aware direction helpers ──────────────────────────────────────────────

def test_opp_own_goal_and_half_flip_with_attack_sign():
    set_attack_sign(1.0)
    assert opp_goal() == (4.5, 0.0)
    assert own_goal() == (-4.5, 0.0)
    assert in_own_half(-1.0) and not in_own_half(1.0)

    set_attack_sign(-1.0)
    assert opp_goal() == (-4.5, 0.0)
    assert own_goal() == (4.5, 0.0)
    assert in_own_half(1.0) and not in_own_half(-1.0)


# ── shoot aim (keeper-aware, side-aware) ──────────────────────────────────────

def test_best_goal_target_aims_away_from_keeper():
    # Keeper high (+y) in the mouth → aim low (-y), the open side. Attacking +x.
    snap = _snap(own=[_r(1, (3.0, 0.0))], enemy=[_r(0, (4.5, 0.4))], ball=(3.0, 0.0))
    aim = best_goal_target(snap)
    assert aim[0] == 4.5
    assert aim[1] < 0.0


def test_best_goal_target_defaults_to_centre_without_opponents():
    snap = _snap(own=[_r(1, (3.0, 0.0))], ball=(3.0, 0.0))
    assert best_goal_target(snap) == (4.5, 0.0)


def test_best_goal_target_follows_attack_direction():
    # Attacking -x → aim point is at the -x goal.
    set_attack_sign(-1.0)
    snap = _snap(own=[_r(1, (-3.0, 0.0))], enemy=[_r(0, (-4.5, 0.4))], ball=(-3.0, 0.0))
    aim = best_goal_target(snap)
    assert aim[0] == -4.5
    assert aim[1] < 0.0


# ── forward outlet selection (side-aware) ─────────────────────────────────────

def test_forward_outlet_picks_most_advanced_open_teammate():
    own = [
        _r(1, (0.0, 0.0)),   # carrier
        _r(2, (1.0, 0.0)),   # forward, open
        _r(3, (2.0, 1.2)),   # further forward, open → should win
        _r(4, (-1.0, 0.0)),  # behind the carrier → ignored
    ]
    snap = _snap(own=own, ball=(0.0, 0.0))
    assert forward_outlet(snap, own[0]) == (2.0, 1.2)


def test_forward_outlet_follows_attack_direction():
    # Attacking -x → "forward" is decreasing x.
    set_attack_sign(-1.0)
    own = [_r(1, (0.0, 0.0)), _r(2, (-1.0, 0.0)), _r(3, (-2.0, 1.2)), _r(4, (1.0, 0.0))]
    snap = _snap(own=own, ball=(0.0, 0.0))
    assert forward_outlet(snap, own[0]) == (-2.0, 1.2)


def test_forward_outlet_rejects_marked_teammate():
    own = [_r(1, (0.0, 0.0)), _r(2, (2.0, 0.0))]
    enemy = [_r(0, (2.1, 0.0))]  # tightly marking teammate 2
    snap = _snap(own=own, enemy=enemy, ball=(0.0, 0.0))
    assert forward_outlet(snap, own[0]) is None


def test_pass_to_outlet_faces_ball_when_no_outlet():
    own = [_r(1, (0.0, 0.0))]
    snap = _snap(own=own, ball=(1.0, 1.0))
    assert isinstance(pass_to_outlet(snap, own[0], None), IntentOrient)


def test_chase_ball_targets_ball_directly_with_speed_gain():
    own = [_r(1, (0.0, 0.0), o=0.0)]
    snap = _snap(own=own, ball=(1.0, 1.0))
    intent = chase_ball(snap, own[0], None)

    assert isinstance(intent, IntentMove)
    assert intent.target_pos == snap.ball_position
    assert intent.target_orientation == pytest.approx(0.7853981633974483)
    assert intent.max_speed is None
    assert intent.speed_gain == CHASE_BALL_SPEED_GAIN


def test_chase_ball_returns_none_without_visible_robot():
    snap = _snap(own=[], ball=(1.0, 0.0))
    assert chase_ball(snap, None, None) is None


# ── kick skills emit IntentKick; the PD executor fires the kicker ─────────────

def test_kick_at_goal_emits_kick_at_open_aim():
    # Robot far from the ball still emits an IntentKick at the open aim — the
    # motion layer handles the drive-in and the actual kicker fire.
    own = [_r(1, (-2.0, 0.0), o=0.0)]
    snap = _snap(own=own, ball=(0.0, 0.0))  # no enemies → aim goal centre
    intent = kick_at_goal(snap, own[0], None)
    assert isinstance(intent, IntentKick)
    assert intent.target_pos == (4.5, 0.0)


def test_executor_fires_kicker_when_pose_ready():
    # Robot just behind the ball, aligned toward the +x goal, within contact
    # range → the PD MotionExecutor sets kick=1 (the whole point of the fix).
    from TeamControl.bt.adapter import MotionExecutor

    own = [_r(1, (-0.15, 0.0), o=0.0)]
    snap = _snap(own=own, ball=(0.0, 0.0))
    cmd = MotionExecutor().resolve_command(IntentKick(target_pos=(4.5, 0.0)), 1, snap, True)
    assert cmd is not None
    assert cmd.kick == 1


def test_executor_does_not_fire_when_far_from_ball():
    from TeamControl.bt.adapter import MotionExecutor

    own = [_r(1, (-1.0, 0.0), o=0.0)]  # too far to be in contact
    snap = _snap(own=own, ball=(0.0, 0.0))
    cmd = MotionExecutor().resolve_command(IntentKick(target_pos=(4.5, 0.0)), 1, snap, True)
    assert cmd is not None
    assert cmd.kick == 0
    # ...but it should be driving toward the ball (non-zero command).
    assert (cmd.vx, cmd.vy, cmd.w) != (0.0, 0.0, 0.0)


# ── move_then_attack counter-attack logic ─────────────────────────────────────

def test_move_then_attack_shoots_in_enemy_half():
    reset_robot_state(1)
    own = [_r(1, (3.88, 0.0), o=0.0)]
    snap = _snap(own=own, ball=(4.0, 0.0))  # enemy half (+x) → open-goal shot
    intent = move_then_attack(snap, own[0], None)
    assert isinstance(intent, IntentKick)
    assert intent.target_pos == (4.5, 0.0)


def test_move_then_attack_passes_to_outlet_in_own_half():
    reset_robot_state(1)
    own = [_r(1, (-3.12, 0.0), o=0.0), _r(2, (-1.0, 0.0))]  # teammate forward + open
    snap = _snap(own=own, ball=(-3.0, 0.0))  # own half (-x)
    intent = move_then_attack(snap, own[0], None)
    assert isinstance(intent, IntentKick)
    assert intent.target_pos == (-1.0, 0.0)


# ── receive_pass ──────────────────────────────────────────────────────────────

def test_receive_pass_holds_spot_when_ball_far():
    own = [_r(1, (0.0, 0.0))]
    snap = _snap(own=own, ball=(5.0, 5.0))
    intent = receive_pass(snap, own[0], (2.0, 0.0))
    assert isinstance(intent, IntentMove)
    assert intent.target_pos == (2.0, 0.0)


def test_receive_pass_traps_when_ball_near():
    own = [_r(1, (0.0, 0.0))]
    snap = _snap(own=own, ball=(0.5, 0.0))
    assert isinstance(receive_pass(snap, own[0], (2.0, 0.0)), IntentDribble)


def test_receive_pass_leads_a_moving_ball():
    own = [_r(1, (0.0, 0.0))]
    snap = _snap(own=own, ball=(0.5, 0.0), ball_vel=(1.0, 0.0))
    intent = receive_pass(snap, own[0], (2.0, 0.0))
    assert isinstance(intent, IntentDribble)
    assert intent.target_pos[0] > 0.5  # aimed ahead of the ball


# ── registry wiring ───────────────────────────────────────────────────────────

def test_new_skills_registered():
    for skill_id in ("chase_ball", "receive_pass", "shoot_open_goal", "pass_to_outlet"):
        assert skill_id in BEHAVIOURS_BY_ID
