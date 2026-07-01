"""Tests for keep_ball_in_bounds — steering a dribbled ball to stay inside the
field lines (rule_following._keep_carried_ball_in_field).
"""
from __future__ import annotations

import pytest

from TeamControl.bt.contracts.blackboard import RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.tactics.rule_following import (
    MovementSafetyConfig,
    apply_rule_following,
    has_rule_following_enabled,
)

# Field 9 x 6, margin 0.05, carry 0.15 → safe interior lim_x=4.30, lim_y=2.80.
_CFG = MovementSafetyConfig(
    keep_robots_in_bounds=False,
    keep_goalie_in_goal_box=False,
    keep_non_goalies_out_of_goalie_box=False,
    avoid_ball_touch_in_opponent_defense_area=False,
    keep_ball_in_bounds=True,
    field_length=9.0,
    field_width=6.0,
    field_margin=0.05,
    ball_carry_margin=0.15,
    ball_save_margin=0.30,
)


def _snap(ball) -> Snapshot:
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=[RobotState(robot_id=1, position=(0.0, 0.0), orientation=0.0)],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _apply(intent, ball, cfg=_CFG):
    return apply_rule_following(
        snapshot=_snap(ball),
        robot_id=1,
        robot_pos=(0.0, 0.0),
        role=RoleType.ATTACKER,
        intent=intent,
        config=cfg,
        own_goal_line_x=-4.5,
        opponent_goal=(4.5, 0.0),
    )


def test_enabled_flag_turns_on_rule_following():
    assert has_rule_following_enabled(_CFG) is True


def test_dribble_target_on_touchline_pulled_inside():
    # Ball interior, dribble aimed at the +y sideline → clamped to the safe
    # interior so the carried ball stays in.
    out = _apply(IntentDribble(target_pos=(0.0, 3.0)), ball=(0.0, 0.0))
    assert isinstance(out, IntentDribble)
    assert out.target_pos == pytest.approx((0.0, 2.8))


def test_dribble_target_on_goal_line_pulled_inside():
    out = _apply(IntentDribble(target_pos=(4.5, 0.0)), ball=(0.0, 0.0))
    assert out.target_pos == pytest.approx((4.3, 0.0))


def test_ball_on_verge_of_sideline_is_steered_back_in():
    # Ball already past the safe interior (y=2.9 > 2.8) and being dribbled along
    # the line → aim ball_save_margin (0.30) back inside, not just to the limit.
    out = _apply(IntentDribble(target_pos=(0.0, 2.9)), ball=(0.0, 2.9))
    assert out.target_pos == pytest.approx((0.0, 2.5))


def test_ball_on_verge_of_goal_line_is_steered_back_in():
    out = _apply(IntentDribble(target_pos=(4.5, 0.0)), ball=(4.4, 0.0))
    assert out.target_pos == pytest.approx((4.0, 0.0))


def test_interior_dribble_target_unchanged():
    intent = IntentDribble(target_pos=(2.0, 1.0))
    out = _apply(intent, ball=(2.0, 1.0))
    assert out is intent  # untouched (no rewrite)


def test_move_intent_not_affected_by_ball_keep():
    # keep_ball_in_bounds only guards carried (dribbled) balls, not plain moves.
    intent = IntentMove(target_pos=(4.5, 3.0), target_orientation=None)
    out = _apply(intent, ball=(0.0, 0.0))
    assert out is intent


def test_disabled_leaves_dribble_untouched():
    cfg = MovementSafetyConfig(
        keep_robots_in_bounds=False,
        keep_goalie_in_goal_box=False,
        keep_non_goalies_out_of_goalie_box=False,
        avoid_ball_touch_in_opponent_defense_area=False,
        keep_ball_in_bounds=False,
    )
    intent = IntentDribble(target_pos=(4.5, 3.0))
    out = _apply(intent, ball=(4.4, 2.9), cfg=cfg)
    assert out is intent


def test_default_config_has_ball_keep_off():
    # Backwards compatible: off unless explicitly enabled (e.g. in match yaml).
    assert MovementSafetyConfig().keep_ball_in_bounds is False
