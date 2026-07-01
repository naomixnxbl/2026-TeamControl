"""Predictive goalkeeper tests — DoBallTrajectory extrapolates ball velocity to
the goal line and stands at the crossing point, rather than tracking current y.

Complements test_goalie_tree.py (which uses ball_velocity=(0,0) and pins the
reactive track-y behaviour). Here the ball has a real velocity, enabled by the
adapter's tracked-velocity feed.
"""
from __future__ import annotations

import math

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.trees.goalie import GoalieTree


def _snap(ball_pos, ball_vel, goalie_pos=(-3.9, 0.0)) -> Snapshot:
    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        own_robots=[RobotState(robot_id=0, position=goalie_pos, orientation=0.0)],
        enemy_robots=[],
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _tick(snap, us_positive=False):
    tree = GoalieTree(us_positive=us_positive)
    bb = RobotBlackboard(robot_id=0, current_role=RoleType.GOALIE)
    tree.set_snapshot(snap)
    tree.tick(bb)
    return tree, bb


def test_goalie_moves_to_predicted_crossing_not_current_y():
    # Ball at y=+1.0 struck toward the -x goal and downward → it will cross low.
    # A reactive keeper would go to y=+1.0; the predictive one goes to the
    # crossing point y = 1.0 + (-2.0)*t, t = (-4 - 0)/(-4) = 1.0 → y = -1.0.
    tree, bb = _tick(_snap(ball_pos=(0.0, 1.0), ball_vel=(-4.0, -2.0)))
    assert tree._is_shot_incoming is True
    assert tree.predicted_intercept == (-4.0, -1.0)
    assert isinstance(bb.current_intent, IntentMove)
    assert bb.current_intent.target_pos == (-4.0, -1.0)
    assert bb.intent_source == "GoalieInterceptShot"


def test_prediction_clamps_to_goal_mouth():
    # Steep cross → predicted y would be off the mouth; clamp to [-1, 1].
    tree, _ = _tick(_snap(ball_pos=(0.0, 0.0), ball_vel=(-4.0, -8.0)))
    assert tree._is_shot_incoming is True
    assert tree.predicted_intercept == (-4.0, -1.0)


def test_goalie_tracks_y_when_ball_receding():
    # Ball moving AWAY from our goal (+x) → no shot, track current y.
    tree, bb = _tick(_snap(ball_pos=(0.0, 0.5), ball_vel=(4.0, 0.0)))
    assert tree._is_shot_incoming is False
    assert tree.predicted_intercept == (-4.0, 0.5)
    assert bb.intent_source == "GoaliePosition"


def test_goalie_tracks_y_when_ball_slow():
    # Ball drifting toward goal but below SHOT_SPEED_MIN → track y, don't predict.
    tree, _ = _tick(_snap(ball_pos=(0.0, 0.5), ball_vel=(-0.1, 0.0)))
    assert tree._is_shot_incoming is False
    assert tree.predicted_intercept == (-4.0, 0.5)


def test_goalie_tracks_y_when_shot_beyond_horizon():
    # Slow-ish shot still far away (t = 7 s > horizon) → track y until it nears.
    tree, _ = _tick(_snap(ball_pos=(3.0, 0.5), ball_vel=(-1.0, 0.0)))
    assert tree._is_shot_incoming is False
    assert tree.predicted_intercept == (-4.0, 0.5)


def test_prediction_follows_goal_side_when_us_positive():
    # Own goal at +4.0; ball struck toward +x and up → crosses high.
    tree, bb = _tick(
        _snap(ball_pos=(0.0, 0.5), ball_vel=(4.0, 1.0), goalie_pos=(3.9, 0.0)),
        us_positive=True,
    )
    assert tree._is_shot_incoming is True
    assert tree.predicted_intercept == (4.0, 1.0)  # 0.5 + 1.0*1.0 = 1.5 → clamp 1.0
    assert bb.current_intent.target_pos == (4.0, 1.0)
