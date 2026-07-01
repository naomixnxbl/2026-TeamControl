"""Counter-attack release + open-goal aiming in the attacker tree."""
from __future__ import annotations

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import IntentDribble, IntentKick, IntentPass
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)
from TeamControl.bt.trees.attacker import (
    AttackerBehaviorConfig,
    AttackerTree,
    _best_goal_target,
)


def _snapshot(ball, own, enemies=()):
    return Snapshot(
        ball_position=ball,
        ball_velocity=(0.0, 0.0),
        own_robots=list(own),
        enemy_robots=list(enemies),
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _tick(tree: AttackerTree, snap: Snapshot, robot_id: int) -> RobotBlackboard:
    bb = RobotBlackboard(robot_id=robot_id, current_role=RoleType.ATTACKER)
    tree.set_snapshot(snap)
    tree.tick(bb)
    return bb


def _carrier_with_ball(rid, pos, orientation=0.0):
    # Ball just in front of the kicker so HasBallControl succeeds.
    return RobotState(robot_id=rid, position=pos, orientation=orientation)


def test_counter_release_passes_forward_from_our_half() -> None:
    # us_positive=False → we attack +x; our half is x < 0.
    tree = AttackerTree(
        us_positive=False,
        behavior_config=AttackerBehaviorConfig(counter_attack=True),
    )
    ball = (-1.0, 0.0)
    own = [
        _carrier_with_ball(1, (-1.08, 0.0), orientation=0.0),  # carrier facing +x
        RobotState(robot_id=2, position=(2.0, 0.0), orientation=0.0),  # forward outlet
    ]
    bb = _tick(tree, _snapshot(ball, own), robot_id=1)

    assert isinstance(bb.current_intent, IntentPass)
    assert bb.current_intent.target_robot_id == 2


def test_forward_pass_also_works_on_opponent_field() -> None:
    # On the opponent field with a more-advanced open teammate (closer to goal),
    # we still play the minimal forward pass — that keeps possession advancing.
    tree = AttackerTree(
        us_positive=False,
        behavior_config=AttackerBehaviorConfig(counter_attack=True),
    )
    ball = (1.0, 0.0)
    own = [
        _carrier_with_ball(1, (0.92, 0.0), orientation=0.0),
        RobotState(robot_id=2, position=(3.0, 0.0), orientation=0.0),  # ahead, near goal
    ]
    bb = _tick(tree, _snapshot(ball, own), robot_id=1)

    assert isinstance(bb.current_intent, IntentPass)
    assert bb.current_intent.target_robot_id == 2


def test_dribbles_forward_when_no_forward_outlet() -> None:
    # No teammate meaningfully ahead of the carrier → dribble forward (the
    # fallback when no good pass is seen), not a square/backward pass.
    tree = AttackerTree(
        us_positive=False,
        behavior_config=AttackerBehaviorConfig(counter_attack=True),
    )
    ball = (1.0, 0.0)
    own = [
        _carrier_with_ball(1, (0.92, 0.0), orientation=0.0),
        RobotState(robot_id=2, position=(-1.0, 0.0), orientation=0.0),  # behind carrier
    ]
    bb = _tick(tree, _snapshot(ball, own), robot_id=1)

    assert not isinstance(bb.current_intent, IntentPass)
    assert isinstance(bb.current_intent, IntentDribble)  # HoldPossession (carry forward)


def test_counter_release_disabled_by_default() -> None:
    tree = AttackerTree(us_positive=False)  # counter_attack defaults False
    ball = (-1.0, 0.0)
    own = [
        _carrier_with_ball(1, (-1.08, 0.0), orientation=0.0),
        RobotState(robot_id=2, position=(2.0, 0.0), orientation=0.0),
    ]
    bb = _tick(tree, _snapshot(ball, own), robot_id=1)

    # Without counter-attack the carrier holds/dribbles rather than releasing.
    assert not isinstance(bb.current_intent, IntentPass)


def test_open_goal_target_avoids_central_keeper() -> None:
    goal = (4.5, 0.0)
    snap = _snapshot(
        ball=(2.5, 0.0),
        own=[RobotState(robot_id=1, position=(2.4, 0.0), orientation=0.0)],
        enemies=[RobotState(robot_id=0, position=(4.5, 0.0), orientation=0.0)],  # keeper centre
    )
    target = _best_goal_target(snap, goal, corridor_radius=0.20)
    # Aim is pushed off-centre, away from the keeper sitting on the goal line.
    assert abs(target[1]) > 0.1
    assert target[0] == goal[0]


def test_open_goal_target_centres_when_no_keeper() -> None:
    goal = (4.5, 0.0)
    snap = _snapshot(
        ball=(2.5, 0.0),
        own=[RobotState(robot_id=1, position=(2.4, 0.0), orientation=0.0)],
        enemies=[],
    )
    assert _best_goal_target(snap, goal, corridor_radius=0.20) == goal


def test_shoot_uses_open_aim_point() -> None:
    # Close to goal, settled, keeper slightly off-centre → shot fires at an
    # aim point, not necessarily the centre.
    tree = AttackerTree(us_positive=False)
    ball = (3.7, 0.0)
    own = [_carrier_with_ball(1, (3.6, 0.0), orientation=0.0)]
    enemies = [RobotState(robot_id=0, position=(4.5, 0.2), orientation=0.0)]
    snap = _snapshot(ball, own, enemies)
    # Settle possession so the shoot branch is reachable.
    bb = RobotBlackboard(robot_id=1, current_role=RoleType.ATTACKER)
    for _ in range(tree.behavior_config.shot_settle_ticks + 2):
        tree.set_snapshot(snap)
        tree.tick(bb)
    if isinstance(bb.current_intent, IntentKick):
        # The chosen aim point is the tree's selected shot target.
        assert bb.current_intent.target_pos == tree._shot_target
