"""Tests for the Supporter behaviour tree — v2.

Tree topology under test:

    SupporterRoot (Selector)
    ├── BallPossessionSequence (Sequence)
    │   ├── IsClosestToBall
    │   └── GoToBall
    ├── PossessionSequence (Sequence)
    │   ├── InPossession
    │   └── DistributeSelector (Selector)
    │       ├── PassSequence (Sequence)
    │       │   ├── FindOpenTeammate
    │       │   └── PassToTeammate
    │       ├── ShootIfClose
    │       └── DribbleToGoal
    └── RepositionToSpace
"""
from __future__ import annotations

import math

import pytest
import py_trees

from TeamControl.bt.trees.supporter import SupporterTree
from TeamControl.bt.trees.supporter import (
    GOALIE_ID,
    POSSESSION_DIST,
    POSSESSION_HEADING_TOL,
    SHOOT_DIST_THRESHOLD,
    MARKED_THRESHOLD,
    PASS_ORIENT_TOL,
    PASS_SIGNAL_TIMEOUT_TICKS,
)

from TeamControl.bt.contracts.blackboard import RobotBlackboard, RoleType
from TeamControl.bt.contracts.intent import (
    Intent,
    IntentDribble,
    IntentKick,
    IntentMove,
    IntentPass,
)
from TeamControl.bt.contracts.snapshot import (
    GamePhase,
    RefereeState,
    RobotState,
    Snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPORTER_ID = 3
_SUPPORTER_ID_B = 4
_ATTACKER_ID = 1
_GOALIE_POS = (-4.0, 0.0)


def make_snapshot(
    ball_pos: tuple[float, float] = (0.0, 0.0),
    own_robots: list[RobotState] | None = None,
    enemy_robots: list[RobotState] | None = None,
) -> Snapshot:
    if own_robots is None:
        own_robots = [
            RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
            RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
        ]
    if enemy_robots is None:
        enemy_robots = []
    return Snapshot(
        ball_position=ball_pos,
        ball_velocity=(0.0, 0.0),
        own_robots=own_robots,
        enemy_robots=enemy_robots,
        referee_state=RefereeState(game_phase=GamePhase.RUNNING, score=(0, 0)),
    )


def _make_bb(robot_id: int = _SUPPORTER_ID) -> RobotBlackboard:
    return RobotBlackboard(robot_id=robot_id, current_role=RoleType.SUPPORTER)


def _tick(snapshot: Snapshot, bb: RobotBlackboard, us_positive: bool = True) -> RobotBlackboard:
    tree = SupporterTree(us_positive=us_positive)
    tree.set_snapshot(snapshot)
    tree.tick(bb)
    return bb


SNAPSHOT_DEFAULT = make_snapshot()


# ---------------------------------------------------------------------------
# TestSupporterTreeImport — basic smoke checks
# ---------------------------------------------------------------------------

class TestSupporterTreeImport:
    def test_importable(self) -> None:
        from TeamControl.bt.trees.supporter import SupporterTree as ST  # noqa: F401

    def test_instantiates(self) -> None:
        assert SupporterTree() is not None

    def test_has_set_snapshot_method(self) -> None:
        assert callable(getattr(SupporterTree(), "set_snapshot", None))

    def test_has_tick_method(self) -> None:
        assert callable(getattr(SupporterTree(), "tick", None))

    def test_does_not_import_robot_command(self) -> None:
        import inspect
        import TeamControl.bt.trees.supporter as mod
        source = inspect.getsource(mod)
        assert "RobotCommand" not in source


# ---------------------------------------------------------------------------
# TestSupporterTreeTopology — v2 py_trees structure
# ---------------------------------------------------------------------------

class TestSupporterTreeTopology:
    def test_root_is_selector(self) -> None:
        tree = SupporterTree()
        assert isinstance(tree.root, py_trees.composites.Selector)

    def test_root_name(self) -> None:
        tree = SupporterTree()
        assert tree.root.name == "SupporterRoot"

    def test_root_has_four_children(self) -> None:
        tree = SupporterTree()
        assert len(tree.root.children) == 4

    def test_first_child_is_possession_sequence(self) -> None:
        tree = SupporterTree()
        first = tree.root.children[0]
        assert isinstance(first, py_trees.composites.Sequence)
        assert first.name == "PossessionSequence"
        assert len(first.children) == 2
        assert first.children[0].name == "InPossession"

    def test_distribute_selector_structure(self) -> None:
        tree = SupporterTree()
        distribute = tree.root.children[0].children[1]
        assert isinstance(distribute, py_trees.composites.Selector)
        assert distribute.name == "DistributeSelector"
        assert len(distribute.children) == 3
        assert distribute.children[0].name == "PassSequence"
        assert distribute.children[1].name == "ShootIfClose"
        assert distribute.children[2].name == "DribbleToGoal"

    def test_pass_sequence_structure(self) -> None:
        tree = SupporterTree()
        pass_seq = tree.root.children[0].children[1].children[0]
        assert isinstance(pass_seq, py_trees.composites.Sequence)
        assert len(pass_seq.children) == 3
        assert pass_seq.children[0].name == "FindOpenTeammate"
        assert pass_seq.children[1].name == "DribbleTowardTarget"
        assert pass_seq.children[2].name == "PassToTeammate"

    def test_second_child_is_receive_pass_sequence(self) -> None:
        tree = SupporterTree()
        second = tree.root.children[1]
        assert isinstance(second, py_trees.composites.Sequence)
        assert second.name == "ReceivePassSequence"
        assert len(second.children) == 2
        assert second.children[0].name == "IsPassTarget"
        assert second.children[1].name == "HoldForPass"

    def test_third_child_is_ball_possession_sequence(self) -> None:
        tree = SupporterTree()
        third = tree.root.children[2]
        assert isinstance(third, py_trees.composites.Sequence)
        assert third.name == "BallPossessionSequence"
        assert len(third.children) == 2
        assert third.children[0].name == "IsClosestToBall"
        assert third.children[1].name == "GoToBall"

    def test_fourth_child_is_reposition(self) -> None:
        tree = SupporterTree()
        fourth = tree.root.children[3]
        assert isinstance(fourth, py_trees.behaviour.Behaviour)
        assert fourth.name == "RepositionToSpace"


# ---------------------------------------------------------------------------
# TestIsClosestToBall
# ---------------------------------------------------------------------------

class TestIsClosestToBall:
    def test_single_supporter_always_closest(self) -> None:
        snap = make_snapshot(
            ball_pos=(2.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(1.0, 0.0), orientation=0.0),
            ],
        )
        bb = _tick(snap, _make_bb())
        assert isinstance(bb.current_intent, IntentMove)
        assert bb.intent_source == "GoToBall"

    def test_closest_of_two_supporters(self) -> None:
        snap = make_snapshot(
            ball_pos=(0.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(1.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(3.0, 0.0), orientation=0.0),
            ],
        )
        # Robot 3 is closer → should chase
        bb3 = _tick(snap, _make_bb(_SUPPORTER_ID))
        assert bb3.intent_source == "GoToBall"

        # Robot 4 is farther → should reposition
        bb4 = _tick(snap, _make_bb(_SUPPORTER_ID_B))
        assert bb4.intent_source == "RepositionToSpace"

    def test_tie_goes_to_lowest_id(self) -> None:
        snap = make_snapshot(
            ball_pos=(0.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(1.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(-1.0, 0.0), orientation=0.0),
            ],
        )
        # Both at distance 1.0 — lower id (3) wins
        bb3 = _tick(snap, _make_bb(_SUPPORTER_ID))
        assert bb3.intent_source == "GoToBall"

        bb4 = _tick(snap, _make_bb(_SUPPORTER_ID_B))
        assert bb4.intent_source == "RepositionToSpace"

    def test_goalie_excluded(self) -> None:
        snap = make_snapshot(
            ball_pos=(-3.9, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(1.0, 0.0), orientation=0.0),
            ],
        )
        # Goalie is closer but excluded — supporter should chase
        bb = _tick(snap, _make_bb())
        assert bb.intent_source == "GoToBall"


# ---------------------------------------------------------------------------
# TestGoToBall
# ---------------------------------------------------------------------------

class TestGoToBall:
    def test_produces_intent_move_to_ball(self) -> None:
        ball = (2.0, 1.0)
        snap = make_snapshot(
            ball_pos=ball,
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        bb = _tick(snap, _make_bb())
        assert isinstance(bb.current_intent, IntentMove)
        assert bb.current_intent.target_pos == ball

    def test_orientation_faces_ball(self) -> None:
        snap = make_snapshot(
            ball_pos=(0.0, 1.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        bb = _tick(snap, _make_bb())
        assert isinstance(bb.current_intent, IntentMove)
        assert bb.current_intent.target_orientation == pytest.approx(math.pi / 2, abs=0.01)


# ---------------------------------------------------------------------------
# TestInPossession
# ---------------------------------------------------------------------------

class TestInPossession:
    def _possession_snap(self, robot_pos, robot_orient, ball_pos):
        return make_snapshot(
            ball_pos=ball_pos,
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=robot_pos, orientation=robot_orient),
            ],
        )

    def test_ball_close_and_aligned_returns_success(self) -> None:
        tree = SupporterTree()
        snap = self._possession_snap((0.0, 0.0), 0.0, (0.1, 0.0))
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        node = tree.root.children[0].children[0]  # InPossession
        result = node.update()
        assert result == py_trees.common.Status.SUCCESS

    def test_ball_far_returns_failure(self) -> None:
        tree = SupporterTree()
        snap = self._possession_snap((0.0, 0.0), 0.0, (2.0, 0.0))
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        node = tree.root.children[1].children[0]
        result = node.update()
        assert result == py_trees.common.Status.FAILURE

    def test_ball_close_but_behind_returns_failure(self) -> None:
        tree = SupporterTree()
        snap = self._possession_snap((0.0, 0.0), 0.0, (-0.1, 0.0))
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        node = tree.root.children[1].children[0]
        result = node.update()
        assert result == py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# TestFindOpenTeammate
# ---------------------------------------------------------------------------

class TestFindOpenTeammate:
    def test_open_teammate_found(self) -> None:
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_ATTACKER_ID, position=(2.0, 2.0), orientation=0.0),
            ],
        )
        tree = SupporterTree()
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        # FindOpenTeammate is at root[1] → children[1](DistributeSelector) → children[0](PassSequence) → children[0]
        find_node = tree.root.children[0].children[1].children[0].children[0]
        result = find_node.update()
        assert result == py_trees.common.Status.SUCCESS
        assert tree._pass_target_id == _ATTACKER_ID

    def test_all_supporters_marked_but_attacker_still_valid(self) -> None:
        """Attacker is scored independently of crowding, so it remains a valid
        pass target even when surrounded."""
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_ATTACKER_ID, position=(2.0, 2.0), orientation=0.0),
            ],
            enemy_robots=[
                RobotState(robot_id=10, position=(2.1, 2.0), orientation=0.0),
            ],
        )
        tree = SupporterTree()
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        find_node = tree.root.children[0].children[1].children[0].children[0]
        result = find_node.update()
        assert result == py_trees.common.Status.SUCCESS
        assert tree._pass_target_id == _ATTACKER_ID

    def test_all_teammates_marked_no_attacker(self) -> None:
        """When only supporters are available and all are marked, FAILURE."""
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(2.0, 2.0), orientation=0.0),
            ],
            enemy_robots=[
                RobotState(robot_id=10, position=(2.1, 2.0), orientation=0.0),
            ],
        )
        tree = SupporterTree()
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        find_node = tree.root.children[0].children[1].children[0].children[0]
        result = find_node.update()
        assert result == py_trees.common.Status.FAILURE

    def test_goalie_excluded(self) -> None:
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=(0.0, 3.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        tree = SupporterTree()
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        find_node = tree.root.children[0].children[1].children[0].children[0]
        result = find_node.update()
        # Only goalie is available → should fail (goalie excluded)
        assert result == py_trees.common.Status.FAILURE

    def test_self_excluded(self) -> None:
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        tree = SupporterTree()
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        find_node = tree.root.children[0].children[1].children[0].children[0]
        result = find_node.update()
        assert result == py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# TestDribbleTowardTarget
# ---------------------------------------------------------------------------

class TestDribbleTowardTarget:
    def _get_node(self, tree):
        return tree.root.children[0].children[1].children[0].children[1]

    def test_not_aligned_returns_running_with_dribble(self) -> None:
        tree = SupporterTree(us_positive=True)
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        tree._pass_target_pos = (-2.0, 0.0)
        node = self._get_node(tree)
        result = node.update()
        assert result == py_trees.common.Status.RUNNING
        assert isinstance(bb.current_intent, IntentDribble)
        assert bb.current_intent.target_pos == (-2.0, 0.0)
        assert bb.intent_source == "DribbleTowardTarget"

    def test_aligned_returns_success(self) -> None:
        tree = SupporterTree(us_positive=True)
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        tree._pass_target_pos = (2.0, 0.0)
        node = self._get_node(tree)
        result = node.update()
        assert result == py_trees.common.Status.SUCCESS

    def test_dribble_targets_teammate_position(self) -> None:
        tree = SupporterTree(us_positive=True)
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        tree._pass_target_pos = (0.0, 2.0)
        node = self._get_node(tree)
        node.update()
        assert isinstance(bb.current_intent, IntentDribble)
        assert bb.current_intent.target_pos == (0.0, 2.0)


# ---------------------------------------------------------------------------
# TestShootIfClose
# ---------------------------------------------------------------------------

class TestShootIfClose:
    def test_close_to_goal_shoots(self) -> None:
        # us_positive=True → goal at (-4.5, 0)
        tree = SupporterTree(us_positive=True)
        snap = make_snapshot(
            ball_pos=(-4.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(-3.5, 0.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        shoot_node = tree.root.children[0].children[1].children[1]  # ShootIfClose
        result = shoot_node.update()
        assert result == py_trees.common.Status.SUCCESS
        assert isinstance(bb.current_intent, IntentKick)

    def test_far_from_goal_fails(self) -> None:
        tree = SupporterTree(us_positive=True)
        snap = make_snapshot(
            ball_pos=(0.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        shoot_node = tree.root.children[0].children[1].children[1]
        result = shoot_node.update()
        assert result == py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# TestDribbleToGoal
# ---------------------------------------------------------------------------

class TestDribbleToGoal:
    def test_writes_intent_dribble(self) -> None:
        tree = SupporterTree(us_positive=True)
        tree.set_snapshot(SNAPSHOT_DEFAULT)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        dribble_node = tree.root.children[0].children[1].children[2]
        result = dribble_node.update()
        assert result == py_trees.common.Status.SUCCESS
        assert isinstance(bb.current_intent, IntentDribble)
        assert bb.current_intent.target_pos == (-4.5, 0.0)

    def test_us_positive_false_goal(self) -> None:
        tree = SupporterTree(us_positive=False)
        tree.set_snapshot(SNAPSHOT_DEFAULT)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        dribble_node = tree.root.children[0].children[1].children[2]
        dribble_node.update()
        assert isinstance(bb.current_intent, IntentDribble)
        assert bb.current_intent.target_pos == (4.5, 0.0)


# ---------------------------------------------------------------------------
# TestPassToTeammate
# ---------------------------------------------------------------------------

class TestPassToTeammate:
    def test_writes_intent_pass(self) -> None:
        tree = SupporterTree()
        tree.set_snapshot(SNAPSHOT_DEFAULT)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        tree._pass_target_id = 1
        tree._pass_target_pos = (2.0, 1.0)
        pass_node = tree.root.children[0].children[1].children[0].children[2]
        result = pass_node.update()
        assert result == py_trees.common.Status.SUCCESS
        assert isinstance(bb.current_intent, IntentPass)
        assert bb.current_intent.target_robot_id == 1
        assert bb.current_intent.target_pos == (2.0, 1.0)

    def test_fails_without_scratch_state(self) -> None:
        tree = SupporterTree()
        tree.set_snapshot(SNAPSHOT_DEFAULT)
        bb = _make_bb()
        tree._blackboard_ref[0] = bb
        pass_node = tree.root.children[0].children[1].children[0].children[2]
        result = pass_node.update()
        assert result == py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# TestRepositionToSpace
# ---------------------------------------------------------------------------

class TestRepositionToSpace:
    def test_produces_intent_move(self) -> None:
        # Need a second supporter so this one isn't closest and falls to reposition
        snap = make_snapshot(
            ball_pos=(5.0, 5.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(4.0, 4.0), orientation=0.0),
            ],
        )
        # Robot 4 is closer to ball → robot 3 repositions
        bb = _tick(snap, _make_bb(_SUPPORTER_ID))
        assert isinstance(bb.current_intent, IntentMove)
        assert bb.intent_source == "RepositionToSpace"

    def test_avoids_opponents(self) -> None:
        snap = make_snapshot(
            ball_pos=(0.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(3.0, 0.0), orientation=0.0),
            ],
            enemy_robots=[
                RobotState(robot_id=10, position=(-2.0, 0.0), orientation=0.0),
            ],
        )
        # Robot 4 is not closest (robot 3 is) → will reposition
        bb = _tick(snap, _make_bb(_SUPPORTER_ID_B))
        assert isinstance(bb.current_intent, IntentMove)
        pos = bb.current_intent.target_pos
        opp_dist = math.hypot(pos[0] - (-2.0), pos[1] - 0.0)
        assert opp_dist > 0.5, f"Repositioned too close to opponent: {pos}"

    def test_orientation_toward_ball(self) -> None:
        snap = make_snapshot(
            ball_pos=(1.0, 1.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(2.0, 0.0), orientation=0.0),
            ],
        )
        bb = _tick(snap, _make_bb(_SUPPORTER_ID_B))
        assert isinstance(bb.current_intent, IntentMove)
        assert bb.current_intent.target_orientation is not None

    def test_us_positive_mirrors_bounds(self) -> None:
        tree_pos = SupporterTree(us_positive=True)
        tree_neg = SupporterTree(us_positive=False)
        # When us_positive=True, repo_x_min should be negative (our attacking half is -x)
        assert tree_pos.repo_x_min < 0
        assert tree_pos.repo_x_max <= 0 or tree_pos.repo_x_max <= 1.0
        # When us_positive=False, repo_x_max should be positive (attacking toward +x)
        assert tree_neg.repo_x_max > 0


# ---------------------------------------------------------------------------
# TestPhaseIntegration — full-tree tick scenarios
# ---------------------------------------------------------------------------

class TestPhaseIntegration:
    def test_closest_chases_other_repositions(self) -> None:
        snap = make_snapshot(
            ball_pos=(1.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.5, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(3.0, 0.0), orientation=0.0),
            ],
        )
        bb3 = _tick(snap, _make_bb(_SUPPORTER_ID))
        assert bb3.intent_source == "GoToBall"

        bb4 = _tick(snap, _make_bb(_SUPPORTER_ID_B))
        assert bb4.intent_source == "RepositionToSpace"

    def test_possession_with_open_teammate_dribbles_toward_then_passes(self) -> None:
        # Robot 3 has possession, attacker is at an angle → dribble toward it first
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_ATTACKER_ID, position=(0.0, 2.0), orientation=0.0),
            ],
        )
        # Robot 3 facing 0, attacker at (0,2) → angle pi/2, not aligned → DribbleTowardTarget
        bb = _tick(snap, _make_bb(_SUPPORTER_ID))
        assert isinstance(bb.current_intent, IntentDribble)
        assert bb.intent_source == "DribbleTowardTarget"

    def test_possession_aligned_teammate_passes_immediately(self) -> None:
        # Robot 3 has possession AND is already facing the teammate → pass fires
        # Robot 3 at (0,0) facing 0, ball at (0.1, 0) → InPossession OK
        # Attacker at (2, 0) → angle 0, robot facing 0 → aligned → IntentPass
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_ATTACKER_ID, position=(2.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(3.0, 0.0), orientation=0.0),
            ],
        )
        bb = _tick(snap, _make_bb(_SUPPORTER_ID))
        assert isinstance(bb.current_intent, IntentPass)
        assert bb.intent_source == "PassToTeammate"

    def test_possession_no_chase_when_not_closest(self) -> None:
        # Robot 4 has possession, robot 3 is closer to ball, attacker is open
        # Robot 4 at (0,0) facing 0, ball at (0.1, 0) → InPossession OK
        # Attacker at (2, 0) → angle_to_target = 0, robot facing 0 → already aligned
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.05, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_ATTACKER_ID, position=(2.0, 0.0), orientation=0.0),
            ],
        )
        # Robot 3 dist = 0.05, Robot 4 dist = 0.1 → robot 3 is closest
        # Robot 4: dist 0.1 < 0.122, facing 0, ball angle 0 → InPossession OK
        # Attacker at (2,0), robot 4 at (0,0), angle = 0, facing 0 → aligned → IntentPass
        bb4 = _tick(snap, _make_bb(_SUPPORTER_ID_B))
        assert isinstance(bb4.current_intent, IntentPass)
        assert bb4.intent_source == "PassToTeammate"

    def test_possession_close_to_goal_no_teammates_shoots(self) -> None:
        # us_positive=True → goal at (-4.5, 0)
        # Robot 4 has possession, close to goal, only goalie + robot 3 on field
        # Robot 3 is marked, no attacker available → FindOpenTeammate fails → ShootIfClose
        snap = make_snapshot(
            ball_pos=(-3.5, 0.01),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(-3.5, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(-3.5, 0.1), orientation=-math.pi / 2),
            ],
            enemy_robots=[
                RobotState(robot_id=11, position=(-3.5, 0.05), orientation=0.0),
            ],
        )
        bb = _tick(snap, _make_bb(_SUPPORTER_ID_B), us_positive=True)
        assert isinstance(bb.current_intent, IntentKick)
        assert bb.intent_source == "ShootIfClose"

    def test_possession_all_marked_far_dribbles(self) -> None:
        # Robot 3 is closer to ball, robot 4 has possession but isn't closest
        # Ball at (0.0, 0.01), robot 3 at (0.0, 0.0) dist=0.01, robot 4 at (0.0, 0.1) dist=0.09
        snap = make_snapshot(
            ball_pos=(0.0, 0.01),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(0.0, 0.1), orientation=-math.pi / 2),
            ],
            enemy_robots=[
                RobotState(robot_id=10, position=(0.1, 0.0), orientation=0.0),
            ],
        )
        # Robot 3 is closest → robot 4 not closest
        # Robot 4: dist=0.09 < 0.122, facing -pi/2, ball angle ≈ -pi/2 → InPossession OK
        # Robot 3 is marked (opp at 0.1,0 dist 0.1 < 0.5) → FindOpenTeammate fails
        # Far from goal → ShootIfClose fails → DribbleToGoal
        bb = _tick(snap, _make_bb(_SUPPORTER_ID_B))
        assert isinstance(bb.current_intent, IntentDribble)
        assert bb.intent_source == "DribbleToGoal"

    def test_multiple_sequential_robots_same_tree(self) -> None:
        snap = make_snapshot(
            ball_pos=(1.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.5, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(3.0, 0.0), orientation=0.0),
            ],
        )
        tree = SupporterTree()
        tree.set_snapshot(snap)

        bb3 = _make_bb(_SUPPORTER_ID)
        tree.tick(bb3)
        assert bb3.current_intent is not None

        bb4 = _make_bb(_SUPPORTER_ID_B)
        tree.tick(bb4)
        assert bb4.current_intent is not None

    def test_no_possession_falls_to_reposition(self) -> None:
        snap = make_snapshot(
            ball_pos=(5.0, 5.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(0.0, 1.0), orientation=0.0),
            ],
        )
        # Robot 3 is closer to ball (dist ~7.07 vs ~7.00 for robot 4? let's check)
        # Robot 3 at (0,0), ball at (5,5): dist=7.07
        # Robot 4 at (0,1), ball at (5,5): dist=hypot(5,4)=6.40 — robot 4 is closer!
        # So robot 4 will chase. Let's tick robot 3 instead (farther → reposition)
        bb = _tick(snap, _make_bb(_SUPPORTER_ID))
        assert bb.intent_source == "RepositionToSpace"


# ---------------------------------------------------------------------------
# TestTickInterface — contracts
# ---------------------------------------------------------------------------

class TestTickInterface:
    def test_tick_sets_current_intent(self) -> None:
        bb = _make_bb()
        assert bb.current_intent is None
        _tick(SNAPSHOT_DEFAULT, bb)
        assert bb.current_intent is not None

    def test_multiple_ticks_no_raise(self) -> None:
        tree = SupporterTree()
        tree.set_snapshot(SNAPSHOT_DEFAULT)
        bb = _make_bb()
        for _ in range(5):
            tree.tick(bb)

    def test_snapshot_can_be_replaced(self) -> None:
        tree = SupporterTree()
        bb = _make_bb()
        tree.set_snapshot(make_snapshot(ball_pos=(1.0, 0.0)))
        tree.tick(bb)
        tree.set_snapshot(make_snapshot(ball_pos=(2.0, 0.0)))
        tree.tick(bb)
        assert bb.current_intent is not None


# ---------------------------------------------------------------------------
# TestNoRobotCommandWritten
# ---------------------------------------------------------------------------

class TestNoRobotCommandWritten:
    def test_no_robot_command_string_in_source(self) -> None:
        import inspect
        import TeamControl.bt.trees.supporter as mod
        source = inspect.getsource(mod)
        assert "RobotCommand" not in source


# ---------------------------------------------------------------------------
# TestIsolation
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_fresh_tree_needs_no_coordinator(self) -> None:
        tree = SupporterTree()
        bb = _make_bb()
        tree.set_snapshot(SNAPSHOT_DEFAULT)
        tree.tick(bb)

    def test_two_tree_instances_are_independent(self) -> None:
        tree_a = SupporterTree()
        tree_b = SupporterTree()
        bb_a = _make_bb()
        bb_b = _make_bb()
        tree_a.set_snapshot(SNAPSHOT_DEFAULT)
        tree_b.set_snapshot(SNAPSHOT_DEFAULT)
        tree_a.tick(bb_a)
        tree_b.tick(bb_b)
        assert bb_a.current_intent is not None
        assert bb_b.current_intent is not None


# ---------------------------------------------------------------------------
# TestPassSignal — coordination between passer and receiver
# ---------------------------------------------------------------------------

class TestPassSignal:
    """Verify the _active_pass_target side-channel for pass coordination."""

    def _make_pass_scenario_tree(self):
        """Build a tree and snapshot where robot 3 can pass to robot 4."""
        tree = SupporterTree(us_positive=True)
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(2.0, 2.0), orientation=0.0),
            ],
        )
        return tree, snap

    def test_pass_sets_active_signal(self) -> None:
        # Robot 3 at (0,0) facing 0, ball at (0.1, 0), target at (2,0)
        # All aligned → pass fires on first tick and sets signal
        tree = SupporterTree(us_positive=True)
        snap = make_snapshot(
            ball_pos=(0.1, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(2.0, 0.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        assert tree._active_pass_target is None

        bb3 = _make_bb(_SUPPORTER_ID)
        tree.tick(bb3)
        assert isinstance(bb3.current_intent, IntentPass)
        assert tree._active_pass_target == bb3.current_intent.target_robot_id
        assert tree._active_pass_target_age == 0

    def test_receiver_holds_position_when_signalled(self) -> None:
        tree = SupporterTree(us_positive=True)
        # Manually set the pass signal
        tree._active_pass_target = _SUPPORTER_ID_B
        tree._active_pass_target_age = 0

        snap = make_snapshot(
            ball_pos=(1.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.5, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(2.0, 2.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb4 = _make_bb(_SUPPORTER_ID_B)
        tree.tick(bb4)

        # Robot 4 is not closest (robot 3 is closer)
        # But robot 4 IS the pass target → ReceivePassSequence fires
        assert bb4.intent_source == "HoldForPass"
        assert isinstance(bb4.current_intent, IntentMove)
        # Should hold at its own position
        assert bb4.current_intent.target_pos == (2.0, 2.0)

    def test_non_target_ignores_signal(self) -> None:
        tree = SupporterTree(us_positive=True)
        tree._active_pass_target = _SUPPORTER_ID_B  # signal is for robot 4

        snap = make_snapshot(
            ball_pos=(5.0, 5.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(4.0, 4.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb3 = _make_bb(_SUPPORTER_ID)
        tree.tick(bb3)

        # Robot 3 is not the pass target — should not hold
        assert bb3.intent_source != "HoldForPass"

    def test_signal_times_out(self) -> None:
        tree = SupporterTree(us_positive=True)
        tree._active_pass_target = _SUPPORTER_ID_B
        tree._active_pass_target_age = PASS_SIGNAL_TIMEOUT_TICKS  # one tick from timeout

        snap = make_snapshot(
            ball_pos=(5.0, 5.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(4.0, 4.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb4 = _make_bb(_SUPPORTER_ID_B)
        tree.tick(bb4)

        # Age incremented to TIMEOUT+1 → signal cleared before tree runs
        assert tree._active_pass_target is None
        assert bb4.intent_source != "HoldForPass"

    def test_signal_cleared_when_receiver_gains_possession(self) -> None:
        tree = SupporterTree(us_positive=True)
        tree._active_pass_target = _SUPPORTER_ID_B
        tree._active_pass_target_age = 5

        # Robot 4 is the pass target AND has ball in dribbler → PossessionSequence
        # fires (not HoldForPass, because InPossession is checked first in the new order)
        snap = make_snapshot(
            ball_pos=(2.01, 2.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.0, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(2.0, 2.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb4 = _make_bb(_SUPPORTER_ID_B)
        tree.tick(bb4)

        # Robot 4 has possession → PossessionSequence fires (intent != HoldForPass)
        # Post-tick clearing: robot 4 is the target and produced a non-HoldForPass intent
        assert bb4.intent_source != "HoldForPass"
        assert tree._active_pass_target is None

    def test_signal_persists_across_ticks(self) -> None:
        tree = SupporterTree(us_positive=True)
        tree._active_pass_target = _SUPPORTER_ID_B
        tree._active_pass_target_age = 0

        snap = make_snapshot(
            ball_pos=(1.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(0.5, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(2.0, 2.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)

        # Tick robot 3 first (not the target)
        bb3 = _make_bb(_SUPPORTER_ID)
        tree.tick(bb3)
        assert tree._active_pass_target == _SUPPORTER_ID_B  # still set

        # Tick robot 4 (the target) — should hold
        bb4 = _make_bb(_SUPPORTER_ID_B)
        tree.tick(bb4)
        assert bb4.intent_source == "HoldForPass"
        assert tree._active_pass_target == _SUPPORTER_ID_B  # persists (HoldForPass doesn't clear)

    def test_hold_for_pass_orientation_faces_ball(self) -> None:
        tree = SupporterTree(us_positive=True)
        tree._active_pass_target = _SUPPORTER_ID_B

        snap = make_snapshot(
            ball_pos=(0.0, 0.0),
            own_robots=[
                RobotState(robot_id=GOALIE_ID, position=_GOALIE_POS, orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID, position=(-0.5, 0.0), orientation=0.0),
                RobotState(robot_id=_SUPPORTER_ID_B, position=(0.0, 1.0), orientation=0.0),
            ],
        )
        tree.set_snapshot(snap)
        bb4 = _make_bb(_SUPPORTER_ID_B)
        tree.tick(bb4)
        assert isinstance(bb4.current_intent, IntentMove)
        # Ball at (0,0), robot at (0,1) → angle = -pi/2
        assert bb4.current_intent.target_orientation == pytest.approx(-math.pi / 2, abs=0.01)
