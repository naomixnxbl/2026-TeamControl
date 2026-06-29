"""Team-perspective / GC-correspondence tests.

The set-piece states (corner/goal/penalty + the rest) must resolve correctly no
matter which colour we are (yellow or blue) and which half we defend. Colour is
resolved in the GC FSM ("ours vs theirs"); side is resolved per-team in
run_bt_v2_process; the per-team perspective flip lives in the adapter. These
tests pin all three so a colour/side swap can't silently break the states.
"""
from __future__ import annotations

import queue
from multiprocessing import Event

import pytest

from TeamControl.SSL.game_controller.common import Command, GameState, Stage
from TeamControl.bt.adapter import _phase_from_state, _phase_for_perspective
from TeamControl.bt.contracts.snapshot import GamePhase


class _DummyLogger:
    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _make_gcfsm(monkeypatch, us_yellow):
    """Build a GCfsm with its multicast socket stubbed out (no network)."""
    import TeamControl.process_workers.gcfsm_runner as mod

    monkeypatch.setattr(mod, "GameControl", lambda *a, **k: object())
    fsm = mod.GCfsm(Event(), _DummyLogger())
    fsm.us_yellow = us_yellow
    fsm.output_q = queue.Queue()
    fsm.current_state = None
    fsm.current_command = None
    return fsm


# ---------------------------------------------------------------------------
# GC FSM: referee command -> GameState, resolved by our colour (§ referee proto)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("us_yellow", [True, False])
def test_stop_is_colourless(monkeypatch, us_yellow):
    fsm = _make_gcfsm(monkeypatch, us_yellow)
    fsm.update_state(Command.STOP, Stage.NORMAL_FIRST_HALF)
    assert fsm.current_state == GameState.STOPPED


@pytest.mark.parametrize(
    "us_yellow, command, expected",
    [
        # We are YELLOW.
        (True, Command.DIRECT_FREE_YELLOW, GameState.OUR_FREE_KICK),
        (True, Command.DIRECT_FREE_BLUE, GameState.ENEMY_FREE_KICK),
        (True, Command.INDIRECT_FREE_YELLOW, GameState.OUR_FREE_KICK),
        (True, Command.PREPARE_PENALTY_YELLOW, GameState.OUR_PREPARE_PENALTY),
        (True, Command.PREPARE_PENALTY_BLUE, GameState.ENEMY_PREPARE_PENALTY),
        (True, Command.BALL_PLACEMENT_YELLOW, GameState.OUR_BALL_PLACEMENT),
        (True, Command.PREPARE_KICKOFF_YELLOW, GameState.OUR_PREPARE_KICKOFF),
        (True, Command.PREPARE_KICKOFF_BLUE, GameState.ENEMY_PREPARE_KICKOFF),
        # We are BLUE — everything mirrors.
        (False, Command.DIRECT_FREE_BLUE, GameState.OUR_FREE_KICK),
        (False, Command.DIRECT_FREE_YELLOW, GameState.ENEMY_FREE_KICK),
        (False, Command.INDIRECT_FREE_BLUE, GameState.OUR_FREE_KICK),
        (False, Command.PREPARE_PENALTY_BLUE, GameState.OUR_PREPARE_PENALTY),
        (False, Command.PREPARE_PENALTY_YELLOW, GameState.ENEMY_PREPARE_PENALTY),
        (False, Command.BALL_PLACEMENT_BLUE, GameState.OUR_BALL_PLACEMENT),
        (False, Command.PREPARE_KICKOFF_BLUE, GameState.OUR_PREPARE_KICKOFF),
        (False, Command.PREPARE_KICKOFF_YELLOW, GameState.ENEMY_PREPARE_KICKOFF),
    ],
)
def test_command_resolves_by_colour(monkeypatch, us_yellow, command, expected):
    fsm = _make_gcfsm(monkeypatch, us_yellow)
    fsm.update_state(command, Stage.NORMAL_FIRST_HALF)
    assert fsm.current_state == expected


@pytest.mark.parametrize(
    "us_yellow, prep, expected",
    [
        (True, Command.PREPARE_PENALTY_YELLOW, GameState.OUR_PENALTY_SHOOTOUT),
        (True, Command.PREPARE_PENALTY_BLUE, GameState.ENEMY_PENALTY_SHOOTOUT),
        (False, Command.PREPARE_PENALTY_BLUE, GameState.OUR_PENALTY_SHOOTOUT),
        (False, Command.PREPARE_PENALTY_YELLOW, GameState.ENEMY_PENALTY_SHOOTOUT),
    ],
)
def test_normal_start_after_penalty_prepare_becomes_shootout(monkeypatch, us_yellow, prep, expected):
    fsm = _make_gcfsm(monkeypatch, us_yellow)
    fsm.current_command = prep  # the command that preceded NORMAL_START
    fsm.update_state(Command.NORMAL_START, Stage.NORMAL_FIRST_HALF)
    assert fsm.current_state == expected


# ---------------------------------------------------------------------------
# Adapter: GameState -> GamePhase, and the per-team perspective flip
# ---------------------------------------------------------------------------

def test_phase_map_handles_both_team_states():
    assert _phase_from_state(GameState.OUR_FREE_KICK) == GamePhase.FREE_KICK
    assert _phase_from_state(GameState.ENEMY_FREE_KICK) == GamePhase.ENEMY_FREE_KICK
    assert _phase_from_state(GameState.OUR_PENALTY_SHOOTOUT) == GamePhase.PENALTY_SHOOT
    assert _phase_from_state(GameState.ENEMY_PENALTY_SHOOTOUT) == GamePhase.PENALTY_DEFEND
    assert _phase_from_state(GameState.OUR_BALL_PLACEMENT) == GamePhase.BALL_PLACEMENT
    assert _phase_from_state(GameState.ENEMY_BALL_PLACEMENT) == GamePhase.BALL_PLACEMENT


class _FakeWM:
    def __init__(self, state, us_yellow):
        self._s = state
        self._y = us_yellow

    def get_game_state(self):
        return self._s

    def us_yellow(self):
        return self._y


def test_perspective_same_colour_is_direct():
    wm = _FakeWM(GameState.OUR_FREE_KICK, us_yellow=True)
    assert _phase_for_perspective(wm, is_yellow=True) == GamePhase.FREE_KICK


def test_perspective_other_colour_flips_free_kick():
    # The blue view of "yellow's free kick" is an enemy free kick, and vice versa.
    wm = _FakeWM(GameState.OUR_FREE_KICK, us_yellow=True)
    assert _phase_for_perspective(wm, is_yellow=False) == GamePhase.ENEMY_FREE_KICK


def test_perspective_other_colour_flips_penalty():
    wm = _FakeWM(GameState.OUR_PENALTY_SHOOTOUT, us_yellow=True)
    assert _phase_for_perspective(wm, is_yellow=True) == GamePhase.PENALTY_SHOOT
    assert _phase_for_perspective(wm, is_yellow=False) == GamePhase.PENALTY_DEFEND


def test_perspective_blue_as_us():
    # We are blue; our free kick stays a free kick from our own perspective.
    wm = _FakeWM(GameState.OUR_FREE_KICK, us_yellow=False)
    assert _phase_for_perspective(wm, is_yellow=False) == GamePhase.FREE_KICK
    # The yellow view of it is an enemy free kick.
    assert _phase_for_perspective(wm, is_yellow=True) == GamePhase.ENEMY_FREE_KICK
