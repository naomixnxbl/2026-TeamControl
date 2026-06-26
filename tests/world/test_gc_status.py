from TeamControl.SSL.game_controller.common import Command, GameState, PacketType, Stage
from TeamControl.world.model import WorldModel
from TeamControl.world.model_manager import WorldModelManager


def test_world_model_updates_gc_status_and_version():
    wm = WorldModel()
    before = wm.get_version()

    wm.update_gc_data(
        (
            PacketType.GC_STATUS,
            {
                "stage": Stage.NORMAL_FIRST_HALF,
                "command": Command.STOP,
                "state": GameState.STOPPED,
                "us_yellow": False,
                "us_positive": True,
                "yellow_cards": 1,
                "red_cards": 0,
                "fouls": 2,
                "yellow_card_times": [120_000_000],
                "packet_timestamp": 123,
                "received_at": 456.0,
            },
        )
    )

    status = wm.get_gc_status()
    assert status["stage"] == Stage.NORMAL_FIRST_HALF
    assert status["command"] == Command.STOP
    assert status["state"] == GameState.STOPPED
    assert status["us_yellow"] is False
    assert status["us_positive"] is True
    assert status["yellow_cards"] == 1
    assert status["red_cards"] == 0
    assert status["fouls"] == 2
    assert status["yellow_card_times"] == [120_000_000]
    assert wm.get_game_state() == GameState.STOPPED
    assert wm.get_version() > before


def test_gc_status_is_available_through_world_model_manager():
    manager = WorldModelManager()
    manager.start()
    try:
        wm = manager.WorldModel()
        wm.update_gc_data(
            (
                PacketType.GC_STATUS,
                {
                    "stage": Stage.NORMAL_FIRST_HALF,
                    "command": Command.FORCE_START,
                    "state": GameState.RUNNING,
                    "us_yellow": True,
                    "us_positive": False,
                    "received_at": 123.0,
                },
            )
        )

        status = wm.get_gc_status()
        assert status["command"] == Command.FORCE_START
        assert status["state"] == GameState.RUNNING
        assert status["us_yellow"] is True
    finally:
        manager.shutdown()
