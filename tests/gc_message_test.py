"""Integration check for receiving SSL game-controller messages."""

import os
import time
from multiprocessing import Event

import pytest

from TeamControl.SSL.game_controller.Message import RefereeMessage
from TeamControl.network.ssl_sockets import GameControl


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TEAMCONTROL_RUN_NETWORK_TESTS") != "1",
    reason="Set TEAMCONTROL_RUN_NETWORK_TESTS=1 to run live network tests.",
)
def test_game_controller_message_can_be_received_and_parsed():
    is_running = Event()
    is_running.set()
    gc_recv = GameControl(is_running=is_running)

    ref_msg = gc_recv.listen()
    start_time = time.time()
    parsed = RefereeMessage.from_proto(referee=ref_msg)
    elapsed_ms = (time.time() - start_time) * 1000

    assert parsed is not None
    assert elapsed_ms >= 0.0
