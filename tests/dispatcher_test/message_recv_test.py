"""Integration check for the raw dispatcher receiver."""

import os
from multiprocessing import Event

import pytest

from TeamControl.network.receiver import Receiver


IP = "127.0.0.1"
PORT_NUMBER = 50514

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TEAMCONTROL_RUN_NETWORK_TESTS") != "1",
    reason="Set TEAMCONTROL_RUN_NETWORK_TESTS=1 to wait for live receiver data.",
)
def test_receiver_can_listen_for_one_message():
    is_running = Event()
    is_running.set()
    receiver = Receiver(is_running, ip=IP, port=PORT_NUMBER)

    message, addr = receiver.listen()

    assert message is not None
    assert addr is not None
