"""Integration check for the Vision socket wrapper."""

import os
from multiprocessing import Event

import pytest

from TeamControl.network.ssl_sockets import Vision


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TEAMCONTROL_RUN_NETWORK_TESTS") != "1",
    reason="Set TEAMCONTROL_RUN_NETWORK_TESTS=1 to listen for live vision packets.",
)
def test_vision_initialization_and_receive():
    is_running = Event()
    is_running.set()
    vision = Vision(is_running=is_running)

    assert vision is not None
    assert vision.addr is not None
    assert vision.listen() is not None
