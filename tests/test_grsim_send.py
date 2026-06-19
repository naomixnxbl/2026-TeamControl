"""Manual integration check for sending a command to GrSim."""

import os
import time

import pytest

from TeamControl.network.robot_command import RobotCommand
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.utils.yaml_config import Config


pytestmark = pytest.mark.manual


@pytest.mark.skipif(
    os.getenv("TEAMCONTROL_RUN_GRSIM_TESTS") != "1",
    reason="Set TEAMCONTROL_RUN_GRSIM_TESTS=1 to send live GrSim commands.",
)
def test_yellow_robot_zero_spin_command_can_be_sent_to_grsim():
    preset = Config()
    ip, port = preset.grSim_addr
    sender = grSimSender(ip=ip, port=port)

    for _ in range(10):
        cmd = RobotCommand(
            robot_id=0,
            vx=0.5,
            vy=0.0,
            w=1.0,
            kick=0,
            dribble=0,
            isYellow=True,
        )
        sender.send_robot_command(cmd, override_id=0)
        time.sleep(0.02)

    assert sender is not None
