from multiprocessing import Event, Queue

from TeamControl.dispatcher.dispatch import Dispatcher
from TeamControl.network.robot_command import RobotCommand


class _DummySender:
    def __init__(self, *args, **kwargs):
        self.sent = []

    def send(self, command, ip, port):
        self.sent.append((command, ip, port))

    def send_robot_command(self, command, override_id=None):
        self.sent.append((command, override_id))


class _Config:
    send_to_grSim = True
    grSim_addr = ("127.0.0.1", 20011)
    robot_ip = "127.0.0.1"
    yellow = {
        "A": {"shellID": 0, "grSimID": 4, "ip": "127.0.0.1", "port": 50514}
    }
    blue = {}


def _dispatcher(monkeypatch):
    monkeypatch.setattr("TeamControl.dispatcher.dispatch.Sender", _DummySender)
    monkeypatch.setattr("TeamControl.dispatcher.dispatch.grSimSender", _DummySender)
    d = Dispatcher(Event(), None)
    d.setup(Queue(), _Config(), Queue(), Queue())
    return d


def test_manual_command_is_not_blocked_by_field_override(monkeypatch):
    d = _dispatcher(monkeypatch)
    d._manual_field_blocked.add((0, True))
    cmd = RobotCommand(0, vx=1.0, isYellow=True)

    d.add(cmd, 0.2, source="manual")
    d.handle_commands(now=1.0)

    assert d.g_sender.sent == [(cmd, 4)]
    assert d.r_sender.sent == [(cmd, "127.0.0.1", 50514)]


def test_auto_command_is_blocked_by_field_override(monkeypatch):
    d = _dispatcher(monkeypatch)
    d._manual_field_blocked.add((0, True))
    cmd = RobotCommand(0, vx=1.0, isYellow=True)

    d.add(cmd, 0.2, source="auto")
    d.handle_commands(now=1.0)

    assert d.g_sender.sent == []
    assert d.r_sender.sent == []
