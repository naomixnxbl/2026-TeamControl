"""Integration diagnostic for receiving SSL vision multicast data."""

import os
import socket
import struct

import pytest

from TeamControl.network.proto2 import ssl_vision_wrapper_pb2


MULTICAST_GROUP = "224.5.23.2"
PORT = 10006

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TEAMCONTROL_RUN_NETWORK_TESTS") != "1",
    reason="Set TEAMCONTROL_RUN_NETWORK_TESTS=1 to listen for live vision packets.",
)
def test_vision_multicast_packet_can_be_received():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", PORT))
        mreq = struct.pack(
            "=4sl",
            socket.inet_aton(MULTICAST_GROUP),
            socket.INADDR_ANY,
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(3)

        data, _addr = sock.recvfrom(6000)
        packet = ssl_vision_wrapper_pb2.SSL_WrapperPacket()
        packet.ParseFromString(data)

        assert packet.HasField("detection") or packet.HasField("geometry")
    finally:
        sock.close()
