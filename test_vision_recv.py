"""Diagnostic: check if vision multicast data is being received from GrSim."""
import socket
import struct
from TeamControl.network.proto2 import ssl_vision_wrapper_pb2

MULTICAST_GROUP = "224.5.23.2"
PORT = 10006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", PORT))
mreq = struct.pack("=4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
sock.settimeout(3)

print(f"Listening for vision on {MULTICAST_GROUP}:{PORT} ...")
print("(If nothing appears within 3 seconds, GrSim multicast is NOT reaching this machine)\n")

cameras_seen = set()
for i in range(20):
    try:
        data, addr = sock.recvfrom(6000)
        packet = ssl_vision_wrapper_pb2.SSL_WrapperPacket()
        packet.ParseFromString(data)
        if packet.HasField("detection"):
            d = packet.detection
            cameras_seen.add(d.camera_id)
            n_yellow = len(d.robots_yellow)
            n_blue = len(d.robots_blue)
            n_balls = len(d.balls)
            print(f"  Frame {d.frame_number} | camera {d.camera_id} | "
                  f"yellow={n_yellow} blue={n_blue} balls={n_balls}")
        elif packet.HasField("geometry"):
            print(f"  [geometry packet received]")
    except socket.timeout:
        print("TIMEOUT - no vision data received. Check GrSim vision multicast address is 224.5.23.2")
        break

if cameras_seen:
    print(f"\nCameras seen: {sorted(cameras_seen)} (total: {len(cameras_seen)})")
    print(f"Code expects: 4 cameras (GRSIM_CAMERAS = 4)")
    if len(cameras_seen) < 4:
        print(f"*** PROBLEM: only {len(cameras_seen)} camera(s) but code waits for 4 -- frames never complete!")
else:
    print("\nNo detection data received at all.")

sock.close()
