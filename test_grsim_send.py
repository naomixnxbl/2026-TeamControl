"""Quick test: send a move command directly to GrSim to verify connectivity.
Robot 0 (yellow) should spin in place if GrSim receives the packet.
"""
import time
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.utils.yaml_config import Config

preset = Config()
ip, port = preset.grSim_addr
print(f"Sending to GrSim at {ip}:{port}")

sender = grSimSender(ip=ip, port=port)

# Send a spin command to robot 0 (yellow) for 3 seconds
for i in range(150):
    cmd = RobotCommand(robot_id=0, vx=0.5, vy=0.0, w=1.0, kick=0, dribble=0, isYellow=True)
    sender.send_robot_command(cmd, override_id=0)
    time.sleep(0.02)
    if i % 50 == 0:
        print(f"Sent {i} packets...")

print("Done. Did yellow robot 0 move in GrSim?")
