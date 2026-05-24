from TeamControl.network.robot_command import RobotCommand
from TeamControl.network.receiver import Receiver

def robot_listener():
    recv = Receiver(ip='',port=50513)
    while True:
        data, addr = recv.listen()
        print(data)
        # print(f"Received command from {addr}: {data}")
        
robot_listener()