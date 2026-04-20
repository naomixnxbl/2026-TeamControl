import random
from TeamControl.network.robot_command import RobotCommand


def create_input():
    random_id = random.randint(1, 3) # good
    run_time = random.randrange(1, 10) #good
    
    random_vx = random.randint(0,2)

    command = RobotCommand(robot_id=random_id,vx=random_vx) #changed this

    return command,run_time #this creates tuple automatically 


def test_input_gen():
    command,runtime = create_input()
    print(f"{command=}")
    print(f"{runtime=}")
        
def test_input_gen2():
    packet = create_input()
    print(f"{packet=}")

# tests
test_input_gen()
test_input_gen2()