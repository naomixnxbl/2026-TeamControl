import random

from TeamControl.network.robot_command import RobotCommand


def create_input(rng=random):
    robot_id = rng.randint(1, 3)
    run_time = rng.randrange(1, 10)
    vx = rng.randint(0, 2)

    command = RobotCommand(robot_id=robot_id, vx=vx)

    return command, run_time


def test_input_gen_returns_robot_command_and_runtime():
    rng = random.Random(7)

    command, runtime = create_input(rng)

    assert isinstance(command, RobotCommand)
    assert 1 <= command.robot_id <= 3
    assert 0 <= command.vx <= 2
    assert 1 <= runtime < 10


def test_input_gen_returns_tuple_shape():
    packet = create_input(random.Random(11))

    assert len(packet) == 2
