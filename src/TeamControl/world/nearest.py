import math
from TeamControl.SSL.vision.frame import Robot
import logging


class Nearest():

    """
    Function - Find the nearest robot using the surrounding radius
    Author : Gauri , Emma
    """
    @staticmethod
    # Calculate distance from other robots using Euclidean's distance
    def calculate_dist_frm_target(pos_1, pos_2):
        if not (pos_1 is None or pos_2 is None):
            dx = pos_1[0] - pos_2[0]
            dy = pos_1[1] - pos_2[1]
            return math.hypot(dx, dy)

    # Finding nearest robot
    @classmethod
    def robot (cls,target_position, list_robots:list[Robot], radius = float('inf')) -> Robot:
        """
        Getting nearest Robot to Target position

        Args:
            target_position (tuple[float,float]): Targeted position x,y. Usually ball_position.
            list_robots (list[Robot]): list of robots to be compared. Usually our team or both teams.
            radius (float, optional): within a certain radius. Defaults to float('inf'). INFINITY

        Returns:
            Robot: nearest 'Robot' object
        """
        nearest_robot = None
        min_distance = float('inf')
        if list_robots is None:
            return
        elif len(list_robots) == 0:
            return
        for robot in list_robots:
            distance = cls.calculate_dist_frm_target(target_position, robot.get_position())
            if distance is None:
                return
            if distance <= radius and distance < min_distance:
                min_distance = distance
                nearest_robot = robot

        return nearest_robot

    @classmethod
    def robot_ordered(cls,target_position: tuple[float,float],list_robot: list[Robot])->list[Robot]:
        """Nearest Robot Ordered
        returns list sorted by distance to target.

        Args:
            target_position(tuple[float,float]): targeted position, usually ball_position
            list_robot(list[Robot]): list of 'Robot' objects, robots to be compared to targetted position

        Returns:
            list[Robot] : list of Robots sorted by closest to target
        """
        return sorted(list_robot, key=lambda r: cls.calculate_dist_frm_target(target_position, r.get_position()) or float('inf'))




if __name__ == "__main__":
    from TeamControl.network.ssl_sockets import Vision
    from TeamControl.world.model import WorldModel
    world_model:WorldModel = WorldModel(isYellow= True, isPositive=True)
    recv = Vision(world_model)
    updated = False
    while True:
        updated = recv.listen()
        if world_model.is_detection_updated is True:
            ball_pos = world_model.get_ball()
            our_robots = world_model.get_our_robot(robot_id=None,format=Robot.SELF)
            enemy_robots = world_model.get_enemy_robot(robot_id=None,format=Robot.SELF)
            list_robot = our_robots+enemy_robots
            sorted = Nearest.robot_ordered(ball_pos,list_robot)
            print(f"Nearest Robots {sorted}")
