from queue import Queue

from TeamControl.robot.voronoi_navigator import (
    _publish_planned_path,
    _robot_is_in_front_of_possessor,
    _steal_ignore_keys,
)


class _Robots:
    def __init__(self, poses):
        self._poses = poses

    def iter_team(self, is_yellow):
        for (team_yellow, robot_id), pose in self._poses.items():
            if team_yellow == is_yellow:
                yield robot_id, pose

    def relative_to_ball(self, is_yellow, robot_id, ball):
        pose = self._poses[(is_yellow, robot_id)]
        from TeamControl.world.transform_cords import world2robot

        rel = world2robot(pose, ball)
        import math

        return rel, math.hypot(rel[0], rel[1]), math.atan2(rel[1], rel[0])


class _Cache:
    def __init__(self, poses):
        self.robots = _Robots(poses)


class _Plan:
    def __init__(self, *, is_path_free, waypoints=(), active_target_pose=None):
        self.is_path_free = is_path_free
        self.waypoints = waypoints
        self.active_target_pose = active_target_pose
        self.need_reroute = False
        self.did_reroute = False


def test_direct_free_path_publishes_empty_planned_path():
    planner_path_q = Queue()

    _publish_planned_path(
        planner_path_q,
        robot_id=0,
        is_yellow=True,
        robot_pose=(0.0, 0.0, 0.0),
        plan=_Plan(
            is_path_free=True,
            active_target_pose=(1000.0, 0.0, 0.0),
        ),
        now_s=1.0,
    )

    update = planner_path_q.get_nowait()
    assert update["points"] == ()
    assert update["is_path_free"] is True


def test_rerouted_path_publishes_robot_pose_and_waypoints():
    planner_path_q = Queue()

    _publish_planned_path(
        planner_path_q,
        robot_id=0,
        is_yellow=True,
        robot_pose=(0.0, 0.0, 0.0),
        plan=_Plan(
            is_path_free=False,
            waypoints=((100.0, 0.0, 0.0), (200.0, 0.0, 0.0)),
        ),
        now_s=1.0,
    )

    update = planner_path_q.get_nowait()
    assert update["points"] == ((0.0, 0.0), (100.0, 0.0), (200.0, 0.0))
    assert update["is_path_free"] is False


def test_robot_in_front_of_possessor_rule():
    possessor = (0.0, 0.0, 0.0)

    assert _robot_is_in_front_of_possessor((400.0, 0.0, 0.0), possessor)
    assert not _robot_is_in_front_of_possessor((-100.0, 0.0, 0.0), possessor)
    assert not _robot_is_in_front_of_possessor((400.0, 400.0, 0.0), possessor)


def test_steal_ignore_key_requires_possession_and_front_position():
    poses = {
        (True, 0): (350.0, 0.0, 3.14),
        (False, 1): (0.0, 0.0, 0.0),
    }
    cache = _Cache(poses)

    keys = _steal_ignore_keys(
        cache,
        is_yellow=True,
        robot_id=0,
        robot_pose=poses[(True, 0)],
        ball_pos=(80.0, 0.0),
    )

    assert keys == ((False, 1),)


def test_steal_ignore_key_rejects_loose_or_off_angle_ball():
    poses = {
        (True, 0): (350.0, 0.0, 3.14),
        (False, 1): (0.0, 0.0, 0.0),
    }
    cache = _Cache(poses)

    loose = _steal_ignore_keys(
        cache,
        is_yellow=True,
        robot_id=0,
        robot_pose=poses[(True, 0)],
        ball_pos=(95.0, 0.0),
    )
    off_angle = _steal_ignore_keys(
        cache,
        is_yellow=True,
        robot_id=0,
        robot_pose=poses[(True, 0)],
        ball_pos=(80.0, 2.0),
    )

    assert loose == ()
    assert off_angle == ()
