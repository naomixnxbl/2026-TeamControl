from __future__ import annotations
import math
from TeamControl.bt.contracts.intent import IntentMove
from TeamControl.bt.contracts.snapshot import RobotState, Snapshot
from TeamControl.skills._shared import HALF_LEN_M, GOAL_HW_M, DEF_BLOCK_ADVANCE_M, angle_to


def defender_block(snap: Snapshot, robot: RobotState | None, target) -> IntentMove | None:
    """Cover the goal mouth segment the goalie is not protecting.

    Finds the own teammate nearest the goal (assumed to be the goalie), then
    positions on the opposite side so together they split the full goal width.
    """
    if robot is None:
        return None
    goal_x   = -HALF_LEN_M
    goalie_y = 0.0
    min_dist = float('inf')
    for r in snap.own_robots:
        if r.robot_id == robot.robot_id:
            continue
        d = math.hypot(r.position[0] - goal_x, r.position[1])
        if d < min_dist:
            min_dist, goalie_y = d, r.position[1]

    if abs(goalie_y) < 0.05:
        block_y = GOAL_HW_M * 0.55 if snap.ball_position[1] < 0 else -GOAL_HW_M * 0.55
    else:
        block_y = -math.copysign(GOAL_HW_M * 0.55, goalie_y)

    block_pos = (goal_x + DEF_BLOCK_ADVANCE_M, block_y)
    return IntentMove(target_pos=block_pos,
                      target_orientation=angle_to(block_pos, snap.ball_position))
