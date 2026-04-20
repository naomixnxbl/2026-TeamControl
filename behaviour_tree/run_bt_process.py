from multiprocessing import Process, Queue,Event
import random
from .main_tree import MainTree
from TeamControl.world.model import WorldModel
from .test_tree import TestTreeSeq
from .goalie_tree import GoalieRunningSeq
from TeamControl.utils.Logger import LogSaver

import typing
import py_trees

def run_bt_process(is_running:Event, wm:WorldModel, dispatcher_q:Queue,
                   robot_roles:dict=None)->None:
    """
    Run a behaviour tree in a separate process.

    ARGS :
        wm (WorldModel): Shared World Model from Main Loop
        dispatcher_q (multiprocessing.Queue): Queue to send RobotCommands to Dispatcher
        robot_roles (dict): Optional mapping of robot_id -> role name.
                            Valid roles: "striker", "goalie", "default".
                            Example: {0: "striker", 2: "goalie"}
    """

    py_trees.console.has_colours = False

    # create the root of the behaviour tree
    logger = LogSaver()

    # Initialise with default state -- RUNNING
    root = MainTree(wm, dispatcher_q, logger, robot_roles=robot_roles)
    bt = py_trees.trees.BehaviourTree(root)
    bt.setup(timeout=15) # remember to add timeout

    # while is_running.is_set():
    #     bt.tick_tock(1, stop_on_terminal_state=True)
    #     logger.debug(py_trees.display.unicode_tree(root, show_status=True))

    tick_count = 0
    while is_running.is_set():
        bt.tick()
        tick_count += 1

        # Only generate tree snapshot every 100 ticks to avoid expensive string formatting
        if tick_count % 100 == 0:
            tree_snapshot = py_trees.display.unicode_tree(
                root,
                show_status=True,
                visited=True
            )
            logger.debug("\n" + tree_snapshot)