from TeamControl.SSL.game_controller.common import *
import enum 
class STATE(Enum):
    RUNNING = 0
    HALTED = 1
    STOPPED = 2
    TIMEOUT = 3

    def update_stage(self,stage):
        match stage:
            # prepare for game
            case Stage.NORMAL_FIRST_HALF_PRE | Stage.EXTRA_FIRST_HALF_PRE :
                return 
            case Stage.NORMAL_SECOND_HALF_PRE | Stage.EXTRA_SECOND_HALF_PRE:
                return 
            
            # first, second half
            case Stage.NORMAL_FIRST_HALF | Stage.EXTRA_FIRST_HALF :
                return 
            case Stage.NORMAL_SECOND_HALF | Stage.EXTRA_SECOND_HALF :
                return 
            
            # half time breaks
            case Stage.NORMAL_HALF_TIME :
                return 
            case Stage.EXTRA_TIME_BREAK :
                return 
            case Stage.EXTRA_HALF_TIME :
                return
            
            # Penalty 
            case  Stage.PENALTY_SHOOTOUT_BREAK :
                return 
            case  Stage.PENALTY_SHOOTOUT :
                return 
            
            # post game
            case Stage.POST_GAME :
                return 

    
    def update_state(self,command):
        match command:
            # basic
            case Command.HALT:
                return STATE.HALTED
            case Command.STOP:
                return STATE.STOPPED
            case Command.NORMAL_START | Command.FORCE_START:
                return STATE.RUNNING
            
            #timeout
            case Command.TIMEOUT_BLUE:
                return STATE.TIMEOUT
            case Command.TIMEOUT_YELLOW:
                return STATE.TIMEOUT
            
            case Command.PREPARE_KICKOFF_BLUE:
                return
            case Command.PREPARE_KICKOFF_YELLOW:
                return
            
            case Command.DIRECT_FREE_BLUE | Command.INDIRECT_FREE_BLUE:
                return
            case Command.DIRECT_FREE_YELLOW | Command.INDIRECT_FREE_YELLOW:
                return

            # Ball placements
            case Command.BALL_PLACEMENT_BLUE:
                return
            case Command.BALL_PLACEMENT_YELLOW:
                return
            
            # Penalty
            case Command.PREPARE_PENALTY_YELLOW:
                return
            case Command.PREPARE_PENALTY_BLUE:
                return
            