from TeamControl.Formation.strategic_position import *
from TeamControl.Network.Receiver import grSimVision
from TeamControl.Network.Sender import grSimSender
from TeamControl.Model.world import World
from TeamControl.RobotBehaviour.goToTarget import go_To_Target
from TeamControl.Model.transform_cords import *
from TeamControl.SSL.grSimAction import grSim_Action
from TeamControl.Model.world import World as wm

class activeFormation():
    def __init__(self, isYellow: bool  ) -> None:
        self.isyellow = isYellow
        self.feildSize = (9000, 6000)
        self.world_model = wm(isYellow)
        
    def formation_212_pos(self,player_no, ball_pos):

        '''
        Setup the feild position of the robot depending on the player number 

        [0] is Goalie
        [1] is left Defender
        [2] is Right Defender
        [3] is Mid Feilder
        [4] is left Wing AKA Attacker
        [5] is right Wing 

        This function takes in 3 arguments 
        Feild_size 
        ball_pos aka ball position
        player_no[]

        Output 

        Position of of the player type in the field release to the ball 
        ''' 
        
        FieldPosition.set_field_size(self.feildSize[0], self.feildSize[1]) 
        #Sets feild position 
        #sets the plyer type and their and their reactivness to the ball, waether they are alwyas behind the ball
        #and their minimum and maxinum positions 

        goalie_type = PlayerType("Goalie", 0.2, 0.2, True, -1.0, -0.8)
        left_side_def = PlayerType("Left Defender", 0.5, 0.2, True, -0.9, 0.0)
        right_side_def = PlayerType("Right Defender", 0.5, 0.2, True, -0.9, 0.0)
        mid_feild = PlayerType("Mid Feilder", 1, 0.6, False, -0.6, 0.87)
        left_wing = PlayerType("Left Wing", 1, 0.9, False, -0.9, 1)
        right_wing= PlayerType("Right Wing", 1, 0.9, False, -0.9, 1)

        # Assuming positions are normalised to -1..1 in both X and Y
        # we get the feild coordinates for the respective position 
        positions = [FieldPosition(goalie_type, -0.95, 0), 
        FieldPosition(left_side_def, -0.6,  0.4),
        FieldPosition(right_side_def, -0.6, -0.4),
        FieldPosition(mid_feild, -0.35, 0),
        FieldPosition(left_wing, -0.1, -0.25),
        FieldPosition(right_wing, 0.1, 0.25)]
        
        #find the position in the feild in realtive to the ball depending on which player_no your select 
        player_position = positions[player_no]
        relative_position = player_position.pos(ball_pos[0], ball_pos[1])
        return relative_position
        
if __name__ == "__main__":
    isYellow = False
    robot_id = 1
    role_id = 1
    world_model = Model(isYellow)
    receiver = grSimVision(world_model)
    sender = grSimSender()
    while True:
        updated = receiver.listen()
        if updated : 
            ball_pos = world_model.get_ball()
            robot_pos = world_model.get_our_robot(robot_id)
            target = fromation_212_pos((9000,6000),ball_pos,role_id)
            tar = trans_matrix.world2robot(robot_pos,target)
            vx,vy = go_To_Target(tar)
            action = grSim_Action(isYellow,robot_id, vx,vy)
            sender.send_action(action)