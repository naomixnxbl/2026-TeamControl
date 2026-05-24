#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:41:45 2024

@author: oliver
"""


class PlayerType: 
    
    def __init__(self, role_name, x_attr, y_attr, behind_ball, x_min, x_max):
        self.role_name = role_name
        self.x_attr = x_attr
        self.y_attr = y_attr
        self.behind_ball = behind_ball
        self.x_min = x_min
        self.x_max = x_max

class FieldPosition:
    
    field_x = 5000
    field_y = 3000

    def __init__(self, player_type, home_x, home_y):
        self.player_type = player_type
        self.home_x = home_x
        self.home_y = home_y
        
    @classmethod
    def set_field_size(cls, length, width):
        cls.field_x = length
        cls.field_y = width
        
    @classmethod
    def normalise_pos(cls, x, y):
        x_norm = 2 * x / cls.field_x
        y_norm = 2 * y / cls.field_y
        return x_norm, y_norm
    
    @classmethod
    def denormalise_pos(cls, x_norm, y_norm):
        x = x_norm * cls.field_x / 2
        y = y_norm * cls.field_y / 2
        return x, y
    
    def pos(self, ball_x, ball_y):
        x, y = FieldPosition.normalise_pos(ball_x, ball_y)
        dynamic_x = self.home_x + (x * self.player_type.x_attr)
        dynamic_y = self.home_y + (y * self.player_type.y_attr)
        # cap the dynamic_x to be within the range for the player type
        dynamic_x = max(self.player_type.x_min, 
                        min(self.player_type.x_max, dynamic_x))
        # we also may have to stay behind the ball
        if self.player_type.behind_ball:
            dynamic_x = min(dynamic_x, x)
            
        dynamic_y = max(-1.0, min(1.0, dynamic_y))
        
        return FieldPosition.denormalise_pos(dynamic_x, dynamic_y)


if __name__ == "__main__":
    # Set the soccer field size (length and width in millimeters)
    FieldPosition.set_field_size(9000, 6000)  

    # Create a player type
    defender = PlayerType(role_name="Defender", x_attr=0.5, y_attr=0.3, behind_ball=True, x_min=-1.0, x_max=0.0)

    # Initialise a player's field position with normalised home coordinates
    defender_position = FieldPosition(player_type=defender, home_x=-0.5, home_y=0.0)

    # Example ball position (in millimeters, with (0,0) being the center of the field)
    ball_x, ball_y = 1000, -1000  # Ball is 1m to the right and 1m down from the center

    # Calculate the defender's dynamic position based on the ball's position
    dynamic_position = defender_position.pos(ball_x, ball_y)

    # Print the calculated dynamic position in millimeters
    print(f"Dynamic Position (in mm): X={dynamic_position[0]:.2f}, Y={dynamic_position[1]:.2f}")

    # Example to demonstrate normalization and denormalization
    normalized_ball_pos = FieldPosition.normalise_pos(ball_x, ball_y)
    print(f"Normalised Ball Position: X={normalized_ball_pos[0]:.2f}, Y={normalized_ball_pos[1]:.2f}")

    denormalized_ball_pos = FieldPosition.denormalise_pos(*normalized_ball_pos)
    print(f"Denormalised Ball Position: X={denormalized_ball_pos[0]:.2f}, Y={denormalized_ball_pos[1]:.2f}")

      