#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:02:54 2024

@author: oliver
"""

from strategic_position import PlayerType, FieldPosition

class Formation:
    
    def __init__(self, name, player_positions, player_types):
        """
        Initialises a Formation instance.
        
        :param name: The name of the formation (e.g., '4-4-2').
        :param player_positions: A dictionary mapping each player's number or identifier
                                 to their FieldPosition. 
                                 Example structure:
                                 {0: FieldPosition_instance1, 1: FieldPosition_instance2, ...}
        :param player_types: A dicttionary mapping a type id to a PlayerType.
        """
        self.name = name
        self.player_positions = player_positions
        self.player_types = player_types
        
    def display_formation(self):
        """
        Prints the formation details, including the name and the assigned positions and types for each player.
        """
        print(f"Formation: {self.name}")
        for player, position in self.player_positions.items():
            print(f"Player {player}: Position - ({position.home_x:.2f}, {position.home_y:.2f})")


# Example usage:
if __name__ == "__main__":
    goalie_type = PlayerType("Goalie", 0, 0, True, -1.0, -0.5)
    sweeper = PlayerType("Sweeper", 0.5, 0.3, True, -0.5, 0.0)
    centre_back = PlayerType("Centre Back", 0.5, 0.3, True, -0.5, 0.0)
    full_back = PlayerType("Full Back", 0.5, 0.3, True, -0.5, 0.0)
    centre_midfield = PlayerType("Centre Midfield", 0.4, 0.4, False, -0.3, 0.3)
    side_midfield = PlayerType("Side Midfield", 0.4, 0.4, False, -0.3, 0.3)
    centre_forward = PlayerType("Centre Forward", 0.3, 0.5, False, -0.1, 0.5)
    winger = PlayerType("Winger", 0.3, 0.5, False, -0.1, 0.5)

    # Assuming positions are normalised to -1..1 in both X and Y
    goalie = FieldPosition(goalie_type, -0.95, 0)
    defender_1 = FieldPosition(centre_back, -0.5,  0.4)
    defender_2 = FieldPosition(centre_back, -0.5, -0.4)
    defender_3 = FieldPosition(full_back, -0.45,  0.8)
    defender_4 = FieldPosition(full_back, -0.45, -0.8)
    midfielder_1 = FieldPosition(centre_midfield, 0, 0)
    midfielder_2 = FieldPosition(centre_midfield, 0.1, -0.6)
    midfielder_3 = FieldPosition(centre_midfield, 0.1,  0.6)
    attacker_1 = FieldPosition(centre_forward, 0.8, 0)
    attacker_2 = FieldPosition(winger, 0.7, -0.8)
    attacker_3 = FieldPosition(winger, 0.7,  0.8)

    # Creating a formation
    formation_433 = Formation("4-3-3", {
        0: goalie,
        1: defender_1,
        2: defender_2,
        3: defender_3,
        4: defender_4,
        5: midfielder_1,
        6: midfielder_2,
        7: midfielder_3,
        8: attacker_1,
        9: attacker_2,
        10: attacker_3,
        }, {
            'Goalie': goalie_type,
            'Sweeper': sweeper,
            'Centre Back': centre_back,
            'Full Back': full_back,
            'Centre Midfield': centre_midfield,
            'Side Midfield': side_midfield,
            'Centre Forward': centre_forward,
            'Winger': winger,
        })

    # Displaying the formation
    formation_433.display_formation()
