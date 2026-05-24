#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:53:35 2024

@author: oliver
"""

from formation import Formation
from strategic_position import PlayerType, FieldPosition

class FormationManager:
    
    def __init__(self):
        self.formations = {}
            
    def remove_comment(self,s):
        return s.split('#', 1)[0].strip()   

    def store(self, fname, x_positions, y_positions, p_types, x_attr, y_attr, behind_ball, x_min, x_max):
        player_types = {}
        for index, (x_attr_val, y_attr_val, behind_ball_val, x_min_val, x_max_val) in enumerate(zip(x_attr, y_attr, behind_ball, x_min, x_max)):
            pt = PlayerType(index, x_attr_val, y_attr_val, behind_ball_val, x_min_val, x_max_val)
            player_types[index] = pt
        player_positions = {}
        for index, (x_pos, y_pos, p_type) in enumerate(zip(x_positions, y_positions, p_types)):
            fp = FieldPosition(player_types[p_type], x_pos, y_pos)
            player_positions[index] = fp
        formation = Formation(fname, player_positions, player_types)
        self.formations[fname] = formation


    def load(self, filepath):
        self.formations = {}
        
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
        current_formation = None
        x_positions, y_positions, p_types = [], [], []
        x_attr, y_attr, behind_ball, x_min, x_max = [], [], [], [], []
        
        for line in lines:
            # skip comments and empty lines
            line = self.remove_comment(line)
            if not line:
                continue

            # formation identifier found
            if line.startswith('f:'):
                if current_formation is not None:
                    self.store(current_formation, x_positions, y_positions, p_types, x_attr, y_attr, behind_ball, x_min, x_max)
                current_formation = line[2:].strip()
                x_positions, y_positions, p_types = [], [], []
                x_attr, y_attr, behind_ball, x_min, x_max = [], [], [], [], []
            elif line.startswith('x_pos:'):                
                x_positions = [float(val) for val in line[6:].split()]
            elif line.startswith('y_pos:'):
                y_positions = [float(val) for val in line[6:].split()]
            elif line.startswith('p_type:'):
                p_types = [int(val) for val in line[7:].split()]
            elif line.startswith('x_attr:'):                
                x_attr = [float(val) for val in line[7:].split()]
            elif line.startswith('y_attr:'):                
                y_attr = [float(val) for val in line[7:].split()]
            elif line.startswith('behind_ball:'):                
                behind_ball = [True if item == '1' or item == 'True' else False for item in line[12:].split()]
            elif line.startswith('x_min:'):                
                x_min = [float(val) for val in line[6:].split()]
            elif line.startswith('x_max:'):                
                x_max = [float(val) for val in line[6:].split()]

        if current_formation is not None:
            self.store(current_formation, x_positions, y_positions, p_types, x_attr, y_attr, behind_ball, x_min, x_max)

# Example usage:
if __name__ == "__main__":
    fm = FormationManager()
    fm.load('formations.txt')
    
            