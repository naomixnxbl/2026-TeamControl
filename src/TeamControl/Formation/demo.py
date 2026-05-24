#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:50:28 2024

@author: 30045063
"""

from formation_manager import FormationManager 

import tkinter as tk
from functools import partial

class SoccerFieldSimulator:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=500, height=300, bg='green')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.players_shapes = {}

        # side frame with radio buttons for choice of formation
        self.side_frame = tk.Frame(root)
        self.side_frame.pack(side='right', fill='y')
        self.formation_var = tk.IntVar()  # GUI var for current formation index

        self.fm = FormationManager()
        self.fm.load('formations.txt')
        self.formations = list(self.fm.formations.keys())  # List of formation names
        self.current_formation = 0  # Default to the first formation
        self.formation_var.set(self.current_formation)  # Set the radio button variable default

        # Create radio buttons for each formation
        for i, formation_name in enumerate(self.formations):
            tk.Radiobutton(self.side_frame, text=formation_name, variable=self.formation_var,
                           value=i, command=self.change_formation).pack(anchor='w')

        self.change_formation()
        
    def load_formation(self):
        for shape in self.players_shapes.values():
            self.canvas.delete(shape)
        
        self.current_formation = self.formation_var.get()
        self.players = self.fm.formations[self.formations[self.current_formation]].player_positions
        self.players_shapes = {}

        # Draw players on canvas
        for player, position in self.players.items():
            home_x, home_y = position.denormalise_pos(position.home_x, position.home_y)
            canvas_x = 250 + home_x * 500 / position.field_x
            canvas_y = 150 + home_y * 300 / position.field_y
            self.players_shapes[player] = self.canvas.create_oval(canvas_x-10, canvas_y-10, 
                                                                  canvas_x+10, canvas_y+10, 
                                                                  fill='blue')

        # Create the ball if not already created
        if not hasattr(self, 'ball'):
            self.ball = self.canvas.create_oval(240, 140, 260, 160, fill='white')
            self.canvas.bind("<B1-Motion>", partial(self.move_ball))

    def change_formation(self):
        self.load_formation()

        x1, y1, x2, y2 = self.canvas.coords(self.ball)
        ball_x = (x1 + x2) / 2
        ball_y = (y1 + y2) / 2
        
        self.update_positions(ball_x, ball_y)  # update player positions
        
    def move_ball(self, event):
        self.canvas.coords(self.ball, event.x-10, event.y-10, event.x+10, event.y+10)
        self.update_positions(event.x, event.y)

    def update_positions(self, ball_x, ball_y):
        p = next(iter(self.players))
        # Convert canvas coordinates to field coordinates
        field_ball_x = (ball_x - 250) * self.players[p].field_x / 500
        field_ball_y = (ball_y - 150) * self.players[p].field_y / 300
        # Update player positions
        for player, position in self.players.items():
            dynamic_pos = position.pos(field_ball_x, field_ball_y)
            canvas_x = 250 + dynamic_pos[0] * 500 / position.field_x
            canvas_y = 150 + dynamic_pos[1] * 300 / position.field_y
            self.canvas.coords(self.players_shapes[player], canvas_x-10, 
                               canvas_y-10, canvas_x+10, canvas_y+10)

def main():
    root = tk.Tk()
    app = SoccerFieldSimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
