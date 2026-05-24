# Formations and roles 

This directory contains classes to manage formations and roles. 

The demo.py can be run by itself, to visualise and expiremtn with the different 
formations, and should give an idea for how to use the code.
The tk part is just for visualisation, and not needed for the robots.

The text file with the formations has the following format:
- Each new formation starts with "f: ", followed by the formation name. The 
  formation name can be an arbitrary string (no whitespace)
- for all players in the formation, there should be values for the default 
  x-coordinate, y-coordinate, and player type, in 3 seperate lines. These lines
  start with x_pos:, y_pos:, and p_type:, respectively. 
  That is, if there are 5 players in the team, there will be 5 values for the 
  x-coordinates, 5 values for the y-coordinates, and 5 integers indicating the 
  player types. x_pos and y_pos specify the "home coordinate" for a robot, the
  position when the ball is in the centre of the field.
  The x- and y-coordinates should be floats between -1.0 and 1.0. (0.0, 0.0) 
  represents the centre of the field, -1.0 / 1.0 is on the field boundary.
  These values are mapped to the actual field size.
- The next values specify player types. We can specify an arbitrary number of
  player types (can be fewer or more than actual robots).
  A player type specifies how reactive a robot is, that is, how much the robot
  is attracted by the current ball position. x_attr and y_attr are used to 
  seperate the reactiveness in x-direction (left to right), and y-direction 
  (front to back). A value of 0.0 means the ball position does not influence 
  the robot at all, and higher values mean the robot follows the ball position 
  more in x- or y-direction. The values should probably be between 0.0 and 1.0.
- behind_ball is a binary value (0 or 1). 1 means the robot should always stay 
  behind the ball, independent of what is computed from x_attr and the home 
  coordinate. 
- x_min and x_max specify how much the robot should be moving back and forward
  at most, independent of the calculation.
  
Example formation with 3 robots and 2 player types:

```text
f: beta-3
x_pos: -0.95 -0.35 -0.35  # home position for 3 robots (x) 
y_pos:  0.00  0.47 -0.47  # home position for 3 robots (y) 
p_type: 0     1     1     # player types for 3 robots (integer index, 0..n) 
x_attr: 0.1   0.2      # x-attraction for 2 player types 
y_attr: 0.1   0.2      # y-attraction for 2 player types 
behind_ball: 1  0      # player type 0 stays behind ball, player type 1 doesn't
x_min: -1.0  -0.96     # min x coordinate for 2 player types 
x_max: -0.5   0.97     # max x coordinate for 2 player types 
```

The other files in this directory:
- strategic_position.py: contains the PlayerType class and the FieldPosition
                         class. FieldPosition stores the home position and the 
                         PlayerType, and can calculate the desired position of
                         the player based on its home position, player type, 
                         and a given ball position.
- formation.py: Contains the class Formation, used to store an individual 
                formation, i.e., the name of the formation (a string), a 
                dictionary mapping player IDs to their FieldPosition, and a 
                disctionary mapping type IDs to PlayerType.
- formation_manager.py: Contains the FormationManager class that is able to 
                        load formations from a text file, and store them for
                        use during the game.
                        
What is not in this directory (yet):
- The demo.py shows how the formations can be used. There is no code yet 
  for selecting which formation to use. This is (possibly) out of scope for the 
  roles / formations, but the job of a different component (like the strategy).
- Assuming that we have situations where we do not play with the full team (of
  6 or 11 robots, but with 5 or 4 or fewer robots), it might be useful to 
  create different formations for each of these situations.
- There is no code yet how do we select which robot assumes what role. This could 
  be hard coded, but it might be better to dynamically select this based on
  what robots are on the field.
- the positions need to be flipped when we play on the "other" side of the 
  field, and we could include code in here to do this automatically, depending
  on which side we play on.



# How to run the demo:

From the command line, change to the directory that contains the demo.py file and then

```
python ./demo.py
```


The demo.py contains a graphical interface (tkinter) that is not needed for use on robot.

Without the tkinter part, this here are steps that explain how to use it:

```python
# Set the field size before loading formations
field_length = 5000  # Set your desired field length
field_width = 3750   # Set your desired field width
FieldPosition.set_field_size(length, width)
```

```python
# initialise the formations manager and load the formations file:
fm = FormationManager()
fm.load('formations.txt')
```

```python
# this prints all available formation names on screen
formations = list(fm.formations.keys())
print("Available formations:")
for i, formation_name in enumerate(formations):
    print(f"{i}: {formation_name}")
```

```python
# select a specific formation with a given name
formation = fm.formations[formation_name]
players = formation.player_positions
```

```python
# print all player home positions (ball on centre point)
for player, position in players.items():
    home_x, home_y = position.denormalise_pos(position.home_x, position.home_y)
    print(f"Player {player} position: ({home_x}, {home_y}) on field size ({position.field_x}, {position.field_y})")
```

```python
# print all player strategic positions (ball at a given field position)
ball_x = 1234
ball_y = -45
for player, position in players.items():
    pos_x, pos_y = position.pos(ball_x, ball_y)
    print(f"Player {player} strategic position: ({pos_x}, {pos_y}) on field size ({position.field_x}, {position.field_y})")
```
