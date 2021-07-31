import time
import numpy as np
from enum import Enum

class action_enum(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


class EgoController:
    def __init__(self, debug=False):
        self.last_obs = None
        self.current_lane = None
        self.current_action = None
        self.desired_lane = None
        self.debug = debug

        # Create the observations windows
        self.car_infront_window = None
        self.car_left_window = None
        self.car_right_window = None


    # It checks if this is the same vehicle
    def drive(self, current_obs):

        # Init the observations windows 
        self.car_infront_window = np.zeros(np.shape(current_obs)[0])
        self.car_left_window = np.zeros(np.shape(current_obs)[0])
        self.car_right_window = np.zeros(np.shape(current_obs)[0])

        # Get the current lane
        self.current_lane = int(round(current_obs[0][-1]))

        # Get the ego heading:
        ego_heading = np.round(current_obs[0][4],1)

        # Default action is faster if going straight and slower if turning
        if (abs(ego_heading) < 1):
            action = action_enum.FASTER.value
        elif (abs(ego_heading) > 3.5):
            action = action_enum.SLOWER.value
        else:
            action = action_enum.IDLE.value

        # Init variables
        if self.desired_lane is None:
            self.desired_lane = self.current_lane

        # Init variables we want to compute
        car_infront = False
        car_left = False
        car_right = False
    
        # Look through observations
        for i in np.arange(1, np.shape(current_obs)[0]):
            current_observation = current_obs[i]

            # Check we can see the vehicle
            if current_observation[0] == 0:
                continue

            # Compute required information: (24 units is 2 car lengths away)
            car_infront_check = ((-8 <= current_observation[1] <= 16) and (-1.5 <= current_observation[2] <= 1.5)) or car_infront
            car_left_check    = ((-8 <= current_observation[1] <= 16) and (-5 <= current_observation[2] <= -1.6)) or car_left
            car_right_check   = ((-8 <= current_observation[1] <= 16) and (1.6 <= current_observation[2] <= 5)) or car_right

            # Move up the window and save the data
            self.car_infront_window = np.roll(self.car_infront_window, 1) 
            self.car_left_window = np.roll(self.car_left_window, 1) 
            self.car_right_window = np.roll(self.car_right_window, 1) 
            self.car_infront_window[0] = int(car_infront_check)
            self.car_left_window[0] = int(car_left_check)
            self.car_right_window[0] = int(car_right_check)

        
        print(self.car_infront_window)
        print(self.car_left_window)
        print(self.car_right_window)
        # print(self.car_left_window)
        # print(self.car_right_window)

        # Get the latest info
        car_infront = np.any(self.car_infront_window)
        car_right = np.any(self.car_right_window)
        car_left = np.any(self.car_left_window)

        if car_infront:
            if self.current_lane == 0:
                if not car_right:
                    # Make sure we aren't already turning
                    if self.desired_lane == self.current_lane:
                        action = action_enum.LANE_RIGHT.value
                        self.desired_lane += 1
                else:
                    action = action_enum.SLOWER.value
            if self.current_lane == 1:
                if not car_right:
                    # Make sure we aren't already turning
                    if self.desired_lane == self.current_lane:
                        action = action_enum.LANE_RIGHT.value
                        self.desired_lane += 1
                elif not car_left:
                    # Make sure we aren't already turning
                    if self.desired_lane == self.current_lane:
                        action = action_enum.LANE_LEFT.value
                        self.desired_lane -= 1
                else:
                    action = action_enum.SLOWER.value
            if self.current_lane == 2:
                if not car_left:
                    # Make sure we aren't already turning
                    if self.desired_lane == self.current_lane:
                        action = action_enum.LANE_LEFT.value
                        self.desired_lane -= 1
                elif not car_right:
                    # Make sure we aren't already turning
                    if self.desired_lane == self.current_lane:
                        action = action_enum.LANE_RIGHT.value
                        self.desired_lane += 1
                else:
                    action = action_enum.SLOWER.value
            if self.current_lane == 3:
                if not car_left:
                    # Make sure we aren't already turning
                    if self.desired_lane == self.current_lane:
                        action = action_enum.LANE_LEFT.value
                        self.desired_lane -= 1
                else:
                    action = action_enum.SLOWER.value

        # If the desired lane is not equal to the current lane & the car isnt turning, we need to turn
        if (action != action_enum.LANE_RIGHT.value) and (action != action_enum.LANE_LEFT.value):
            if (self.desired_lane != self.current_lane) and (abs(ego_heading) < 1):
                self.desired_lane = self.current_lane

        if self.debug:
            print("Car Controller: ")
            print("|--Current Heading:\t" + str(ego_heading))
            print("|--Current Lane:\t" + str(self.current_lane))
            print("|--Desired Lane:\t" + str(self.desired_lane))
            print("|--Car infront:\t\t" + str(car_infront))
            print("|--Car left:\t\t" + str(car_left))
            print("|--Car right:\t\t" + str(car_right))
            print("|--Action:\t\t" + str(action_enum(action).name))

        # Return the action
        return action

    def default_action(self):
        return action_enum.IDLE.value