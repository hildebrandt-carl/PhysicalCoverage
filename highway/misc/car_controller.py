from enum import Enum
import numpy as np

class action_enum(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


class EgoController:
    def __init__(self, debug=False, spawn_location=None):
        self.last_obs = None
        self.current_lane = None
        self.current_action = None
        self.desired_lane = None
        self.debug = debug

    # It checks if this is the same vehicle
    def drive(self, current_obs):
        # Get the current lane
        self.current_lane = int(round(current_obs[0][-1]))

        # Default action is faster
        action = action_enum.FASTER.value

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

            # Compute required information:
            car_infront = ((-0.05 <= current_observation[1] <= 15) and (-1 <= current_observation[2] <= 1)) or car_infront
            car_left    = ((-0.05 <= current_observation[1] <= 10) and (-5 <= current_observation[2] <= -3)) or car_left
            car_right   = ((-0.05 <= current_observation[1] <= 10) and (3 <= current_observation[2] <= 5)) or car_right

            if car_infront:
                if self.current_lane == 0:
                    if not car_right:
                        # Make sure we arent already turning
                        if self.desired_lane == self.current_lane:
                            action = action_enum.LANE_RIGHT.value
                            self.desired_lane += 1
                    else:
                        action = action_enum.SLOWER.value
                if self.current_lane == 1:
                    if not car_right:
                        # Make sure we arent already turning
                        if self.desired_lane == self.current_lane:
                            action = action_enum.LANE_RIGHT.value
                            self.desired_lane += 1
                    elif not car_left:
                        # Make sure we arent already turning
                        if self.desired_lane == self.current_lane:
                            action = action_enum.LANE_LEFT.value
                            self.desired_lane -= 1
                    else:
                        action = action_enum.SLOWER.value
                if self.current_lane == 2:
                    if not car_left:
                        # Make sure we arent already turning
                        if self.desired_lane == self.current_lane:
                            action = action_enum.LANE_LEFT.value
                            self.desired_lane -= 1
                    elif not car_right:
                        # Make sure we arent already turning
                        if self.desired_lane == self.current_lane:
                            action = action_enum.LANE_RIGHT.value
                            self.desired_lane += 1
                    else:
                        action = action_enum.SLOWER.value
                if self.current_lane == 3:
                    if not car_left:
                        # Make sure we arent already turning
                        if self.desired_lane == self.current_lane:
                            action = action_enum.LANE_LEFT.value
                            self.desired_lane -= 1
                    else:
                        action = action_enum.SLOWER.value

        if self.debug:
            print("Car Controller: ")
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