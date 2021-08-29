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

        # Coverage counter
        total_lines = 90
        self.code_coverage = np.zeros(total_lines)

        self.code_coverage[0] = 1
        self.last_obs = None
        self.code_coverage[1] = 1
        self.current_lane = None
        self.code_coverage[2] = 1
        self.current_action = None
        self.code_coverage[3] = 1
        self.desired_lane = None
        self.code_coverage[4] = 1
        self.debug = debug

        # Create the observations windows
        self.code_coverage[5] = 1
        self.car_infront_window = None
        self.code_coverage[6] = 1
        self.car_left_window = None
        self.code_coverage[7] = 1
        self.car_right_window = None


    # It checks if this is the same vehicle
    def drive(self, current_obs):

        # Init the observations windows
        self.code_coverage[8] = 1 
        self.car_car_infront_window = np.zeros(np.shape(current_obs)[0])
        self.code_coverage[9] = 1
        self.car_infront_window     = np.zeros(np.shape(current_obs)[0])
        self.code_coverage[0] = 1
        self.car_left_window        = np.zeros(np.shape(current_obs)[0])
        self.code_coverage[10] = 1
        self.car_right_window       = np.zeros(np.shape(current_obs)[0])

        # Get the current lane
        self.code_coverage[11] = 1
        self.current_lane = int(round(current_obs[0][-1]))

        # Get the ego heading:
        self.code_coverage[12] = 1
        ego_heading = np.round(current_obs[0][4],1)

        # Default action is faster if going straight and slower if turning
        self.code_coverage[13] = 1
        if (abs(ego_heading) < 1):
            self.code_coverage[14] = 1
            action = action_enum.FASTER.value
        elif (abs(ego_heading) > 3.5):
            self.code_coverage[15] = 1 # elif
            self.code_coverage[16] = 1
            action = action_enum.SLOWER.value
        else:
            self.code_coverage[17] = 1 #elif
            self.code_coverage[18] = 1 #else
            self.code_coverage[19] = 1
            action = action_enum.IDLE.value

        # Init variables
        self.code_coverage[20] = 1
        if self.desired_lane is None:
            self.code_coverage[21] = 1
            self.desired_lane = self.current_lane

        # Init variables we want to compute
        self.code_coverage[22] = 1
        car_far_infront = False
        self.code_coverage[23] = 1
        car_infront     = False
        self.code_coverage[24] = 1
        car_left        = False
        self.code_coverage[25] = 1
        car_right       = False
    
        # Look through observations
        self.code_coverage[26] = 1
        for i in np.arange(1, np.shape(current_obs)[0]):
            self.code_coverage[27] = 1
            current_observation = current_obs[i]

            # Check we can see the vehicle
            self.code_coverage[28] = 1
            if current_observation[0] == 0:
                self.code_coverage[29] = 1
                continue

            # Compute required information: (24 units is 2 car lengths away)
            self.code_coverage[30] = 1
            car_far_infront_check   = (16 <= current_observation[1] <= 24) and (-1.5 <= current_observation[2] <= 1.5)
            self.code_coverage[31] = 1
            car_infront_check       = (-8 <= current_observation[1] <= 16) and (-1.5 <= current_observation[2] <= 1.5)
            self.code_coverage[32] = 1
            car_left_check          = (-8 <= current_observation[1] <= 16) and (-5 <= current_observation[2] <= -1.6)
            self.code_coverage[33] = 1
            car_right_check         = (-8 <= current_observation[1] <= 16) and (1.6 <= current_observation[2] <= 5)

            # Move up the window and save the data
            self.code_coverage[34] = 1
            self.car_car_infront_window = np.roll(self.car_car_infront_window, 1)
            self.code_coverage[35] = 1
            self.car_infront_window     = np.roll(self.car_infront_window, 1) 
            self.code_coverage[36] = 1
            self.car_left_window        = np.roll(self.car_left_window, 1) 
            self.code_coverage[37] = 1
            self.car_right_window       = np.roll(self.car_right_window, 1) 

            self.code_coverage[38] = 1
            self.car_car_infront_window[0] = int(car_far_infront_check)
            self.code_coverage[39] = 1
            self.car_infront_window[0] = int(car_infront_check)
            self.code_coverage[40] = 1
            self.car_left_window[0] = int(car_left_check)
            self.code_coverage[41] = 1
            self.car_right_window[0] = int(car_right_check)

        # Get the latest info
        self.code_coverage[42] = 1
        car_far_infront = np.any(self.car_car_infront_window)
        self.code_coverage[43] = 1
        car_infront = np.any(self.car_infront_window)
        self.code_coverage[44] = 1
        car_right = np.any(self.car_right_window)
        self.code_coverage[45] = 1
        car_left = np.any(self.car_left_window)

        self.code_coverage[46] = 1
        if car_far_infront:
            self.code_coverage[47] = 1
            action = action_enum.IDLE.value

        self.code_coverage[48] = 1
        if car_infront:
            self.code_coverage[49] = 1
            if self.current_lane == 0:
                self.code_coverage[50] = 1
                if not car_right:
                    # Make sure we aren't already turning
                    self.code_coverage[51] = 1
                    if self.desired_lane == self.current_lane:
                        self.code_coverage[52] = 1
                        action = action_enum.LANE_RIGHT.value
                        self.code_coverage[53] = 1
                        self.desired_lane += 1
                else:
                    self.code_coverage[54] = 1 # else
                    self.code_coverage[55] = 1
                    action = action_enum.SLOWER.value
            self.code_coverage[56] = 1
            if self.current_lane == 1:
                self.code_coverage[57] = 1
                if not car_right:
                    # Make sure we aren't already turning
                    self.code_coverage[58] = 1
                    if self.desired_lane == self.current_lane:
                        self.code_coverage[59] = 1
                        action = action_enum.LANE_RIGHT.value
                        self.code_coverage[60] = 1
                        self.desired_lane += 1
                elif not car_left:
                    self.code_coverage[61] = 1 # elif
                    # Make sure we aren't already turning
                    self.code_coverage[62] = 1
                    if self.desired_lane == self.current_lane:
                        self.code_coverage[63] = 1
                        action = action_enum.LANE_LEFT.value
                        self.code_coverage[64] = 1
                        self.desired_lane -= 1
                else:
                    self.code_coverage[61] = 1 # elif
                    self.code_coverage[65] = 1 # else
                    self.code_coverage[66] = 1
                    action = action_enum.SLOWER.value
            self.code_coverage[67] = 1
            if self.current_lane == 2:
                self.code_coverage[68] = 1
                if not car_left:
                    # Make sure we aren't already turning
                    self.code_coverage[69] = 1
                    if self.desired_lane == self.current_lane:
                        self.code_coverage[70] = 1
                        action = action_enum.LANE_LEFT.value
                        self.code_coverage[71] = 1
                        self.desired_lane -= 1
                elif not car_right:
                    self.code_coverage[72] = 1 # elif
                    # Make sure we aren't already turning
                    self.code_coverage[73] = 1
                    if self.desired_lane == self.current_lane:
                        self.code_coverage[74] = 1
                        action = action_enum.LANE_RIGHT.value
                        self.code_coverage[75] = 1
                        self.desired_lane += 1
                else:
                    self.code_coverage[72] = 1 #elif
                    self.code_coverage[76] = 1 #else
                    self.code_coverage[77] = 1
                    action = action_enum.SLOWER.value
            self.code_coverage[78] = 1
            if self.current_lane == 3:
                self.code_coverage[79] = 1
                if not car_left:
                    # Make sure we aren't already turning
                    self.code_coverage[80] = 1
                    if self.desired_lane == self.current_lane:
                        self.code_coverage[81] = 1
                        action = action_enum.LANE_LEFT.value
                        self.code_coverage[82] = 1
                        self.desired_lane -= 1
                else:
                    self.code_coverage[83] = 1 # else
                    self.code_coverage[84] = 1
                    action = action_enum.SLOWER.value

        # If the desired lane is not equal to the current lane & the car isnt turning, we need to turn
        self.code_coverage[85] = 1
        if (action != action_enum.LANE_RIGHT.value) and (action != action_enum.LANE_LEFT.value):
            self.code_coverage[86] = 1
            if (self.desired_lane != self.current_lane) and (abs(ego_heading) < 1):
                self.code_coverage[87] = 1
                self.desired_lane = self.current_lane

        if self.debug:
            print("Car Controller: ")
            print("|--Current Heading:\t" + str(ego_heading))
            print("|--Current Lane:\t" + str(self.current_lane))
            print("|--Desired Lane:\t" + str(self.desired_lane))
            print("|--Car far infront:\t" + str(car_far_infront))
            print("|--Car infront:\t\t" + str(car_infront))
            print("|--Car left:\t\t" + str(car_left))
            print("|--Car right:\t\t" + str(car_right))
            print("|--Action:\t\t" + str(action_enum(action).name))

        # Return the action
        self.code_coverage[88] = 1
        return action

    def default_action(self):
        self.code_coverage[89] = 1
        return action_enum.IDLE.value

    def get_lines_covered(self):
        all_lines = np.arange(len(self.code_coverage))
        covered_lines = []
        for i in range(len(self.code_coverage)):
            if self.code_coverage[i] == 1:
                covered_lines.append(i)
        covered_lines = np.array(covered_lines)
        return covered_lines, all_lines