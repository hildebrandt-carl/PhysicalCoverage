import math
import copy
import numpy as np

from misc.pid import PID


class TrafficController:

    
    def __init__(self, traffic_vehicles, scenario_file=None):
        self.traffic_vehicles = traffic_vehicles
        self.number_of_vehicles = len(self.traffic_vehicles)
        self.vel_controllers = []
        self.ang_controllers = []

        # Load goals to start
        if scenario_file is not None:
            self.goals = self.load_scenario(scenario_file)
        else:
            self.goals = np.array([[10,4], [10,0], [10, -4]])
            self.goals = self.goals.reshape(1, self.goals.shape[0], self.goals.shape[1])

        self.goal_index = 0
        self.goal_length_counter = 0
        self.initialization = 0

        # Unlock the traffic vehicles speed
        for vehicle in self.traffic_vehicles:
            vehicle.SPEED_MIN = 0 # [m/s]
            vehicle.SPEED_MAX = 60 # [m/s]
            # Create the velocity and angle controllers for each
            self.vel_controllers.append(PID(Kp_in    = 1.0,
                                            Ki_in    = 0.0,
                                            Kd_in    = 2.0,
                                            rate_in  = 4.0))

            self.ang_controllers.append(PID(Kp_in    = 0.05,
                                            Ki_in    = 0.0,
                                            Kd_in    = 0.05,
                                            rate_in  = 4.0))

        # Assign goals based on goal_index
        self.current_goal = self.assign_goal(index = self.goal_index)


    def assign_goal(self, index):

        # Current goal
        temp_current_goal = copy.deepcopy(self.goals[index])

        current_goal = [] 
        for i in range(len(temp_current_goal)):
            goal = temp_current_goal[i]

            # Make sure the goal isn't on the car exactly
            if goal[0] < 5 :
                if i == 0:
                    goal = np.array([7, -4])
                if i == 1:
                    goal = np.array([7, 0])
                if i == 2:
                    goal = np.array([7, 4])

            # Add the goal 
            current_goal.append(goal)
       
        # Turn the goals into an array
        return np.array(current_goal)

    def compute_traffic_commands(self, ego_position, simulation_steps_intervals=10):

        # Used to return if the test is complete or not
        complete = False

        # Make each of the traffic vehicles drive faster
        for i in range(self.number_of_vehicles):

            # Get the vehicle
            vehicle = self.traffic_vehicles[i]
            # Get the vehicle goal
            goal = self.current_goal[i]
            # Get the vehicle controllers
            vel_controller = self.vel_controllers[i]
            ang_controller = self.ang_controllers[i]

            # Ignore the vehicle if its crashed
            if vehicle.crashed:
                continue

            # Compute the distance
            current_position =  vehicle.position - ego_position

            # Compute the angle between the two
            dx = 5
            dy = current_position[1] - goal[1]
            print("Current dx: {}".format(dx))
            print("Current dy: {}".format(dy))
            required_heading = (math.atan2(dx, dy)) - (math.pi / 2)

            # Printing
            print("Current heading: {}".format(math.degrees(vehicle.heading)))
            print("Required heading: {}".format(math.degrees(required_heading)))
            print("Current position: {}".format(current_position))
            print("Current lane: {}".format(vehicle.lane_index[2]))

            # Compute the acceleration and angle
            acc = vel_controller.get_output(goal[0], current_position[0])
            ang = ang_controller.get_output(required_heading, vehicle.heading)

            self.initialization += 1
            if self.initialization < 10:
                acc = 5

            else:
                # If we are not in the init phase
                self.goal_length_counter += 1
                # Wait 10 loops and then increment the goal
                if self.goal_length_counter > simulation_steps_intervals:
                    self.goal_length_counter = 0
                    self.goal_index += 1
                    # Check we are not out of goal positions
                    if self.goal_index >= self.goals.shape[0]:
                        complete = True
                        break
                    else:
                        self.current_goal = self.assign_goal(index = self.goal_index)

            # Better control?
            action = {"steering": ang,
                      "acceleration": acc}
            vehicle.act(action)

        return complete


    def display_goals(self, plt_in):
        # Display the current goals
        c = "k"
        # Plot
        plt_in.scatter(self.current_goal[:,0], self.current_goal[:,1], color=c)
        return plt_in


    def load_scenario(self, file_name):
        return np.load(file_name)
