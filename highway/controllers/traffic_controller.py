import math
import copy
import numpy as np

from controllers.pid import PID


class TrafficController:

    
    def __init__(self, traffic_vehicles, scenario_file=None, index_file=None):
        self.traffic_vehicles = traffic_vehicles
        self.number_of_vehicles = len(self.traffic_vehicles)
        self.vel_controllers = []
        self.ang_controllers = []

        # Load goals to start
        if scenario_file is not None:
            self.goals = self.load_scenario(scenario_file)
            self.expected_indices = self.load_scenario(index_file)
        else:
            self.goals = np.array([[10,4], [10,0], [10, -4]])
            self.expected_indices = np.array([-1], [-1], [-1])
            self.goals = self.goals.reshape(1, self.goals.shape[0], self.goals.shape[1])

        self.goal_index = 0
        self.initialization = 0

        # Unlock the traffic vehicles speed
        for vehicle in self.traffic_vehicles:
            vehicle.SPEED_MIN = 0 # [m/s]
            vehicle.SPEED_MAX = 60 # [m/s]
            # Create the velocity and angle controllers for each
            self.vel_controllers.append(PID(Kp_in    = 0.5,
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

            # Add the goal 
            current_goal.append(goal)
       
        # Turn the goals into an array
        return np.array(current_goal)


    def compute_traffic_commands(self, ego_position, obstacle_size=1):

        # Used to return if the test is complete or not
        complete = False

        # Init the distance sum
        distance_to_goal_sum = 0

        # Make each of the traffic vehicles drive faster
        for i in range(self.number_of_vehicles):

            # Get the vehicle
            vehicle = self.traffic_vehicles[i]
            # Get the vehicle goal
            goal = np.copy(self.current_goal[i])
            # Account for the size of the car
            goal[0] = goal[0] + (obstacle_size * 2.5)
            # Get the vehicle controllers
            vel_controller = self.vel_controllers[i]
            ang_controller = self.ang_controllers[i]

            # Init
            current_position = None
            dx = None
            dy = None
            required_heading = 0
            current_heading = 0
            distance_to_goal = None

            # If the vehicle has not crashed compute the different required metrics
            if not vehicle.crashed:

                # Compute the distance
                current_position =  vehicle.position - ego_position

                # Compute the angle between the two
                dx = 5
                dy = current_position[1] - goal[1]
                required_heading = (math.atan2(dx, dy)) - (math.pi / 2)
                
                # Compute distance to goal
                distance_to_goal = np.linalg.norm(goal - current_position)

                # Get the current heading
                current_heading = vehicle.heading

            # Print Vehicle information
            print("Vehicle: {}".format(i))
            print("|--Crash status:\t{}".format(vehicle.crashed))
            print("|--Current dx:\t\t{}".format(dx))
            print("|--Current dy:\t\t{}".format(dy))
            print("|--Current goal:\t{}".format(goal))
            print("|--Distance to goal:\t{}".format(distance_to_goal))
            print("|--Current heading:\t{}".format(math.degrees(current_heading)))
            print("|--Required heading:\t{}".format(math.degrees(required_heading)))
            print("|--Current position:\t{}".format(current_position))
            print("|--Current lane:\t{}".format(vehicle.lane_index[2]))

            # Ignore the vehicle if its crashed
            if vehicle.crashed:
                continue

            # Sum the distance of the traffic vehicles
            distance_to_goal_sum += np.abs(distance_to_goal)

            # Compute the acceleration and angle
            acc = vel_controller.get_output(goal[0], current_position[0])
            ang = ang_controller.get_output(required_heading, current_heading)

            self.initialization += 1
            if self.initialization < 10:
                acc = 5

            # Better control?
            action = {"steering": ang,
                      "acceleration": acc}
            vehicle.act(action)

        # Only increase the goal when all vehicles are in their positions
        print("Traffic Controller:")
        print("|--Total distance to goal: {}".format(distance_to_goal_sum))
        
        # If the distance to goal is lower than 2.5 move to the next goal:
        if distance_to_goal_sum <= 1:
            self.goal_index += 1
            # Check we are not out of goal positions
            if self.goal_index >= self.goals.shape[0]:
                complete = True
                print("|--All goals complete")
            else:
                self.current_goal = self.assign_goal(index = self.goal_index)
                print("|--Traffic vehicles in position, switching to next goal")

        return complete

    def get_expected_index(self):
        return self.expected_indices[self.goal_index]


    def display_goals(self, plt_in):
        # Display the current goals
        c = "k"
        # Plot
        plt_in.scatter(self.current_goal[:,0], self.current_goal[:,1], color=c)
        return plt_in


    def load_scenario(self, file_name):
        return np.load(file_name)
