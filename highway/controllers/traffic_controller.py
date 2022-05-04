import math
import copy
import numpy as np

from controllers.pid import PID


class TrafficController:

    
    def __init__(self, traffic_vehicles, test_file=None):
        self.traffic_vehicles = traffic_vehicles
        self.number_of_vehicles = len(self.traffic_vehicles)
        self.vel_controllers = []
        self.ang_controllers = []

        # Load goals to start
        if test_file is None:
            print("Error: Please give a test file")
            exit()
        
        self.goals = self.load_scenario(test_file)

        # Unlock the traffic vehicles speed
        for vehicle in self.traffic_vehicles:
            vehicle.SPEED_MIN = 0 # [m/s]
            vehicle.SPEED_MAX = 60 # [m/s]
            # Create the velocity and angle controllers for each
            self.vel_controllers.append(PID(Kp_in    = 1,
                                            Ki_in    = 0,
                                            Kd_in    = 2.0,
                                            rate_in  = 4.0))

            self.ang_controllers.append(PID(Kp_in    = 0.05,
                                            Ki_in    = 0.0,
                                            Kd_in    = 0.05,
                                            rate_in  = 4.0))

    def compute_traffic_commands(self, ego_position, obstacle_size=1, traffic_enabled=True):

        # Used to return if the test is complete or not
        complete = False

        # Init the distance sum
        distance_to_goal_sum = 0

        # Make each of the traffic vehicles drive faster
        for i in range(self.number_of_vehicles):

            # Get the vehicle
            vehicle = self.traffic_vehicles[i]
            # Get the vehicle goal
            goal = np.copy(self.goals[i])
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

            # Compute the acceleration and angle
            acc = vel_controller.get_output(goal[0], current_position[0])
            ang = ang_controller.get_output(required_heading, current_heading)

            # Turn off all traffic commands if you aren't in execution mode
            if not traffic_enabled:
                ang = 0
                acc = -1 # To account for friction

            # Don't allow extreme breaking
            acc = max(acc, -25)

            action = {"steering": ang,
                    "acceleration": acc}
            vehicle.act(action)

        return


    def display_goals(self, plt_in):
        # Display the current goals
        c = "k"
        # Plot
        plt_in.scatter(self.goals[:,0], self.goals[:,1], color=c)
        return plt_in


    def load_scenario(self, file_name):
        print(file_name)
        # Read the data
        with open(file_name) as f:
            lines = f.readlines()
        # Drop the first line
        lines = lines[1:]
        # Get the obstacle positions
        obstacle_pos = np.array([[float(val) for val in line.split(',')] for line in lines])
        obstacle_pos = obstacle_pos[:,1:]

        # Return the obstacle positions
        return obstacle_pos
