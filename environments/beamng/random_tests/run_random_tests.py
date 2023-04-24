import os
import sys
import time
import math
import copy
import string
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry import Point

from beamngpy.beamngcommon import *
from beamngpy.beamngcommon import angle_to_quat

from beamngpy.sensors import Timer
from beamngpy.sensors import Lidar
from beamngpy.sensors import Damage

from beamngpy import Vehicle
from beamngpy import Scenario
from beamngpy import BeamNGpy
from beamngpy import setup_logging

from beamngpy.visualiser import LidarVisualiser

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from shapely.geometry import Polygon


class BeamNG_Random_Test_Generation():
    # Init
    def __init__(self, bng_loc, bng_usr, number_traffic_vehicles, port, ai_mode):

        # Define parameters about the world
        self.lanewidth = 3.75

        # Setup Beamng
        setup_logging()
        self.beamng = BeamNGpy('localhost', port, home=bng_loc, user=bng_usr)

        # Open beamng
        self.bng = self.beamng.open(launch=True)

        # Save the ai mode
        self.current_ai_mode = ai_mode

        # Create the scenarios
        self.scenario = Scenario('west_coast_usa', 'Random Test Generation', description='Tests for science')

        # Define the number of traffic vehicles
        self.number_traffic_vehicles = number_traffic_vehicles

        # Create the vehicles
        self.traffic_vehicles = self.create_traffic_vehicles(self.number_traffic_vehicles)
        self.ego_vehicle = Vehicle('ego_vehicle', model='etk800', licence='Carlos', color='Red')
        self.target_vehicle = Vehicle('target_vehicle', model='hatch', licence='Target', color='Red')

        # Setup the vehicles sensors
        self.ego_vehicle = self.setup_sensors(self.ego_vehicle)

        # Spawn the ego vehicle in a random lane
        lane = random.choice([1,2,3,4])
        self.spawn_ego_vehicle(lane, spawn=True)

        # Spawn the target vehicle at the target
        self.spawn_target_vehicle(target=[516.65, 728.85, 117.5])

        # Spawn the traffic vehicles
        self.spawn_traffic_vehicles()

        # Start the scene for the first time
        self.start_new_scenario()

    # Start a new scenario
    def start_new_scenario(self):

        # Make the scenario
        self.scenario.make(self.beamng)

        # Load the scenario
        self.bng.load_scenario(self.scenario)

        # Configure and start beamng
        self.simulation_rate = 60
        self.bng.set_steps_per_second(self.simulation_rate)
        self.bng.set_deterministic()
        # self.bng.hide_hud()
        self.bng.start_scenario()

        # Start the traffic
        self.bng.start_traffic(self.traffic_vehicles)

        # Start the AI (40m/s == 145kph -> The highway speed limit is 35m/s == 120kmph)
        self.start_ego_ai(max_speed=40, aggression=1)

        # Pause the scenario and wait for the traffic vehicles to spawn
        wait_time_per_vehicle = 4
        self.bng.step((wait_time_per_vehicle * len(self.traffic_vehicles)) / 0.05, wait=True) 
        self.bng.pause()

        # Restart the scenario incase it had already been run before
        self.bng.restart_scenario()

    # Used to run the scenario
    def run_scenario(self, sensor_rate=4, duration=20, save_name=""):

        # Convert the rate to steps
        time_period = 1.0 / sensor_rate
        num_steps = time_period / (1.0 / self.simulation_rate)

        # Create the data file
        f = open(save_name, 'w')
        f.write("timestamp, position, orientation, velocity, lidar_readings, damage, total_accidents, num_traffic_vehicles, closest_traffic_vehicle, closest_traffic_vehicle_velocity\n")

        # Start the scene for the first time
        self.bng.restart_scenario()

        # Make sure you are focused on the ego vehicle
        self.bng.switch_vehicle(self.ego_vehicle)

        self.ego_vehicle.reset_coverage_arrays()
        self.ego_vehicle.start_coverage()
        self.update_positions()

        # Get the start time
        self.ego_vehicle.poll_sensors()

        start_time = self.timer_sensor.data["time"]
        current_time = 0 
        previous_damage = 0
        currently_in_an_accident = False
        total_physical_crashes = 0

        # Run for the duration requested
        while current_time - start_time <= duration:

            # Update the vehicle and get its new sensor data
            self.ego_vehicle.poll_sensors()

            # Get the sensor data
            current_lidar_reading   = self.lidar_sensor.data['points']
            total_damage            = self.damage_sensor.data["damage"]
            current_time            = self.timer_sensor.data["time"]

            # Get the current state
            current_state           = self.ego_vehicle.state

            # Compute the damage received for this sensor readings
            damaged_received        = total_damage - previous_damage
            previous_damage         = total_damage

            # Count the number of crashes
            if damaged_received > 1:
                if not currently_in_an_accident:
                    currently_in_an_accident = True
                    total_physical_crashes += 1
            else:
                currently_in_an_accident = False

            # Get the current vehicle state
            current_position        = np.round(current_state["pos"], 4)
            current_velocity        = np.round(current_state["vel"], 4)
            damaged_received        = np.round(damaged_received, 4)
            total_damage            = np.round(total_damage, 4)

            # Get the origentation of the vehicle
            current_direction       = np.round(current_state["dir"], 4)
            quat, euler             = self.get_vehicle_orientation(current_state)
            euler                   = np.round(euler, 4)
            quat                    = np.round(quat, 4)

            # Get the closest vehicle
            t_distance, t_velocity = self.get_closest_traffic_vehicle()

            print("-----------------------------")
            print("Vehicle Status: t - {}".format(current_time))
            print("|--Position: {}".format(current_position))
            print("|--Velocity: {}".format(current_velocity))
            print("|--Current Damage: {}".format(damaged_received))
            print("|--Total Accidents: {}".format(total_physical_crashes))
            print("|--Total Damage: {}".format(total_damage))
            print("|--Quat Orientation: {}".format(quat))
            print("|--Euler Orientation: {}".format(euler))
            print("Traffic Status:")            
            print("|--Distance to Closest Traffic Vehicle: {}".format(t_distance))
            print("|--Closest Traffic Vehicle Velocity: {}".format(t_velocity))
            print("Mode Status:")        
            print("|--AI mode: {}".format(self.current_ai_mode))
            print("|--Currently in accident: {}".format(currently_in_an_accident))

            # Write the data to file
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(current_time - start_time,
                                                    current_position.tolist(),
                                                    current_direction.tolist(),
                                                    current_velocity.tolist(),
                                                    current_lidar_reading.tolist(),
                                                    damaged_received,
                                                    total_physical_crashes,
                                                    self.number_traffic_vehicles,
                                                    t_distance.tolist(),
                                                    t_velocity.tolist()))

            # Step the simulation forward (1 step is 0.05 seconds)
            self.bng.step(num_steps)

        # Close the file
        f.close()

        # Return true when done
        return total_physical_crashes

    # Close beamng
    def stop(self):
        self.bng.close()
        exit()

    # Starts the AI for the ego vehicle
    def start_ego_ai(self, max_speed, aggression):
        # Tell it to chase the target vehicle
        self.ego_vehicle.ai_drive_in_lane(False)
        self.ego_vehicle.ai_set_target("target_vehicle", "chase")
        self.ego_vehicle.ai_set_mode('chase')

        # Set the speed limit and agression
        self.ego_vehicle.ai_set_speed(max_speed, mode=self.current_ai_mode)
        self.ego_vehicle.ai_set_aggression(aggression)

    # Returns lane counted from left to right (1 through 4)
    def spawn_ego_vehicle(self, lane, spawn=False):
        # Make sure you are sending the correct lane information
        assert(1 <= lane <= 4)

        # Compute the correct lane adjustment
        lane_adjustment = (lane - 1) * self.lanewidth
        sp = {'pos': (-852.024 , -513.641 - lane_adjustment, 106.620), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}

        if spawn:
            # Add the ego vehicle to the scenario
            self.scenario.add_vehicle(self.ego_vehicle, pos=sp['pos'], rot=sp['rot'], rot_quat=sp['rot_quat'])
        else:
            # Update the vehicles position
            self.ego_vehicle.pos = sp['pos']

    # Spawns the target vehicle at the target
    def spawn_target_vehicle(self, target):
        # Compute the correct lane adjustment
        sp = {'pos': (target[0] , target[1], target[2]), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}

        # Add the ego vehicle to the scenario
        self.scenario.add_vehicle(self.target_vehicle, pos=sp['pos'], rot=sp['rot'], rot_quat=sp['rot_quat'])

    # Spawn the traffic vehicles in a predefine box
    def spawn_traffic_vehicles(self):

        # Create the spawn box
        spawn_box = Polygon([(-811.3121337890625, -486.3587341308594), (-820.1568603515625, -478.5815734863281), (-778.927490234375, -437.0894775390625), (-771.5311889648438, -445.2535400390625)])
        spawn_height = 106.5

        # Used to keep track of previous spawn locations
        previous_spawn_locations = []

        # Get n random points inside the spawn box and spawn the vehicles
        for i in range(len(self.traffic_vehicles)):
            # Set the acceptable location flag to false
            acceptable_location = False

            # Keep trying until we have found acceptable locations
            while not acceptable_location:
                # Get a random point inside the box
                p = self.get_random_point_in_polygon(spawn_box)

                # assume the location is acceptable
                acceptable_location = True

                # Check if this fits all other locations
                for o in previous_spawn_locations:
                    # If this is too close to another point reject it
                    if p.distance(o) <= 4:
                        acceptable_location = False

            # Add the point to the previously accepted locations
            previous_spawn_locations.append(p)

            # Create the spawn point
            sp = {'pos': (p.x , p.y, spawn_height), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}
            print(sp["pos"])
            # Get the vehicle
            t = self.traffic_vehicles[i]

            # Spawn the vehicle there
            print("Added vehicle {} in position {}".format(i, sp['pos']))
            self.scenario.add_vehicle(t, pos=sp['pos'], rot=sp['rot'], rot_quat=sp['rot_quat'])

    # Attached sensors to the ego vehicle
    def setup_sensors(self, vehicle):
        # Create the sensors
        self.damage_sensor = Damage()
        self.lidar_sensor = Lidar(offset=(0.35, -1.75, 0.75), direction=(-0.943858, -0.33035, 0), angle=180, vangle=0.1, hz=60, max_dist=200, vres=5, rps=1e5)
        self.timer_sensor = Timer()

        # Attach them to the vehicle
        vehicle.attach_sensor('lidar', self.lidar_sensor)
        vehicle.attach_sensor('damage', self.damage_sensor)
        vehicle.attach_sensor('timer', self.timer_sensor)

        # Return the vehicle
        return vehicle

    # Get the closest vehicle 
    def get_closest_traffic_vehicle(self):
        # keeping track of the closest vehicles
        closest_traffic_vehicle = None
        minimum_distance = np.inf

        ego_position = np.array(self.ego_vehicle.state["pos"])

        # Loop through each of the traffic vehicles
        for t in self.traffic_vehicles:
            t.poll_sensors()
            t_position = np.array(t.state["pos"])
            dist = np.linalg.norm(t_position - ego_position)

            if dist < minimum_distance:
                minimum_distance = dist
                closest_traffic_vehicle = t

        minimum_distance = np.round(minimum_distance, 4)
        traffic_velocity = np.round(t.state["vel"], 4)

        return minimum_distance, traffic_velocity

    # Create and return the traffic vehicles
    def create_traffic_vehicles(self, number_traffic_vehicles):
        traffic_vehicles = []
        # Create the vehicles
        for i in range(number_traffic_vehicles):
            name = 'Traffic{}'.format(i)
            v = Vehicle(name, model='etk800', licence=name, color='White')
            traffic_vehicles.append(v)
        # Return the traffic vehicles
        return traffic_vehicles

    # Randomly places the ego and traffic vehicles in new positions
    def update_positions(self):
        lane = random.choice([1,2,3,4])
        lane_adjustment = (lane - 1) * self.lanewidth
        sp = {'pos': (-852.024 , -513.641 - lane_adjustment, 106.620), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}
        self.bng.teleport_vehicle(self.ego_vehicle.vid, pos=sp['pos'], rot=sp['rot'], rot_quat=sp['rot_quat'])

        # Create the spawn box
        spawn_box = Polygon([(-811.3121337890625, -486.3587341308594), (-820.1568603515625, -478.5815734863281), (-778.927490234375, -437.0894775390625), (-771.5311889648438, -445.2535400390625)])
        spawn_height = 107

        # Used to keep track of previous spawn locations
        previous_spawn_locations = []

        # Get n random points inside the spawn box and spawn the vehicles
        for i in range(len(self.traffic_vehicles)):
            # Set the acceptable location flag to false
            acceptable_location = False

            # Keep trying until we have found acceptable locations
            while not acceptable_location:
                # Get a random point inside the box
                p = self.get_random_point_in_polygon(spawn_box)

                # assume the location is acceptable
                acceptable_location = True

                # Check if this fits all other locations
                for o in previous_spawn_locations:
                    # If this is too close to another point reject it
                    if p.distance(o) <= 4:
                        acceptable_location = False

            # Add the point to the previously accepted locations
            previous_spawn_locations.append(p)

            # Create the spawn point
            sp = {'pos': (p.x , p.y, spawn_height), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}

            # Get the vehicle
            t = self.traffic_vehicles[i]
            self.bng.teleport_vehicle(t.vid, pos=sp['pos'], rot=sp['rot'], rot_quat=sp['rot_quat'])

    # Gets a random point inside a polygon
    def get_random_point_in_polygon(self, poly):
        minx, miny, maxx, maxy = poly.bounds
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(p):
                return p

    # Get the vehicle orientation
    def get_vehicle_orientation(self, vehicle_state):
        # Get the vehicle vector and the base vector
        rot_vec = vehicle_state["dir"]
        rot_vec[2] = 0
        vec2 = [0, -1, 0]

        # Compute the euler angle between both
        cosTh = np.dot(rot_vec, vec2)
        sinTh = np.cross(rot_vec, vec2)
        euler = np.rad2deg(np.arctan2(sinTh,cosTh))  
        quat = angle_to_quat(euler)

        # Return both vectors
        return quat, euler

    # Get the branches covered
    def get_branches_covered(self):
        coverage_array = self.ego_vehicle.get_branch_coverage()

        # Gets all the covered branches
        all_branches = []
        branches_covered = []

        # For each value in the coverage array
        for i in range(len(coverage_array)):
            branch_number = i + 1
            all_branches.append(branch_number)
            covered = coverage_array[i]
            if covered:
                branches_covered.append(branch_number)

        return all_branches, branches_covered

    # Get the lines covered
    def get_lines_covered(self):
        coverage_array = self.ego_vehicle.get_line_coverage()

        # Gets all the covered and uncovered lines
        all_lines = []
        covered_lines = []

        # For each value in the coverage array
        for i in range(len(coverage_array)):
            line_number = i + 1
            all_lines.append(line_number)
            covered = coverage_array[i]
            if covered:
                covered_lines.append(line_number)

        return all_lines, covered_lines

    # Get the lines covered
    def get_path_taken(self):
        path_taken = self.ego_vehicle.get_path_taken()
        return path_taken

    # Reset the code coverage metric
    def reset_code_coverage(self):
        self.ego_vehicle.reset_coverage_array()


# Used to get a random string
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase + string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

# Get the file arguments
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_runs',    type=int, default=100,                help="The number of times you want to run the scenario")
    parser.add_argument('--traffic_count', type=int, default=1,                  help="The number of traffic vehicles")
    parser.add_argument('--port',          type=int, default=64256,              help="The port used to connect.")
    parser.add_argument('--workspace',     type=str, default="BeamNGWorkspace", help="The name of the workspace folder")
    parser.add_argument('--ai_mode',       type=str, default="limit",            help="(limit) -> drive between 0 and max speed as AI sees fit. (set) -> Try and maintain max speed")
    args = parser.parse_args()
    return args

# Main function
def main():

    args = get_arguments()

    # Init variables
    total_runs              = args.total_runs
    number_traffic_vehicles = args.traffic_count
    port                    = args.port
    user_directory          = args.workspace
    current_ai_mode         = args.ai_mode

    # Create the beamng class
    current_user = os.environ.get("USERNAME")
    bng_loc = 'C:\\Users\\{}\\Beamng\\BeamNG.tech-0.23.5.1'.format(current_user)
    bng_usr = 'C:\\Users\\{}\\Beamng\\{}'.format(current_user, user_directory)
    bng_obj = BeamNG_Random_Test_Generation(bng_loc=bng_loc, bng_usr=bng_usr, number_traffic_vehicles=number_traffic_vehicles, port=port, ai_mode=current_ai_mode)

    # Check to see that there is an output folder
    lidar_path = "./output/random_tests/physical_coverage/lidar/{}_external_vehicles".format(number_traffic_vehicles)
    if not os.path.exists(lidar_path):
        os.makedirs(lidar_path)

    code_coverage_path = "./output/random_tests/code_coverage/raw/{}_external_vehicles".format(number_traffic_vehicles)
    if not os.path.exists(code_coverage_path):
        os.makedirs(code_coverage_path)

    # Run the scenario twice
    for i in range(total_runs):
        # Create the output file
        random_string = get_random_string(4)
        current_time = time.time() * 1000
        current_time = str(current_time)
        current_time = current_time[7:current_time.find(".")]
        lidar_save_name = lidar_path + '/beamng_random_r{}_t{}_{}_{}_{}.csv'.format(i, number_traffic_vehicles, current_ai_mode, current_time, random_string)
        code_coverage_save_name = code_coverage_path + '/beamng_random_r{}_t{}_{}_{}_{}.txt'.format(i, number_traffic_vehicles, current_ai_mode, current_time, random_string)

        # Run the scenario
        total_physical_accidents = bng_obj.run_scenario(sensor_rate=4, duration=50, save_name=lidar_save_name)

        print("-----------------------------")
        # Get the code covergage
        all_branches, covered_branches  = bng_obj.get_branches_covered()
        all_lines, covered_lines  = bng_obj.get_lines_covered()
        print("Total covered lines: {}/{}".format(len(covered_lines), len(all_lines)))
        print("Total branches covered: {}/{}".format(len(covered_lines), len(all_lines)))
        
        # Compute the uncovered lines
        all_lines_set = set(all_lines)
        covered_lines_set = set(covered_lines)
        uncovered_lines_set = all_lines_set - covered_lines_set
        print("Total uncovered lines: {}".format(len(uncovered_lines_set)))

        # Compute the coverage
        all_branches_set = set(all_branches)
        branches_covered_set = set(covered_branches)
        uncovered_branches_set = all_branches_set - branches_covered_set
        print("Total uncovered branches: {}".format(len(uncovered_branches_set)))

        # Get the path taken
        path_taken  = bng_obj.get_path_taken()
        print("Received Path Taken: {}".format(len(path_taken)))
        print("-----------------------------")

        # # Save the code coverage
        f = open(code_coverage_save_name, 'w')
        f.write("Lines covered: {}\n".format(sorted(list(covered_lines_set))))
        f.write("Total lines covered: {}\n".format(len(list(covered_lines_set))))
        f.write("-----------------------------\n")
        f.write("All lines: {}\n".format(sorted(list(all_lines_set))))
        f.write("Total all lines: {}\n".format(len(list(all_lines_set))))
        f.write("-----------------------------\n")
        f.write("Lines not covered: {}\n".format(sorted(list(uncovered_lines_set))))
        f.write("Total lines not covered: {}\n".format(len(list(uncovered_lines_set))))
        f.write("-----------------------------\n")
        f.write("Branches covered: {}\n".format(sorted(list(branches_covered_set))))
        f.write("Total branches covered: {}\n".format(len(list(branches_covered_set))))
        f.write("-----------------------------\n")
        f.write("All branches: {}\n".format(sorted(list(all_branches_set))))
        f.write("Total all branches: {}\n".format(len(list(all_branches_set))))
        f.write("-----------------------------\n")
        f.write("Branches not covered: {}\n".format(sorted(list(uncovered_branches_set))))
        f.write("Total branches not covered: {}\n".format(len(list(uncovered_branches_set))))
        f.write("-----------------------------\n")
        f.write("Total physical crashes: {}\n".format(total_physical_accidents))
        f.write("-----------------------------\n")
        f.write("Path Taken Length: {}\n".format(path_taken.count(",") + 1))
        f.write("Path Taken: {}\n".format(path_taken))
        f.close()

    # Close the program
    bng_obj.stop()

# If starting run the main function
if __name__ == '__main__':
    main()
