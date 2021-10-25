import os
import sys
import gym
import time
import argparse
import datetime
import highway_env_v2

import numpy as np
import matplotlib.pyplot as plt

# Hot fix to get general accepted
from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/highway")])
sys.path.append(base_directory)

from math import pi, atan2, degrees

from general.reachset import ReachableSet

from general.environment_configurations import RSRConfig
from general.environment_configurations import HighwayKinematics

from general.highway_config import HighwayEnvironmentConfig

from controllers.tracker import Tracker
from controllers.car_controller import EgoController
from controllers.traffic_controller import TrafficController

parser = argparse.ArgumentParser()
parser.add_argument('--test_name',      type=str, default="test",   help="The input and output name for the run")
parser.add_argument('--total_samples',  type=int, default=-1,       help="Describes the number of samples which were used to generate this set")
parser.add_argument('--total_beams',    type=int, default=-1,       help="The total number of beams")
parser.add_argument('--no_plot',        action='store_true')
args = parser.parse_args()

# Get the full file path
test_path = "../../PhysicalCoverageData/highway/generated_tests/tests_single/tests/{}/{}_beams/{}.npy".format(args.total_samples, args.total_beams, args.test_name)
index_path = test_path[:test_path.rfind("_")] + "_index.npy"

# Figure out how many beam this test was generated using
test_data = np.load(test_path)
beams = test_data.shape[1]

# Check that the given and read beam numbers are the same
if args.total_beams != beams:
    print("Error there is a difference between the given and read beam number")
    exit()

# Get the different configurations
HK = HighwayKinematics()
RSR = RSRConfig(beam_count=beams)

# Variables - Used for generating the reach set
total_lines     = RSR.beam_count
steering_angle  = HK.steering_angle
max_distance    = HK.max_velocity

# Declare the obstacle size (1 - car; 0.5 - motorbike)
obstacle_size = 1

# Decalare how many traffic vehicles there are
environment_vehicles = total_lines

# Create the output directory if it doesn't exists

save_path = "../output/highway/generated_tests/tests_single/raw/{}/{}_external_vehicles".format(args.total_samples, environment_vehicles)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create the save name
save_name = args.test_name + "_" + str(datetime.datetime.now().time())

# Save the output file
text_file = open("{}/{}.txt".format(save_path, save_name), "w")
text_file.write("Name: %s\n" % args.test_name)
e = datetime.datetime.now()
text_file.write("Date: %s/%s/%s\n" % (e.day, e.month, e.year))
text_file.write("Time: %s:%s:%s\n" % (e.hour, e.minute, e.second))
text_file.write("External Vehicles: %d\n" % environment_vehicles)
text_file.write("Reach set total lines: %d\n" % total_lines)
text_file.write("Reach set steering angle: %d\n" % steering_angle)
text_file.write("Reach set max distance: %d\n" % max_distance)
text_file.write("------------------------------\n")

# Suppress exponential notation
np.set_printoptions(suppress=True)

# Create the controllers
hw_config = HighwayEnvironmentConfig(environment_vehicles=0, controlled_vehicle_count=environment_vehicles + 1, duration=100, crash_ends_test=False)
car_controller = EgoController(debug=True)
tracker = Tracker(distance_threshold=5, time_threshold=2, debug=True)
reach = ReachableSet(obstacle_size=obstacle_size)

# Create the environment
env = gym.make("highway-v0")
env.config = hw_config.env_configuration
env.reset()

# Default action is IDLE
action = car_controller.default_action()

# Create the traffic controller
traffic = TrafficController(env.controlled_vehicles[1:], test_path, index_path)

# Main loop
done = False

# Keep track of the previous observation incase of a crash
obs = None

# Init timing variables
start_time = datetime.datetime.now()
simulated_time_counter = 0
simulated_time_period = 1.0 / hw_config.policy_freq

while not done:

    # Increment time
    simulated_time_counter += 1

    # Step the environment
    obs, reward, done, info = env.step(action)
    obs = np.round(obs, 4)

    # Update the traffic vehicles
    complete = traffic.compute_traffic_commands(env.controlled_vehicles[0].position, obstacle_size, info["collided"])

    # End if we are complete
    if complete:
        done = True
        continue

    # Print the observation and crash data
    print("Environment:")
    print("|--Crash: \t\t" + str(info["crashed"]))
    print("|--Collided: \t\t" + str(info["collided"]))
    print("|--Speed: \t\t" + str(np.round(info["speed"], 4)))
    print("|--Observation: \n" + str(obs))
    print("")

    # Get the next action based on the current observation
    action = car_controller.drive(obs)

    # Track objects
    tracker.track(obs)
    tracked_objects = tracker.get_observations()

    # Track the time for this operation
    op_start_time = datetime.datetime.now()

    lane_positions = None

    # Get the reach set simulation
    polygons    = reach.compute_environment(tracked_objects, lane_positions)
    r_set       = reach.estimate_raw_reachset(total_lines=total_lines, 
                                              steering_angle=steering_angle,
                                              max_distance=max_distance)
    final_r_set = reach.estimate_true_reachset(polygons, r_set)
    r_vector    = reach.vectorize_reachset(final_r_set, accuracy=0.001)

    # Track the time for this operation
    current_time = datetime.datetime.now()
    operation_time = (current_time - op_start_time).total_seconds()
    elapsed_time = (current_time - start_time).total_seconds()

    print("")
    print("Vector: " + str(r_vector))
    print("Goal: " + str(traffic.get_expected_index()))
    print("---------------------------------------")

    if not args.no_plot:
        plt.figure(1)
        plt.clf()
        plt.title('Environment')

        # Invert the y axis for easier viewing
        plt.gca().invert_yaxis()

        # Display the environment
        for i in range(len(polygons)):
            # Get the polygon
            p = polygons[i]
            x,y = p.exterior.xy
            # Get the color
            c = "g" if i == 0 else "r"
            # Plot
            plt.plot(x, y, color=c)

        # Display the goals we are trying to reach
        plt = traffic.display_goals(plt)

        # Display the reachset
        for i in range(len(r_set)):
            # Get the polygon
            p = r_set[i]
            x,y = p.xy
            # Get the color
            c = "r"
            # Plot
            plt.plot(x, y, color=c, alpha=0.5)

        # Display the reachset
        for i in range(len(final_r_set)):
            # Get the polygon
            p = final_r_set[i]
            x,y = p.xy
            # Get the color
            c = "g"
            # Plot
            plt.plot(x, y, color=c)

        # Set the size of the graph
        plt.xlim([-30, 30])
        plt.ylim([-30, 30])

        # Invert the y axis as negative is up and show ticks
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        # plot the graph
        plt.pause(0.1)

        # Render environment
        env.render()

    simulated_time = np.round(simulated_time_period * simulated_time_counter, 4)

    text_file.write("Vector: " + str(r_vector) + "\n")
    text_file.write("Ego Position: " + str(np.round(env.controlled_vehicles[0].position,4)) + "\n")
    text_file.write("Ego Velocity: " + str(np.round(env.controlled_vehicles[0].velocity, 4)) + "\n")
    text_file.write("Crash: " + str(info["crashed"]) + "\n")
    text_file.write("Collided: " + str(info["collided"]) + "\n")
    text_file.write("Operation Time: " + str(operation_time) + "\n")
    text_file.write("Total Wall Time: " + str(elapsed_time) + "\n")
    text_file.write("Total Simulated Time: " + str(simulated_time) + "\n")
    text_file.write("\n")
    
    # If it crashed determine under which conditions it crashed
    if info["collided"]:

        try:
            # Get the velocity of the two vehicles (we want the velocities just before we crashed)
            ego_vx, ego_vy = info["kinematic_history"]["velocity"][1]
            veh_vx, veh_vy = info["incident_vehicle_kinematic_history"]["velocity"][1]

            # Get magnitude of both velocity vectors
            ego_mag = np.linalg.norm([ego_vx, ego_vy])
            veh_mag = np.linalg.norm([veh_vx, veh_vy])

            # Get the angle of incidence
            angle_of_incidence = degrees(atan2(veh_vy, veh_vx) - atan2(ego_vy, ego_vx))

            # Round all values to 4 decimal places
            ego_mag = np.round(ego_mag, 4)
            veh_mag = np.round(veh_mag, 4)
            angle_of_incidence = np.round(angle_of_incidence, 4)
        except ValueError:
            ego_mag = 0
            veh_mag = 0 
            angle_of_incidence = 0

        print("Ego velocity magnitude: {}".format(ego_mag))
        print("Incident vehicle velocity magnitude: {}".format(veh_mag))
        print("Angle of incident: {}".format(angle_of_incidence))
        print("")
        print("---------------------------------------")
        text_file.write("Ego velocity magnitude: {}\n".format(ego_mag))
        text_file.write("Incident vehicle velocity magnitude: {}\n".format(veh_mag))
        text_file.write("Angle of incident: {}\n\n".format(angle_of_incidence))

env.close()
text_file.close()