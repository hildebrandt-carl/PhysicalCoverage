import os
import sys
import gym
import time
import argparse
import datetime
import highway_env_v2
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from math import pi, atan2, degrees

# Hot fix to get general accepted
from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/highway")])
sys.path.append(base_directory)

from general.reachset import ReachableSet
from general.highway_config import RSRConfig
from general.highway_config import HighwayKinematics
from general.highway_config import HighwayEnvironmentConfig

from controllers.tracker import Tracker
from controllers.car_controller import EgoController


parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default="output.txt", help="The save name of the run")
parser.add_argument('--environment_vehicles', type=int, default=10, help="total_number of vehicles in the environment")
parser.add_argument('--no_plot', action='store_true')
args = parser.parse_args()

# Get the different configurations
HK = HighwayKinematics()
RSR = RSRConfig(beam_count=30)

# Variables - Used for timing
total_lines     = RSR.beam_count
steering_angle  = HK.steering_angle
max_distance    = HK.max_velocity

# Declare the obstacle size (1 - car; 0.5 - motorbike)
obstacle_size = 1

# Create the output directory if it doesn't exists
if not os.path.exists('../output/run_random_scenarios/{}_external_vehicles'.format(args.environment_vehicles)):
    os.makedirs('../output/run_random_scenarios/{}_external_vehicles'.format(args.environment_vehicles))

# Save the output file
text_file = open("../output/run_random_scenarios/{}_external_vehicles/{}".format(args.environment_vehicles,args.save_name), "w")
text_file.write("Name: %s\n" % args.save_name)
e = datetime.datetime.now()
text_file.write("Date: %s/%s/%s\n" % (e.day, e.month, e.year))
text_file.write("Time: %s:%s:%s\n" % (e.hour, e.minute, e.second))
text_file.write("External Vehicles: %d\n" % args.environment_vehicles)
text_file.write("Reach set total lines: %d\n" % total_lines)
text_file.write("Reach set steering angle: %d\n" % steering_angle)
text_file.write("Reach set max distance: %d\n" % max_distance)
text_file.write("------------------------------\n")

# Suppress exponential notation
np.set_printoptions(suppress=True)

# Create the controllers
hw_config = HighwayEnvironmentConfig(environment_vehicles=args.environment_vehicles, duration=20)
car_controller = EgoController(debug=True)
tracker = Tracker(distance_threshold=5, time_threshold=2, debug=True)
reach = ReachableSet(obstacle_size=obstacle_size)

# Create the environment
env = gym.make("highway-v0")
env.config = hw_config.env_configuration
env.reset()

# Default action is IDLE
action = car_controller.default_action()

# Get the roadway - used when calculating the edge of the road
lanes = env.road.network.graph['0']['1']
lane_width = np.array([0, lanes[0].width/2.0])

# Main loop
done = False

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

    # Print the observation and crash data
    print("Environment:")
    print("|--Crash: \t\t" + str(info["crashed"]))
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

    # Convert the lane positions to be relative to the ego_vehicle
    ego_position = env.controlled_vehicles[0].position
    upperlane = [lanes[0].start-lane_width, lanes[0].end-lane_width] - ego_position
    lowerlane = [lanes[-1].start+lane_width, lanes[-1].end+lane_width] - ego_position
    lane_positions = [upperlane, lowerlane]

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
    text_file.write("Crash: " + str(info["crashed"]) + "\n")
    text_file.write("Operation Time: " + str(operation_time) + "\n")
    text_file.write("Total Wall Time: " + str(elapsed_time) + "\n")
    text_file.write("Total Simulated Time: " + str(simulated_time) + "\n")
    text_file.write("\n")

    # If it crashed determine under which conditions it crashed
    if info["crashed"]:

        try:
            # Get the velocity of the two vehicles (we want the velocities just before we crashed)
            ego_vx, ego_vy = info["kinematic_history"]["velocity"][1]
            veh_vx, veh_vy = info["incident_vehicle_kinematic_history"]["velocity"][1]

            # Get magnitude of both velocity vectors
            ego_mag = np.linalg.norm([ego_vx, ego_vy])
            veh_mag = np.linalg.norm([veh_vx, veh_vy])

            # Get the angle of incidence
            angle_of_incidence = degrees(atan2(veh_vy, veh_vx) - atan2(ego_vy, ego_vx))
        except ValueError:
            ego_mag = 0
            veh_mag = 0 
            angle_of_incidence = 0

        print("Ego velocity magnitude: {}".format(ego_mag))
        print("Incident vehicle velocity magnitude: {}".format(veh_mag))
        print("Angle of incident: {}".format(angle_of_incidence))
        text_file.write("Ego velocity magnitude: {}\n".format(ego_mag))
        text_file.write("Incident vehicle velocity magnitude: {}\n".format(veh_mag))
        text_file.write("Angle of incident: {}\n".format(angle_of_incidence))

env.close()
text_file.close()