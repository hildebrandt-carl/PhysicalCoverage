import os
import gym
import sys
import time
import argparse
import datetime
import highway_env_v2
import numpy as np
import matplotlib.pyplot as plt

from math import radians

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
parser.add_argument('--total_lines', type=int, default="3", help="The number of beams")
parser.add_argument('--no_plot', action='store_true')
args = parser.parse_args()

# Get the different configurations
HK = HighwayKinematics()
RSR = RSRConfig(beam_count=args.total_lines)

# Variables - Used for timing
total_lines     = RSR.beam_count
steering_angle  = HK.steering_angle
max_distance    = HK.max_velocity

# Create the output directory if it doesn't exists
if not os.path.exists('../output/feasibility/raw'):
    os.makedirs('../output/feasibility/raw')

# Declare how accurate you want it
total_headings = 20
total_positions = 20

total_headings = max(3, total_headings)
max_heading = HK.steering_angle # Degrees

# Decalare how many traffic vehicles there are
environment_vehicles = 0

# Save the output file
text_file = open("../output/feasibility/raw/feasible_vectors{}.txt".format(total_lines), "w")
text_file.write("Name: %s\n" % "Feasible vector generation")
e = datetime.datetime.now()
text_file.write("Date: %s/%s/%s\n" % (e.day, e.month, e.year))
text_file.write("Time: %s:%s:%s\n" % (e.hour, e.minute, e.second))
text_file.write("External Vehicles: %d\n" % environment_vehicles)
text_file.write("Reach set total lines: %d\n" % total_lines)
text_file.write("Reach set steering angle: %d\n" % steering_angle)
text_file.write("Reach set max distance: %d\n" % max_distance)
text_file.write("Total positions: %d\n" % total_positions)
text_file.write("Total headings: %d\n" % total_headings)
text_file.write("Max heading: %d\n" % max_heading)
text_file.write("------------------------------\n")

for i in range(total_positions):

    # Create the starting position of the vehicle
    increment = 12.0 / (total_positions - 1)
    start_pos = np.round([100, i * increment], 4)
    print("Starting position {}: {}".format(i, start_pos))

    # Compute the orientation
    for j in range(total_headings):

        heading_range = max_heading * 2.0
        heading_increments = heading_range / float(total_headings - 1)
        current_heading = (-1.0 * max_heading) + (j * heading_increments)
        heading_range = max_heading * 2.0
        print("Starting heading {}: {}".format(j, current_heading))

        # Suppress exponential notation
        np.set_printoptions(suppress=True)

        # Create the controllers
        hw_config = HighwayEnvironmentConfig(environment_vehicles=0, controlled_vehicle_count=environment_vehicles + 1, ego_position=start_pos, ego_heading=radians(current_heading))
        tracker = Tracker(distance_threshold=5, time_threshold=2, debug=True)
        reach = ReachableSet()

        # Create the environment
        env = gym.make("highway-v0")
        env.config = hw_config.env_configuration
        env.reset()

        # Get the roadway
        lanes = env.road.network.graph['0']['1']
        lane_width = np.array([0, lanes[0].width/2.0])
        upperlane = [lanes[0].start-lane_width, lanes[0].end-lane_width]
        lowerlane = [lanes[-1].start+lane_width, lanes[-1].end+lane_width]

        # Convert the lane positions to be relative to the ego_vehicle
        ego_position = env.controlled_vehicles[0].position
        upperlane -= ego_position
        lowerlane -= ego_position
        lane_positions = [upperlane, lowerlane]

        # Step the environment
        obs = env.observation_type.observe()
        obs = np.round(obs, 4)

        # Track objects
        tracker.track(obs)
        tracked_objects = tracker.get_observations()

        # Track the time for this opperation
        start_time = datetime.datetime.now()


        # Get the reach set simulation
        polygons    = reach.compute_environment(tracked_objects, lane_positions)
        r_set       = reach.estimate_raw_reachset(total_lines=total_lines, 
                                                  steering_angle=steering_angle,
                                                  max_distance=max_distance)
        final_r_set = reach.estimate_true_reachset(polygons, r_set)
        r_vector    = reach.vectorize_reachset(final_r_set, accuracy=0.001)

        # Track the time for this opperation
        current_time = datetime.datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()

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

        text_file.write("Vector: " + str(r_vector) + "\n")
        text_file.write("Crash: " + str("False") + "\n")
        text_file.write("Time: " + str(elapsed_time) + "\n")
        text_file.write("\n")

text_file.close()
