import os
import gym
import time
import argparse
import datetime
import highway_env_v2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from misc.tracker import Tracker
from misc.reachset import ReachableSet
from misc.car_controller import EgoController
from misc.highway_config import HighwayKinematics
from misc.traffic_controller import TrafficController
from misc.highway_config import HighwayEnvironmentConfig
from misc.highway_config import RSRConfig

parser = argparse.ArgumentParser()
parser.add_argument('--test_name',      type=str, default="test",   help="The input and output name for the run")
parser.add_argument('--total_samples',  type=int, default=-1,       help="Describes the number of samples which were used to generate this set")
parser.add_argument('--total_beams',    type=int, default=-1,       help="The total number of beams")
parser.add_argument('--no_plot',        action='store_true')
args = parser.parse_args()

# Get the full file path
test_path = "../../PhysicalCoverageData/highway/unseen/{}/tests/{}_beams/{}.npy".format(args.total_samples, args.total_beams, args.test_name)

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

# Decalare how many traffic vehicles there are
environment_vehicles = total_lines

if not os.path.exists('output/{}/results/{}_beams'.format(args.total_samples, beams)):
    os.makedirs('output/{}/results/{}_beams'.format(args.total_samples, beams))

# Create the save name
save_name = args.test_name + "_" + str(datetime.datetime.now().time())

# Save the output file
text_file = open('output/{}/results/{}_beams/{}.txt'.format(args.total_samples, beams, save_name), "w")
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
hw_config = HighwayEnvironmentConfig(environment_vehicles=0, controlled_vehicle_count=environment_vehicles + 1)
car_controller = EgoController(debug=True)
tracker = Tracker(distance_threshold=5, time_threshold=2, debug=True)
reach = ReachableSet()

# Create the environment
env = gym.make("highway-v0")
env.config = hw_config.env_configuration
env.reset()

# Default action is IDLE
action = car_controller.default_action()

# Create the traffic controller
traffic = TrafficController(env.controlled_vehicles[1:], test_path)

# Get the roadway - used when calculating the edge of the road
lanes = env.road.network.graph['0']['1']
lane_width = np.array([0, lanes[0].width/2.0])

# Main loop
done = False
first = True
while not done:

    # Step the environment
    obs, reward, done, info = env.step(action)
    obs = np.round(obs, 4)

    # Update the traffic vehicles
    complete = traffic.compute_traffic_commands(env.controlled_vehicles[0].position)

    # End if we are complete
    if complete:
        done = True
        continue

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

    # Track the time for this opperation
    start_time = datetime.datetime.now()

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

    # Track the time for this opperation
    current_time = datetime.datetime.now()
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

    text_file.write("Vector: " + str(r_vector) + "\n")
    text_file.write("Crash: " + str(info["crashed"]) + "\n")
    text_file.write("Time: " + str(elapsed_time) + "\n")
    text_file.write("\n")

text_file.close()