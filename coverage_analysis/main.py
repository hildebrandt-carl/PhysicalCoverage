import gym
import time
import argparse
import datetime
import highway_env_v2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from highway_config import HighwayConfig
from car_controller import EgoController
from tracker import Tracker
from reachset import ReachableSet

parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default="output.txt", help="The save name of the run")
parser.add_argument('--environment_vehicles', type=int, default=20, help="total_number of vehicles in the environment")
parser.add_argument('--no_plot', action='store_true')
args = parser.parse_args()

# Save the output file
text_file = open("../output/" + args.save_name, "w")
text_file.write("Name: %s\n" % args.save_name)
e = datetime.datetime.now()
text_file.write("Date: %s/%s/%s\n" % (e.day, e.month, e.year))
text_file.write("Time: %s:%s:%s\n" % (e.hour, e.minute, e.second))
text_file.write("External Vehicles: %d\n" % args.environment_vehicles)
text_file.write("------------------------------\n")

# Suppress exponetial notation
np.set_printoptions(suppress=True)

# Create the controllers
hw_config = HighwayConfig(environment_vehicles=args.environment_vehicles)
car_controller = EgoController(debug=True)
tracker = Tracker(distance_threshold=5, time_threshold=2, debug=True)
reach = ReachableSet()

# Create the environment
env = gym.make("highway-v0")
env.config = hw_config.env_configuration
env.reset()

# Default action is IDLE
action = car_controller.default_action()

# Main loop
done = False
while not done:

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

    # Get the reach set simulation
    polygons = reach.compute_environment(tracked_objects)
    r_set = reach.estimate_raw_reachset()
    final_r_set = reach.estimate_true_reachset(polygons, r_set)
    r_vector = reach.vectorize_reachset(final_r_set)

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
            plt.plot(x,y,color=c)

        # Display the reachset
        for i in range(len(r_set)):
            # Get the polygon
            p = r_set[i]
            x,y = p.xy
            # Get the color
            c = "r"
            # Plot
            plt.plot(x,y,color=c, alpha=0.5)

        # Display the reachset
        for i in range(len(final_r_set)):
            # Get the polygon
            p = final_r_set[i]
            x,y = p.xy
            # Get the color
            c = "g"
            # Plot
            plt.plot(x,y,color=c)

        # Set the size of the graph
        plt.xlim([-30,30])
        plt.ylim([-30, 30])

        # Invert the y axis as negative is up and show ticks
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        plt.pause(0.1)

        # Render environment
        env.render()

    print("Vector: " + str(r_vector))
    text_file.write("Vector: " + str(r_vector) + "\n")

    print("---------------------------------------")

text_file.close()