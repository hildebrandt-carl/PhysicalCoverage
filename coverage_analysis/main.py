import gym
import time
import highway_env_v2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from highway_config import HighwayConfig
from car_controller import EgoController
from tracker import Tracker
from reachset import ReachableSet

# Suppress exponetial notation
np.set_printoptions(suppress=True)

recording = False

# Create the controllers
hw_config = HighwayConfig()
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

    print("Vector: " + str(r_vector))

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

    if recording == True:
        time.sleep(10)
        recording = False

    print("---------------------------------------")