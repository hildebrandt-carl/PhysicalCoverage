import gym
import time
import highway_env_v2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from highway_config import HighwayConfig
from car_controller import EgoController
from physical_analysis import PhysicalAnalysis

# Suppress exponetial notation
np.set_printoptions(suppress=True)

look_ahead_distance = 30
recording = True

# Create the controllers
hw_config = HighwayConfig()
car_controller = EgoController(debug=True)
physical_analysis = PhysicalAnalysis(distance_threshold=5, time_threshold=2, debug=True)

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
    tracked_objects = physical_analysis.track(obs)
    physical_analysis.compute_trajectories(history_size=3, look_ahead_distance=look_ahead_distance, steering_angle=5)
    intersection_data = physical_analysis.compute_intersection(look_ahead_distance)
    G, node_pos, node_colors = physical_analysis.compute_analysis_graph()

    # Plot the results
    plt.figure(1)
    plt.clf()
    plt.title('Trajectory Viewer')

    # Invert the y axis for easier viewing
    ax = plt.gca()
    ax.invert_yaxis()

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, node_pos, node_size=500, node_color=node_colors)
    # nx.draw_networkx_edges(G, node_pos, width=3, alpha=0.5, edge_color="b", style="dashed")

    # Draw the trajectories
    for obj in tracked_objects:
        p1_vel, p2_vel = obj.get_trajectory_velocity()
        
    # Show each of the lines
    for obj in tracked_objects:
        reach = obj.reachable_set
        p1, p2 = obj.get_trajectory_points(look_ahead_distance)
        p1_vel, p2_vel = obj.get_trajectory_velocity()
        if ((p1_vel is not None) and (p2_vel is not None) and (reach is not None)):
            # Assume no intersection
            c = 'g'
            # If this line intersects with another line
            for intersection in intersection_data:
                if (intersection['intersection_percentage'] > 0) and obj.obj_id in intersection['id']:
                    c = 'r'

            # Plot the reachable set
            ax.fill(*reach.exterior.xy, alpha=0.5, fc=c , ec='none')

            # Plot the original distance line
            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]
            plt.plot(x, y, color=c)

        if ((p1_vel is not None) and (p2_vel is not None)):
            plt.annotate(text='', xy=p1_vel, xytext=p2_vel, arrowprops=dict(arrowstyle='<-', color=c, lw=3.5))

    # Label each of the nodes
    nx.draw_networkx_labels(G, node_pos)

    # Set the size of the graph
    plt.xlim([-100,100])
    plt.ylim([-14, 14])

    # Invert the y axis as negative is up and show ticks
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.figure(2)
    plt.clf()
    plt.title('Physical Stack')

    # Invert the y axis for easier viewing
    plt.gca().invert_yaxis()

    # Add edges between any two nodes which intersect
    for intersection in intersection_data:
        if intersection['intersection_percentage'] > 0:
            paid_id = intersection['id']
            if paid_id[0] != 1:
                n1 = str(paid_id[0])
            else:
                n1 = "ego"
            n2 = str(paid_id[1])
            G.add_edge(n1, n2, weight=intersection['intersection_percentage'])

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, node_pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, node_pos, width=4, alpha=0.5, edge_color="r")
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, node_pos, edge_labels=labels, bbox=dict(alpha=0.0))

    # Label each of the nodes
    nx.draw_networkx_labels(G, node_pos)

    # Set the size of the graph
    plt.xlim([-100,100])
    plt.ylim([-14, 14])

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