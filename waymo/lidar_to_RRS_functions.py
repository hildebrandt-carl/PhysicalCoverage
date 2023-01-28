import ast
import math
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString

np.set_printoptions(suppress=True)

def create_frame_plot(data, origin, orientation, title, fig_num):
    fig = plt.figure(fig_num)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=5)
    ax.quiver(origin[0], origin[1], orientation[0], orientation[1])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title(title)
    plt.xlim([-175, 175])
    plt.ylim([-175, 175])
    return plt

def create_lidar_plot(data, title, x_range, y_range, fig_num):
    plt.figure(fig_num)
    plt.clf()
    plt.title(title)
    # Display the environment
    for i in range(len(data["polygons"])):
        # Get the polygon
        p = data["polygons"][i]
        x,y = p.exterior.xy
        # Get the color
        c = "g" if i == 0 else "r"
        # Plot
        plt.plot(x, y, color=c)
    # Display the reachset
    for i in range(len(data["r_set"])):
        # Get the polygon
        p = data["r_set"][i]
        x,y = p.xy
        # Get the color
        c = "r"
        # Plot
        plt.plot(x, y, color=c, alpha=0.5)
    # Display the reachset
    for i in range(len(data["final_r_set"])):
        # Get the polygon
        p = data["final_r_set"][i]
        x,y = p.xy
        # Get the color
        c = "g"
        # Plot
        plt.plot(x, y, color=c)
    # Set the size of the graph
    plt.xlim(x_range)
    plt.ylim(y_range)
    # Invert the y axis as negative is up and show ticks
    ax = plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    return plt

def getStep(a, MinClip):
    return round(float(a) / MinClip) * MinClip

def combine_environment_and_reachset(r_set, polygons):
    # Check if any of the reach set intersects with the points
    final_r_set = []
    # For each line
    for l in r_set:
        # Get the origin
        origin = l.coords[0]
        end_position = l.coords[1]
        min_distance = Point(origin).distance(Point(end_position))
        min_point = end_position
        # For each polygon (except the ego vehicle)
        for p in polygons[1:]:
            # Check if they intersect and if so where
            intersect = l.intersection(p)
            if not intersect.is_empty:
                for i in intersect.coords:
                    # Check which distance is the closest
                    dis = Point(origin).distance(Point(i))
                    if dis < min_distance:
                        min_distance = dis
                        min_point = i
                            
        # Update the line
        true_l = LineString([origin, min_point])
        final_r_set.append(true_l)
    return final_r_set

def estimate_reachset(ego_position, steering_angle, total_lines, max_distance):
    # Estimate the reach set
    ego_heading = 0
    r_set = []
    # Convert steering angle to radians
    steering_angle_rad = math.radians(steering_angle)
    # Compute the intervals 
    intervals = 0
    if total_lines > 1:
        intervals = (steering_angle_rad * 2) / float(total_lines - 1)
    # Create each line
    for i in range(total_lines):
        # Compute the angle of the beam
        if total_lines > 1:
            theta = (-1 * steering_angle_rad) + (i * intervals) +  ego_heading
        else:
            theta = ego_heading
        # Compute the new point
        p2 = (ego_position[0] + (max_distance * math.cos(theta)), ego_position[1] + (max_distance * math.sin(theta)))
        # Save the linestring
        l = LineString([ego_position, p2])
        r_set.append(l)
    return r_set

def estimate_obstacles(ego_vehicle, current_lidar_data):
    # Create a list of all polygons for plotting
    polygons = [ego_vehicle]

    # Turn all readings into small polygons
    for pi in range(len(current_lidar_data)):
        # Get the point
        p = current_lidar_data[pi]
        s = 0.2
        new_point = Polygon([(p[0]-s, p[1]-s),
                            (p[0]+s, p[1]-s),
                            (p[0]+s, p[1]+s),
                            (p[0]-s, p[1]+s)])
        polygons.append(new_point)
    return polygons

def vectorize_reachset(lines, accuracy=0.25):
    vector = []
    # For each line:
    for l in lines:
        l_len = l.length
        l_len = getStep(l_len, accuracy)
        l_len = round(l_len, 6)
        vector.append(l_len)
    return vector

def rotate(p, origin=(0, 0), angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)
