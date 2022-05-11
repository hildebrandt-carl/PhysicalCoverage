import sys
import math
import argparse

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import LineString
from scipy.stats import gaussian_kde

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

# Import the distributions
from general.RRS_distributions import linear_distribution
from general.RRS_distributions import center_close_distribution
from general.RRS_distributions import center_mid_distribution

# Import the kinematics
from general.environment_configurations import HighwayKinematics
from general.environment_configurations import BeamNGKinematics


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def create_outline(plt, ego_position, start_point):
    # Compute boundary points
    boundary_points = []

    # Draw the reach set
    boundary_points.append(ego_position)



    # Loop through all the points
    increment = 0.1
    for theta in np.arange(-kinematics.steering_angle - increment, kinematics.steering_angle + increment, increment):
        new_point = rotate(ego_position, start_point, theta)
        boundary_points.append(new_point)

    # Add the start point again
    boundary_points.append(ego_position)

    # Get the data and plot it
    x = [x[0] for x in boundary_points]
    y = [y[1] for y in boundary_points]

    # Draw lines between them
    for current_x, current_y in zip(zip(x, x[1:]), zip(y, y[1:])):
        plt.plot(current_x, current_y, c="blue")

    # Return the plt
    return plt



def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    """
    angle = math.radians(angle)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

parser = argparse.ArgumentParser()
parser.add_argument('--scenario',        type=str, default="",   help="beamng/highway")
parser.add_argument('--distribution',    type=str, default="",   help="linear/center_close/center_mid")
args = parser.parse_args()

print("----------------------------------")
print("-------Loading Distribution-------")
print("----------------------------------")

# Declare the limits
x_lim = [-1, 37]
y_lim = [-20, 20]

# Get the distribution
if args.distribution   == "linear":
    distribution  = linear_distribution(args.scenario)
elif args.distribution == "center_close":
    distribution  = center_close_distribution(args.scenario)
elif args.distribution == "center_mid":
    distribution  = center_mid_distribution(args.scenario)
else:
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the kinematics
if args.scenario   == "highway":
    kinematics  = HighwayKinematics()
elif args.scenario == "beamng":
    kinematics  = BeamNGKinematics()
else:
    print("ERROR: Unknown scenario ({})".format(args.scenario))
    exit()

print("Loading complete")

print("----------------------------------")
print("------Plotting Distribution-------")
print("----------------------------------")

predefined_points   = distribution.get_predefined_points()
sample_distribution = distribution.get_sample_distribution()
angle_distribution  = distribution.get_angle_distribution()

# Holds all the points used to generate heatmap
all_points = []

# Looping in RRS
for RRS in range(1,10):

    plt.figure(RRS, figsize=(4,3))

    # Define the ego_position and start_point
    ego_position = (0,0)
    start_point = (kinematics.max_velocity, 0)

    # Plot the boundary
    plt = create_outline(plt, ego_position, start_point)

    # Get the sample angles
    angles = angle_distribution[RRS]
    samples = sample_distribution[RRS]

    # For each angle
    for a, s in zip(angles, samples):
        # Create the sample 
        sample_x = [0]
        sample_y = [0]
        # Compute the angles
        sample_point = rotate(ego_position, start_point, a)
        # Append the sample
        sample_x.append(sample_point[0])
        sample_y.append(sample_point[1])

        # Show the line
        plt.plot(sample_x, sample_y, c="red")

        # For each line get the number of samples
        points = predefined_points[s]
        for p in points:
            # Compute where the new point is
            new_point = (p, 0)
            rotated_point = rotate(ego_position, new_point, a)
            plt.scatter(rotated_point[0], rotated_point[1], c="black")
            all_points.append(rotated_point)

    # Draw lines between each pair
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0] ,y_lim[1])

# Plot the overview
plt.figure("Final Overview")

# Define the ego_position and start_point
ego_position = (0,0)
start_point = (kinematics.max_velocity, 0)

# Plot the boundary
plt = create_outline(plt, ego_position, start_point)

# Plot all the points
x1 = [x[0] for x in all_points]
y1 = [y[1] for y in all_points]
x = []
y = []

highlighting = 250
for _ in range(highlighting):
    x = x + x1
    y = y + y1

# Create a uniform plot underneath everything
for i in np.arange(x_lim[0], x_lim[1], 1):
    for j in np.arange(y_lim[0], y_lim[1], 1):
        x.append(i)
        y.append(j)

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
x = np.array(x)
y = np.array(y)
z = np.array(z)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

# Plot the distribution
plt.scatter(x, y, c=z, s=200, edgecolor=['none'])
# cbar = plt.colorbar(label="Sample Distribution")
# v1 = np.linspace(z.min(), z.max(), 2, endpoint=True)
# cbar = plt.colorbar(label="Sample Distribution", ticks=v1)
# cbar.ax.set_yticklabels(['No Samples', 'Many Samples'])

# Draw lines between each pair
plt.xlim(x_lim[0], x_lim[1])
plt.ylim(y_lim[0] ,y_lim[1])
plt.show()