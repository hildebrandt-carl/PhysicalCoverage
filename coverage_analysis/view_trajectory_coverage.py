import sys
import math
import glob
import argparse

from tqdm import tqdm
from time import sleep

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from general.trajectory_coverage import crossed_line
from general.trajectory_coverage import load_driving_area
from general.trajectory_coverage import unison_shuffled_copies
from general.trajectory_coverage import load_improved_bounded_driving_area
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",          help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",                                               help="beamng/highway")
args = parser.parse_args()

# Declare drivable area
drivable_x = [-10, 10]
drivable_y = [-10, 10]

# Create the configuration classes
HK = HighwayKinematics()
NG = BeamNGKinematics()

# Save the kinematics and RRS parameters
if args.scenario == "highway":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "beamng":
    new_steering_angle  = NG.steering_angle
    new_max_distance    = NG.max_velocity
else:
    print("ERROR: Unknown scenario")
    exit()

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Max velocity:\t\t" + str(new_max_distance))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = ""
load_name += "_s" + str(new_steering_angle) 
load_name += "_b" + str('*') 
load_name += "_d" + str(new_max_distance) 
load_name += "_t" + str(args.number_of_tests)
load_name += ".npy"

# Checking the distribution
if not (args.distribution == "linear" or args.distribution == "center_close" or args.distribution == "center_mid"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '/mnt/extradrive3/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
position_file_names     = glob.glob(base_path + "ego_positions_*")
crash_file_names        = glob.glob(base_path + "crash_*")
stall_file_names        = glob.glob(base_path + "stall_*")

# Select one of the files (they are all the same)
for file_name in position_file_names:
    if "_b1_" in file_name:
        break

# Get the drivable area
drivable_x, drivable_y = load_driving_area(args.scenario)

# Compute the size of the drivable area
drivable_x_size = drivable_x[1] - drivable_x[0]
drivable_y_size = drivable_y[1] - drivable_y[0]
print("Loaded driving area")

# Load the vehicle positions
vehicle_positions = np.load(file_name)
print("Loaded data")

# Get all X Y and Z positions
x_positions = vehicle_positions[:,:,0]
y_positions = vehicle_positions[:,:,1]
z_positions = vehicle_positions[:,:,2]

# Remove all nans
x_positions[np.isnan(x_positions)] = sys.maxsize
y_positions[np.isnan(y_positions)] = sys.maxsize

# Convert each position into an index
index_x_array = np.round(x_positions, 0).astype(int) - drivable_x[0] -1
index_y_array = np.round(y_positions, 0).astype(int) - drivable_y[0] -1

# Declare the upper bound used to detect nan
upper_bound = sys.maxsize - 1e5

# Shuffle both arrays
index_x_array, index_y_array = unison_shuffled_copies(index_x_array, index_y_array)
print("Shuffling Data")

# Load the upper and lower bound of the driving area
bounds = load_improved_bounded_driving_area(args.scenario)
lower_bound_x, lower_bound_y, upper_bound_x, upper_bound_y = bounds

# Convert the bounds to indices
index_lower_bound_x = np.round(lower_bound_x, 0).astype(int) - drivable_x[0] -1
index_lower_bound_y = np.round(lower_bound_y, 0).astype(int) - drivable_y[0] -1
index_upper_bound_x = np.round(upper_bound_x, 0).astype(int) - drivable_x[0] -1
index_upper_bound_y = np.round(upper_bound_y, 0).astype(int) - drivable_y[0] -1

# Make sure these bounds are confined by available space
index_lower_bound_x = np.clip(index_lower_bound_x, 0, drivable_x_size-1)
index_lower_bound_y = np.clip(index_lower_bound_y, 0, drivable_y_size-1)
index_upper_bound_x = np.clip(index_upper_bound_x, 0, drivable_x_size-1)
index_upper_bound_y = np.clip(index_upper_bound_y, 0, drivable_y_size-1)
print("Loaded improved driving area")

# Creating the coverage array
print("----------------------------------")
print("---Creating Coverage Array Data---")
print("----------------------------------")
# 1  == Covered
# 0  == Not Covered
# -1 == Invalid
# -2 == Boarder
coverage_array = np.full((drivable_x_size, drivable_y_size), 0,  dtype=int)

if args.scenario == "beamng":
    # Loop through the coverage array
    for x in tqdm(range(0, drivable_x_size, 1)):

        # Set the current state (start with invalid)
        state = -1

        # Start saying we have not found lower
        lower_found = False

        for y in range(0, drivable_y_size, 1):
            # Subtract one to account that we start at 0
            y = y - 1

            # If the current index goes over the bounds change it
            if (state == -1) and (lower_found == False):
                # print("here1")

                # Check if we have crossed the line yet
                crossed = crossed_line([index_lower_bound_x, index_lower_bound_y], (x, y))

                # Find the closest point on the line to the current point
                closest_index = np.argmin(crossed)

                # Compare the two points
                comparison_point = (index_lower_bound_x[closest_index], index_lower_bound_y[closest_index])

                if comparison_point[1] < y:
                    state = 0
                    lower_found = True

            # If the current index goes over the bounds change it
            if (state == 0) and (lower_found == True):
                # print("here1")

                # Check if we have crossed the line yet
                crossed = crossed_line([index_upper_bound_x, index_upper_bound_y], (x, y))

                # Find the closest point on the line to the current point
                closest_index = np.argmin(crossed)

                # Compare the two points
                comparison_point = (index_upper_bound_x[closest_index], index_upper_bound_y[closest_index])

                if comparison_point[1] < y:
                    state = -1

            # Set all invalid areas properly
            coverage_array[x, y] = state

    for p in zip(index_lower_bound_x, index_lower_bound_y):
        coverage_array[p[0], p[1]] = -2

    for p in zip(index_upper_bound_x, index_upper_bound_y):
        coverage_array[p[0], p[1]] = -2

elif args.scenario == "highway":
    index_x_array = np.clip(index_x_array, 0, drivable_x_size-1)
    index_y_array = np.clip(index_y_array, 0, drivable_y_size-1)

print("Done")

print("----------------------------------")
print("--------Computing Coverage--------")
print("----------------------------------")

# Get the naive coverage 
naive_coverage_denominator = float(drivable_x_size * drivable_y_size)
improved_coverage_denominator = float(np.count_nonzero(coverage_array==0))

print("Naive denominator: {}".format(naive_coverage_denominator))
print("Improved denominator: {}".format(improved_coverage_denominator))

# Create the final coverage array
naive_coverage_percentage       = np.zeros(args.number_of_tests)
improved_coverage_percentage    = np.zeros(args.number_of_tests)

# Loop through the data and mark off all that we have seen
for i, coverage_index in tqdm(enumerate(zip(index_x_array, index_y_array)), total=len(index_x_array)):

    # Get the x and y index
    x_index = coverage_index[0]
    y_index = coverage_index[1]

    while ((abs(x_index[-1]) >= upper_bound) or (abs(y_index[-1]) >= upper_bound)):
        x_index = x_index[:-1]
        y_index = y_index[:-1]

    # Update the coverage array
    coverage_array[x_index, y_index] = 1

    # Compute the coverage
    naive_coverage_percentage[i]        = (np.count_nonzero(coverage_array==1) / naive_coverage_denominator) * 100
    improved_coverage_percentage[i]     = (np.count_nonzero(coverage_array==1) / improved_coverage_denominator) * 100

print("Naive final coverage: {}".format(naive_coverage_percentage[-1]))
print("Improved final coverage: {}".format(improved_coverage_percentage[-1]))

# Create the output figure
plt.figure(1, figsize=(10,5))
plt.title("Naive Coverage")
plt.plot(naive_coverage_percentage)

plt.figure(4, figsize=(10, 5))
plt.title("Improved Coverage")
plt.plot(improved_coverage_percentage)

plt.figure(2, figsize=(10,5))
plt.title("Coverage Map")
plt.imshow(coverage_array.transpose(), cmap='hot')

plt.show()