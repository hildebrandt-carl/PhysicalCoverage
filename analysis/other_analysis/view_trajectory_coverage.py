import sys
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

from utils.trajectory_coverage import load_driving_area
from utils.trajectory_coverage import create_coverage_array
from utils.trajectory_coverage import compute_trajectory_coverage
from utils.trajectory_coverage import load_improved_bounded_driving_area
from utils.environment_configurations import BeamNGKinematics
from utils.environment_configurations import HighwayKinematics


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",          help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="center_close/center_full")
parser.add_argument('--scenario',           type=str, default="",                                               help="beamng/highway")
parser.add_argument('--random_test_suites', type=int, default=10,                                               help="The number of random line samples used")
args = parser.parse_args()

# Declare drivable area
drivable_x = [-10, 10]
drivable_y = [-10, 10]

# Create the configuration classes
HK = HighwayKinematics()
BK = BeamNGKinematics()

# Save the kinematics and RRS parameters
if args.scenario == "highway":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "beamng":
    new_steering_angle  = BK.steering_angle
    new_max_distance    = BK.max_velocity
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
if not (args.distribution == "center_close" or args.distribution == "center_full"):
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

# Load the upper and lower bound of the driving area
if args.scenario == "beamng":
    print("Loading improved driving area")
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
elif args.scenario == "highway":
    index_lower_bound_x = None
    index_lower_bound_y = None
    index_upper_bound_x = None
    index_upper_bound_y = None
else:
    print("Scenario not known")
    exit()


# Creating the coverage array
print("----------------------------------")
print("---Creating Coverage Array Data---")
print("----------------------------------")
# 1  == Covered
# 0  == Not Covered
# -1 == Invalid
# -2 == Boarder
coverage_array, index_x_array, index_y_array = create_coverage_array(args.scenario, drivable_x_size, drivable_y_size, index_lower_bound_x, index_lower_bound_y, index_upper_bound_x, index_upper_bound_y, index_x_array, index_y_array)
print("Done")

print("----------------------------------")
print("--------Computing Coverage--------")
print("----------------------------------")

naive_coverage_percentage, improved_coverage_percentage, coverage_array = compute_trajectory_coverage(coverage_array, args.random_test_suites, args.number_of_tests, index_x_array, index_y_array, drivable_x_size, drivable_y_size)

print("Naive final coverage: {}".format(naive_coverage_percentage[-1]))
print("Improved final coverage: {}".format(improved_coverage_percentage[-1]))

naive_average_coverage          = np.average(naive_coverage_percentage, axis=0)
naive_upper_bound_coverage      = np.max(naive_coverage_percentage, axis=0)
naive_lower_bound_coverage      = np.min(naive_coverage_percentage, axis=0)
improved_average_coverage       = np.average(improved_coverage_percentage, axis=0)
improved_upper_bound_coverage   = np.max(improved_coverage_percentage, axis=0)
improved_lower_bound_coverage   = np.min(improved_coverage_percentage, axis=0)

# Create the coverage plot
plt.figure(1, figsize=(10,5))
plt.title("Coverage")
x = np.arange(0, len(naive_average_coverage))
# Plot the results
plt.fill_between(x, naive_lower_bound_coverage, naive_upper_bound_coverage, alpha=0.2, color="black") #this is the shaded error
plt.plot(x, naive_average_coverage, c="C0", label="Naive Traj Cov", linestyle="dashed") #this is the line itself
plt.fill_between(x, improved_lower_bound_coverage, improved_upper_bound_coverage, alpha=0.2, color="black") #this is the shaded error
plt.plot(x, improved_average_coverage, c="C0", label="Improved Traj Cov", linestyle="dotted") #this is the line itself
plt.legend()

plt.figure(2, figsize=(10,5))
plt.title("Coverage Map")
plt.imshow(coverage_array.transpose(), cmap='hot')

plt.show()