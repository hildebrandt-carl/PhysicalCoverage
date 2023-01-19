import sys
import math
import glob
import hashlib
import argparse

from tqdm import tqdm
from time import sleep

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from general.file_functions import unison_shuffled_copies
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",          help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",                                               help="beamng/highway")
parser.add_argument('--cores',              type=int, default=4,                                                help="number of available cores")
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

# Load the vehicle positions
vehicle_positions = np.load(file_name)

# Select one of the files (they are all the same)
for file_name in crash_file_names:
    if "_b1_" in file_name:
        break

crashes = np.load(file_name, allow_pickle=True)

# Get all X Y and Z positions
x_positions = vehicle_positions[:,:,0]
y_positions = vehicle_positions[:,:,1]
z_positions = vehicle_positions[:,:,2]

# Get the min and max of the X and Y
min_x = np.nanmin(x_positions)
max_x = np.nanmax(x_positions)
min_y = np.nanmin(y_positions)
max_y = np.nanmax(y_positions)

# For now use these as the drivable area
drivable_x = [int(math.floor(min_x)), int(math.ceil(max_x))]
drivable_y = [int(math.floor(min_y)), int(math.ceil(max_y))]

# Compute the size of the drivable area
x_size = drivable_x[1] - drivable_x[0]
y_size = drivable_y[1] - drivable_y[0]

# Create the coverage boolean array
coverage_array = np.zeros((x_size, y_size), dtype=bool)
coverage_denominator = float(x_size * y_size)

# Remove all nans
x_positions[np.isnan(x_positions)] = sys.maxsize
y_positions[np.isnan(y_positions)] = sys.maxsize

# Convert each position into an index
index_x_array = np.round(x_positions, 0).astype(int) - drivable_x[0] -1
index_y_array = np.round(y_positions, 0).astype(int) - drivable_y[0] -1

# Create the final coverage array
final_coverage = np.zeros(args.number_of_tests)

# Declare the upper bound used to detect nan
upper_bound = sys.maxsize - 1e5

# Shuffle both arrays
index_x_array, index_y_array = unison_shuffled_copies(index_x_array, index_y_array)

# Turn all the signatures into a list
all_signatures = np.full(args.number_of_tests, None, dtype="object")
all_crash_detections = np.zeros(args.number_of_tests)

# Loop through the data and mark off all that we have seen
for i, coverage_index in tqdm(enumerate(zip(index_x_array, index_y_array)), total=len(index_x_array)):

    # Get the x and y index
    x_index = coverage_index[0]
    y_index = coverage_index[1]

    while ((abs(x_index[-1]) >= upper_bound) or (abs(y_index[-1]) >= upper_bound)):
        x_index = x_index[:-1]
        y_index = y_index[:-1]

    # The signature for the trace is the set of all RRS signatures
    position_list = []
    for j in range(len(x_index)):
        x = x_index[j]
        y = y_index[j]
        position_list.append(tuple((x, y)))

    # Make sure we remove any duplicates
    position_set = set(position_list)

    # Create the hash of the signature for each comparison
    position_string = str(tuple(sorted(position_list)))

    # Compute the hash
    position_hash = hashlib.md5(position_string.encode()).hexdigest()

    # Check if a crash was detected
    crash = crashes[i]
    crash_detected = not (crash == None).all()

    # Save the data
    all_signatures[i] = position_hash
    all_crash_detections[i] = crash_detected










# Print out the number of unique signatures
count_of_signatures = Counter(all_signatures)

# Get the signatures and the count
final_signatures, count_of_signatures = zip(*count_of_signatures.items())
count_of_signatures = np.array(count_of_signatures)

# Determine how many classes have more than 1 test
total_multiclasses = np.sum(count_of_signatures >= 2)
consistent_class = np.zeros(total_multiclasses, dtype=bool)

# Loop through each of the final signatures
count_index = 0
for i in range(len(final_signatures)):
    # Get the signature and count
    current_sig = final_signatures[i]
    current_count = count_of_signatures[i]

    if current_count <= 1:
        continue

    # Loop through the signatures and get the indices where this signature is in the array
    interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
    assert(len(interested_indices) == current_count)

    # Get all the crash data for a specific signature
    single_class_crash_data = all_crash_detections[interested_indices]

    # Check if all the data is consistent
    consistent = np.all(single_class_crash_data == single_class_crash_data[0])
    consistent_class[count_index] = bool(consistent)
    count_index += 1

# Final signatures holds the list of all signatures
# Count of signatures holds the list integers representing how many times each signature was seen

# Get the total signatures
total_signatures_count = len(final_signatures)

# Get the total number of single test and multitest signatures
single_test_signatures_count    = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
multi_test_signatures_count     = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

# Get the total number of consistent vs inconsistent classes
consistent_class_count      = np.count_nonzero(consistent_class)
inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

# Compute the percentage of consistency
if np.size(consistent_class) <= 0:
    percentage_of_inconsistency = 0
else:
    percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

# Make sure that there is no count where the count is < 1: Make sure that single + multi == total
assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

print([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])