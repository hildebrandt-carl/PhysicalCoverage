import sys
import math
import random
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/test_generation")])
sys.path.append(base_directory)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_unseen_tests_functions import get_trace_files
from generate_unseen_tests_functions import compute_coverage
from generate_unseen_tests_functions import compute_unseen_vectors
from generate_unseen_tests_functions import compute_reach_set_details
from generate_unseen_tests_functions import create_image_representation
from generate_unseen_tests_functions import save_unseen_data_to_file_single

from general.environment_configurations import RSRConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',          type=int, default=-1,       help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',               type=str, default="",       help="beamng/highway")
parser.add_argument('--cores',                  type=int, default=4,        help="number of available cores")
parser.add_argument('--maximum_unseen_vectors', type=int, default=1000000,  help="The maximum number of unseen vectors")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
NG = BeamNGKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_accuracy        = RSR.accuracy
new_total_lines     = "*"

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
print("Total beams:\t\t" + str(new_total_lines))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/physical_coverage/processed/' + str(args.total_samples) + "/"
file_names = get_trace_files(base_path)

print()
print("Files: " + str(file_names))
print("Loading Complete")

print("----------------------------------")
print("-------Computing Coverage---------")
print("----------------------------------")

manager = multiprocessing.Manager()
return_dict = manager.dict()

# Create a pool with x processes
total_processors = int(args.cores)
pool =  multiprocessing.Pool(total_processors)

# Call our function total_test_suites times
result_object = []
for file_index in range(len(file_names)):

    # Get the file name and the return key
    file_name = file_names[file_index]
    return_key = 'p' + str(file_index)

    if "_b10_" in file_name:
        continue

    if "_b9_" in file_name:
        continue

    if "_b8_" in file_name:
        continue

    if "_b7_" in file_name:
        continue

    result_object.append(pool.apply_async(compute_coverage, args=(file_name, return_dict, return_key, base_path, new_max_distance, new_accuracy)))

# Get the results (results are actually stored in return_dict)
results = [r.get() for r in result_object]
results = np.array(results)

# Convert the return_dict results to a normal dictionary
final_results = {}
for key in return_dict.keys():
    final_results[key] = list(return_dict[key])

# Close the pool
pool.close()

# Compute the number of unique observations per cell
unique_observations_per_cell = (new_max_distance / float(new_accuracy))
unique_values_per_beam = [new_accuracy]
for i in range(math.ceil(unique_observations_per_cell) - 1):
    unique_values_per_beam.append(unique_values_per_beam[-1] + new_accuracy)

print("\n\n\n--------------------------------------------------------")

# Compute the unseen vectors
final_data = compute_unseen_vectors(return_dict, args.maximum_unseen_vectors, new_max_distance, unique_values_per_beam, new_accuracy)

keys = final_data.keys()
keys = list(keys)
keys = sorted(keys)

# Plot the data
for key in keys:
    print(key)
    # Get the seen and unseen data
    seen_data       = final_data[key]["seen"]
    unseen_data     = final_data[key]["unseen"]
    total_lines     = final_data[key]["total_beams"]
    max_distance    = final_data[key]["max_distance"]
    steering_angle  = final_data[key]["steering_angle"]

    # Print what we are plotting
    print("----------------------------------------")
    print("Generating tests with {} beams".format(total_lines))

    # Remove the seen or unseen columns from the data frames
    del seen_data['Name']
    del unseen_data['Name']

    # Convert the unseen data to numpy data:
    seen_data_np = seen_data.to_numpy()
    unseen_data_np = unseen_data.to_numpy()

    print("seen_data Shape: {}".format(seen_data_np.shape))
    print("unseen_data Shape: {}".format(unseen_data_np.shape))
    print("total_lines: {}".format(total_lines))
    print("max_distance: {}".format(max_distance))
    print("steering_angle: {}".format(steering_angle))

    # Load the feasible trajectories
    fname = '../../PhysicalCoverageData/' + str(args.scenario) +'/feasibility/processed/FeasibleVectors_b' + str(total_lines) + ".npy"
    feasible_vectors = np.load(fname)
    feasible_vector_set = set()
    for v in feasible_vectors:
        feasible_vector_set.add(tuple(v))

    # Remove all unseen data that is not part of the feasible data set, as we are not interested in things that are infeasible
    unseen_feasible_data = []
    for v in unseen_data_np:
        if tuple(v) in feasible_vector_set:
            unseen_feasible_data.append(v)

    unseen_feasible_data_np = np.array(unseen_feasible_data)

    if len(unseen_feasible_data) <= 0:
        print("No tests required for {} beams".format(total_lines))
        continue

    # Create an image representation
    print("\tCreating an image representation before sorting")
    plt = create_image_representation("Unordered: Beams " + str(total_lines), unseen_feasible_data_np)

    # Compute the reach set details so that we can use that to reconstruct the test
    print("\tCompute the reach set details")
    ego_vehicle, r_set, exterior_r_set, beams, segmented_lines = compute_reach_set_details(total_lines, max_distance, steering_angle, new_accuracy)

    # Save the data to a file
    print("\tSaving to data file")
    tests_generated = save_unseen_data_to_file_single(unseen_data_np, segmented_lines, new_accuracy, args.total_samples, total_lines, args.scenario)
    print("\tTotal tests generated: {}".format(tests_generated))

plt.show()

