import math
import random
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_unseen_tests_functions import get_trace_files
from generate_unseen_tests_functions import compute_coverage
from generate_unseen_tests_functions import compute_unseen_vectors
from generate_unseen_tests_functions import save_unseen_data_to_file
from generate_unseen_tests_functions import compute_reach_set_details
from generate_unseen_tests_functions import create_image_representation

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',          type=int, default=-1,       help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',               type=str, default="",       help="beamng/highway")
parser.add_argument('--cores',                  type=int, default=4,        help="number of available cores")
parser.add_argument('--maximum_unseen_vectors', type=int, default=1000000,  help="The maximum number of unseen vectors")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy
new_total_lines     = "*"

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

load_name = ""
load_name += "_s" + str(new_steering_angle) 
load_name += "_b" + str(new_total_lines) 
load_name += "_d" + str(new_max_distance) 
load_name += "_a" + str(new_accuracy)
load_name += "_t" + str(args.total_samples)
load_name += ".npy"

base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/processed/' + str(args.total_samples) + "/"
file_names = get_trace_files(base_path, args.scenario, load_name)

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
unique_observations_per_cell = (new_max_distance / float(new_accuracy)) + 1.0
unique_values_per_beam = [0]
for i in range(math.ceil(unique_observations_per_cell) - 1):
    unique_values_per_beam.append(unique_values_per_beam[-1] + new_accuracy)

print("\n\n\n--------------------------------------------------------")
print("The unique values each beam can hold are: {}".format(unique_values_per_beam))

# Compute the unseen vectors
final_data = compute_unseen_vectors(return_dict, args.maximum_unseen_vectors, new_max_distance, unique_values_per_beam, new_accuracy)

# Plot the data
for key in final_data:

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
    unseen_data_np = unseen_data.to_numpy()

    # Create an image representation
    print("\tCreating an image representation before sorting")
    plt = create_image_representation("Unordered: Beams " + str(total_lines), unseen_data_np)

    # Used to keep track of the distance between points
    distance_metric = np.full(unseen_data_np.shape[0], np.inf)
    distance_metric[0] = 0

    # Order the data based on manhattan distance apart
    for i in range(unseen_data_np.shape[0] - 1):
        # Get the current index point
        reference_point = unseen_data_np[i]
        # Compute the manhattan distance
        dist = np.sum(np.abs(unseen_data_np[i:] - reference_point), axis=1)
        # Sort all values based on the index point
        new_indices = np.argsort(dist)
        unseen_data_np[i:] = unseen_data_np[i:][new_indices]
        # Save the selected distance
        distance_metric[i + 1] = dist[new_indices[1]]

    # Create an image representation
    print("\tCreating an image representation after sorting")
    plt = create_image_representation("Sorted: Beams " + str(total_lines), unseen_data_np, distance_metric)

    # Compute the reach set details so that we can use that to reconstruct the test
    print("\tCompute the reach set details")
    ego_vehicle, r_set, exterior_r_set, beams, segmented_lines = compute_reach_set_details(total_lines, max_distance, steering_angle, new_accuracy)

    # Save the data to a file
    print("\tSaving to data file")
    tests_generated = save_unseen_data_to_file(unseen_data_np, distance_metric, segmented_lines, new_accuracy, args.total_samples, total_lines)
    print("\tTotal tests generated: {}".format(tests_generated))

plt.show()