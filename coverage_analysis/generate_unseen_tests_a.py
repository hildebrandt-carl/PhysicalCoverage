import math
import random
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_unseen_tests_functions_a import get_trace_files
from generate_unseen_tests_functions_a import compute_coverage
from generate_unseen_tests_functions_a import order_similar_events
from generate_unseen_tests_functions_a import compute_unseen_vectors
from generate_unseen_tests_functions_a import order_similar_events_raw
from generate_unseen_tests_functions_a import save_unseen_data_to_file
from generate_unseen_tests_functions_a import compute_reach_set_details
from generate_unseen_tests_functions_a import create_parallel_plot_reach
from generate_unseen_tests_functions_a import create_image_representation

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',          type=int, default=-1,    help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',               type=str, default="",    help="beamng/highway")
parser.add_argument('--cores',                  type=int, default=4,     help="number of available cores")
parser.add_argument('--maximum_unseen_vectors', type=int, default=1000000, help="The maximum number of unseen vectors")
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
# load_name += "_b" + str(new_total_lines) 
load_name += "_b" + str(new_total_lines) 
load_name += "_d" + str(new_max_distance) 
load_name += "_a" + str(new_accuracy)
load_name += "_t" + str(args.total_samples)
load_name += ".npy"

base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/processed/' + str(args.total_samples) + "a/"
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

unique_observations_per_cell = (new_max_distance / float(new_accuracy)) + 1.0
unique_values_per_beam = [0]
for i in range(math.ceil(unique_observations_per_cell) - 1):
    unique_values_per_beam.append(unique_values_per_beam[-1] + new_accuracy)

print("The unique values each beam can hold are: {}".format(unique_values_per_beam))
print(unique_values_per_beam)

# Compute the unseen vectors
final_data = compute_unseen_vectors(return_dict, args.maximum_unseen_vectors, new_max_distance, unique_values_per_beam, new_accuracy)

# Plot the data
for key in final_data:

    # Print what we are plotting
    print("Plotting {}".format(key))

    # Get the seen and unseen data
    seen_data       = final_data[key]["seen"]
    unseen_data     = final_data[key]["unseen"]
    total_lines     = final_data[key]["total_beams"]
    max_distance    = final_data[key]["max_distance"]
    steering_angle  = final_data[key]["steering_angle"]

    # Create the combined data
    combined_data = pd.concat([unseen_data, seen_data], axis=0)

    # Compute the reachset details
    ego_vehicle, r_set, exterior_r_set, beams, segmented_lines = compute_reach_set_details(total_lines, max_distance, steering_angle, new_accuracy)

    # Plot all as a parallel plot inside the reachset
    plt, unseen_line_strings, exterior_r_set, ego_vehicle, segmented_lines = create_parallel_plot_reach(combined_data, seen_data, unseen_data, ego_vehicle, exterior_r_set, segmented_lines, beams, new_accuracy, key)

    # Order the events so that similar events are closely related
    unseen_line_strings_ordered, distance_tracker = order_similar_events(unseen_line_strings)

    # Order the events so that similar events are closely related
    unseen_data_ordered, distance_tracker = order_similar_events_raw(unseen_data)

    # Create an image representation
    plt = create_image_representation(unseen_data_ordered, distance_tracker, key)

    # Sve the data to a file
    save_unseen_data_to_file(unseen_data_ordered, distance_tracker, segmented_lines, new_accuracy, args.total_samples, total_lines)

plt.show()