import copy
import random 
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from test_selection_config import plot_config, unique_vector_config, compute_crash_hash

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics


def random_selection(number_of_tests):
    global traces

    # Get the hash size
    hash_size = unique_vector_config(args.scenario, number_of_seconds=1)
    unique_crashes_seen = []

    # Generate the test indices
    local_state = np.random.RandomState()
    indices = local_state.choice(traces.shape[0], size=number_of_tests, replace=False)
    # Init variables
    coverage = 0
    number_of_crashes = 0
    unique_vectors_seen = set()
    # For the X test cases
    for random_i in indices:
        # Get the filename
        vectors = traces[random_i]
        # Look for a crash
        if np.isnan(vectors).any():

            # Compute the hash value for this
            hash_value = compute_crash_hash(vectors, hash_size)

            # Check if this is a unique crash (only if its a unique crash count it)
            if len(np.unique(unique_crashes_seen+[hash_value])) > len(np.unique(unique_crashes_seen)):
                unique_crashes_seen.append(hash_value)
                number_of_crashes += 1

        # Check to see if any of the vectors are new
        for vector in vectors:
            # If this vector contains nan it means it crashed (and so we can ignore it, this traces crash was already counted)
            if np.isnan(vector).any():
                continue

            # If this vector contains inf it means that the trace was extended to match sizes with the maximum sized trace
            if np.isinf(vector).any():
                continue
            unique_vectors_seen.add(tuple(vector))
        
        # Compute coverage
        coverage = (len(unique_vectors_seen) / float(total_possible_observations)) * 100
        
    # Return the data
    return ["random", number_of_tests, 0, coverage, number_of_crashes]

def greedy_selection_best(number_of_tests, greedy_sample_size):
    global traces

    # Get the hash size
    hash_size = unique_vector_config(args.scenario, number_of_seconds=1)
    unique_crashes_seen = []

    # Greedy algorithm
    best_case_vectors = set()
    best_case_crashes = 0

    # Hold indices which we can select from
    available_indices = set(np.arange(traces.shape[0]))

    for k in np.arange(number_of_tests):

        # Randomly select greedy_sample_size traces to compare over
        local_state = np.random.RandomState()
        selected_indices = local_state.choice(list(available_indices), size=min(greedy_sample_size, len(available_indices)),replace=False)

        # Holds the min and max coverage
        max_coverage = None
        best_selected_trace_index = -1

        # For each considered trace
        for i in selected_indices:

            # These hold the new vectors
            current_best_case_vectors = copy.deepcopy(best_case_vectors)

            # Get the current test we are considering
            vectors = traces[i]

            # Check to see if any of the vectors are new
            for vector in vectors:
                # If this vector contains nan it means it crashed (and so we can ignore it, this traces crash was already counted)
                if np.isnan(vector).any():
                    continue

                # If this vector contains inf it means that the trace was extended to match sizes with the maximum sized trace
                if np.isinf(vector).any():
                    continue

                current_best_case_vectors.add(tuple(vector))

            # See if this is the best we can do
            if max_coverage is None:
                best_selected_trace_index = i
                max_coverage = copy.deepcopy(current_best_case_vectors)
            elif len(max_coverage) < len(current_best_case_vectors):
                best_selected_trace_index = i
                max_coverage = copy.deepcopy(current_best_case_vectors)

        # Update the best and worst coverage data
        best_case_vectors = copy.deepcopy(max_coverage)

        # Look for a crash
        if np.isnan(traces[best_selected_trace_index]).any():
            # Compute the hash value for this
            hash_value = compute_crash_hash(traces[best_selected_trace_index], hash_size)
            # Check if this is a unique crash (only if its a unique crash count it)
            if len(np.unique(unique_crashes_seen+[hash_value])) > len(np.unique(unique_crashes_seen)):
                unique_crashes_seen.append(hash_value)
                best_case_crashes += 1

        # Remove the selected index from consideration
        available_indices.remove(best_selected_trace_index)

    best_coverage = (len(best_case_vectors) / float(total_possible_observations)) * 100
    
    # Return the data
    return ["best", number_of_tests, greedy_sample_size, best_coverage, best_case_crashes]

def greedy_selection_worst(number_of_tests, greedy_sample_size):
    global traces

    # Get the hash size
    hash_size = unique_vector_config(args.scenario, number_of_seconds=1)
    unique_crashes_seen = []


    # Greedy algorithm
    worst_case_vectors = set()
    worst_case_crashes = 0

    # Hold indices which we can select from
    available_indices = set(np.arange(traces.shape[0]))

    for k in np.arange(number_of_tests):
        
        # Randomly select greedy_sample_size traces to compare over
        local_state = np.random.RandomState()
        
        selected_indices = local_state.choice(list(available_indices), size=min(greedy_sample_size, len(available_indices)),replace=False)

        # Holds the min and max coverage
        min_coverage = None
        max_coverage = []
        worst_selected_trace_index = -1
        best_selected_trace_index = -1

        # For each considered trace
        for i in selected_indices:

            # These hold the new vectors
            current_worst_case_vectors = copy.deepcopy(worst_case_vectors)

            # Get the current test we are considering
            vectors = traces[i]

            # Check to see if any of the vectors are new
            for vector in vectors:
                # If this vector contains nan it means it crashed (and so we can ignore it, this traces crash was already counted)
                if np.isnan(vector).any():
                    continue

                # If this vector contains inf it means that the trace was extended to match sizes with the maximum sized trace
                if np.isinf(vector).any():
                    continue

                current_worst_case_vectors.add(tuple(vector))

            # See if this is the worst we can do
            if min_coverage is None:
                worst_selected_trace_index = i
                min_coverage = copy.deepcopy(current_worst_case_vectors)
            elif len(min_coverage) > len(current_worst_case_vectors):
                worst_selected_trace_index = i
                min_coverage = copy.deepcopy(current_worst_case_vectors)

        # Update the best and worst coverage data
        worst_case_vectors = copy.deepcopy(min_coverage)

        # Look for a crash
        if np.isnan(traces[worst_selected_trace_index]).any():
            # Compute the hash value for this
            hash_value = compute_crash_hash(traces[worst_selected_trace_index], hash_size)
            # Check if this is a unique crash (only if its a unique crash count it)
            if len(np.unique(unique_crashes_seen+[hash_value])) > len(np.unique(unique_crashes_seen)):
                unique_crashes_seen.append(hash_value)
                worst_case_crashes += 1

        # Remove the selected index from consideration
        available_indices.remove(worst_selected_trace_index)

    worst_coverage = (len(worst_case_vectors) / float(total_possible_observations)) * 100
    
    # Return the data
    return ["worst", number_of_tests, greedy_sample_size, worst_coverage, worst_case_crashes]

parser = argparse.ArgumentParser()
parser.add_argument('--beam_count',     type=int, default=5,     help="The number of beams used to vectorized the reachable set")
parser.add_argument('--total_samples',  type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",    help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,     help="The number of CPU cores available")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig(beam_count=args.beam_count)

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy
new_total_lines     = RSR.beam_count

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Total beams:\t\t" + str(new_total_lines))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

# Compute total possible values using the above
unique_observations_per_cell = (new_max_distance / float(new_accuracy)) + 1.0
total_possible_observations = pow(unique_observations_per_cell, new_total_lines)

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

# Get the file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/processed/' + str(args.total_samples) + "/"

print("Loading: " + load_name)
traces = np.load(base_path + "traces_" + args.scenario + load_name)

print("----------------------------------")
print("--------Crashes vs Coverage-------")
print("----------------------------------")

# Get the configuration
total_random_test_suites, test_suite_size_percentage, total_greedy_test_suites, greedy_sample_sizes = plot_config(args.scenario)

# Create a pool with x processes
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

test_suite_size = []
for percentage in test_suite_size_percentage:
    test_size = int(int(args.total_samples) * percentage)
    test_suite_size.append(test_size)

print("Test suite sizes are: {}".format(test_suite_size))

jobs = []
# For all the different test suite sizes
for suite_size in test_suite_size:
    # Create random samples
    for i in range(total_random_test_suites):
        jobs.append(pool.apply_async(random_selection, args=([suite_size])))
    # Create greedy samples
    for _ in range(total_greedy_test_suites):
        for sample_size in greedy_sample_sizes:
            jobs.append(pool.apply_async(greedy_selection_best, args=([suite_size, sample_size])))
            jobs.append(pool.apply_async(greedy_selection_worst, args=([suite_size, sample_size])))

# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

print("Done!")

# Sort the results based on number of test suites
total_tests = total_random_test_suites + int(2 * total_greedy_test_suites * len(greedy_sample_sizes))
final_coverage          = np.zeros((len(test_suite_size), total_tests))
final_number_crashes    = np.zeros((len(test_suite_size), total_tests))
position_counter        = np.zeros(len(test_suite_size), dtype=int)
for r in results:
    # Get the row
    ind = test_suite_size.index(r[1])
    # Save in the correct position
    final_coverage[ind, position_counter[ind]] = r[3]
    final_number_crashes[ind, position_counter[ind]] = r[4]
    position_counter[ind] += 1

# Save the results
save_name = "../results/rq3_"
print("Saving to: " + str(save_name))

np.save(save_name + "coverage_" + str(args.scenario), final_coverage)
np.save(save_name + "crashes_" + str(args.scenario), final_number_crashes)

final_coverage          = np.load(save_name + "coverage_" + str(args.scenario) + ".npy")
final_number_crashes    = np.load(save_name + "crashes_" + str(args.scenario) + ".npy")

# Run the plotting code
exec(compile(open("test_selection_plot.py", "rb").read(), "test_selection_plot.py", 'exec'))