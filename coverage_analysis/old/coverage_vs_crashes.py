import glob
import random 
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import argparse
import copy

def isUnique(vector, unique_vectors_seen):
    unique = True
    for v2 in unique_vectors_seen:
        # If we have seen this vector break out of this loop
        if np.array_equal(vector, v2):
            unique = False
            break
    return unique

parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,    help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=5,     help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',   type=int, default=30,    help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,     help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=10000, help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--greedy_sample',  type=int, default=100,   help="The unumber of samples considered by the greedy search")
args = parser.parse_args()

new_steering_angle  = args.steering_angle
new_total_lines     = args.beam_count
new_max_distance    = args.max_distance
new_accuracy        = args.accuracy
greedy_sample_size  = args.greedy_sample

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Total beams:\t\t" + str(new_total_lines))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))
print("Greedy Sample Size:\t" + str(greedy_sample_size))

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

print("Loading: " + load_name)
traces = np.load("traces" + load_name)

print("----------------------------------")
print("--------Crashes vs Coverage-------")
print("----------------------------------")

# Select 1000 test suites each with 100 test cases
# Plot the crashes vs coverage data

total_test_suites = 25
tests_per_test_suite = [10, 25, 50, 100, 250, 500]

plt.figure(1)

for j in range(len(tests_per_test_suite)):
    # Get the number of tests
    test_number = tests_per_test_suite[j]

    random_selection_coverage_data = []
    random_selection_crash_data = []

    # Randomly picked tests
    print("Randomly selecting " + str(total_test_suites) + " test suites with " + str(test_number) + " tests")
    for i in tqdm(np.arange(total_test_suites)):
        # Generate the test indices
        indices = np.random.choice(traces.shape[0], test_number, replace=False)
        # Init variables
        coverage = 0
        crashes_found = 0
        unique_vectors_seen = []
        # For the X test cases
        for ind in tqdm(range(len(indices)), leave=False):
            random_i = indices[ind]
            # Get the filename
            vectors = traces[random_i]
            # Look for a crash
            if np.isnan(vectors).any():
                crashes_found += 1
            # Check to see if any of the vectors are new
            for v in vectors:
                if not np.isnan(v).any():
                    unique = isUnique(v, unique_vectors_seen)
                    if unique:
                        unique_vectors_seen.append(v)
        
        # Compute coverage
        cov = (len(unique_vectors_seen) / float(total_possible_observations)) * 100

        # Save the data for that test suite
        random_selection_coverage_data.append(cov)
        random_selection_crash_data.append(crashes_found)

    # Greedy algorithm
    worst_case_vectors = []
    worst_case_crashes = 0
    best_case_vectors = []
    best_case_crashes = 0

    print("Greedy case test suite with " + str(test_number) + " tests")
    for k in tqdm(np.arange(test_number)):
       
        # Randomly select 10 traces to subsample from
        indices = np.random.choice(traces.shape[0], greedy_sample_size, replace=False)

        # Holds the min and max coverage
        min_coverage = np.zeros(1000)
        max_coverage = []
        worst_selected_trace_index = -1
        best_selected_trace_index = -1

        # For each considered trace
        for ind in tqdm(range(len(indices)), leave=False):
            # Select the correct index
            i = indices[ind]

            # These hold the new vectors
            current_worst_case_vectors = copy.deepcopy(worst_case_vectors)
            current_best_case_vectors = copy.deepcopy(best_case_vectors)

            # Get the current test we are considering
            vectors = traces[i]

            # Check to see if any of the vectors are new in this trace
            for v in vectors:
                if not np.isnan(v).any():
                    unique = isUnique(v, current_worst_case_vectors)
                    if unique:
                        current_worst_case_vectors.append(v)

                    unique = isUnique(v, current_best_case_vectors)
                    if unique:
                        current_best_case_vectors.append(v)

            # See if this is the worst we can do
            if len(min_coverage) > len(current_worst_case_vectors):
                worst_selected_trace_index = i
                min_coverage = copy.deepcopy(current_worst_case_vectors)

            # See if this is the best we can do
            if len(max_coverage) < len(current_best_case_vectors):
                best_selected_trace_index = i
                max_coverage = copy.deepcopy(current_best_case_vectors)

        # Update the best and worst coverage data
        worst_case_vectors = copy.deepcopy(min_coverage)
        best_case_vectors = copy.deepcopy(max_coverage)
        # Update the best and worst crash data
        if np.isnan(traces[worst_selected_trace_index]).any():
            worst_case_crashes += 1
        if np.isnan(traces[best_selected_trace_index]).any():
            best_case_crashes += 1

        # Remove the selected index from consideration
        traces = np.delete(traces, [worst_selected_trace_index, best_selected_trace_index], axis=0)

    worst_coverage = (len(worst_case_vectors) / float(total_possible_observations)) * 100
    best_coverage = (len(best_case_vectors) / float(total_possible_observations)) * 100
    print("Worst Coverage: " + str(worst_coverage) + " - Crashes: " + str(worst_case_crashes))
    print("Best Coverage: " + str(best_coverage) + " - Crashes: " + str(best_case_crashes))
    print("----------")

    # Plot the data
    plt.scatter(worst_coverage, worst_case_crashes, color='C' + str(j), marker='*', s=40)
    plt.scatter(best_coverage, best_case_crashes, color='C' + str(j), marker='*', s=40)
    plt.scatter(random_selection_coverage_data, random_selection_crash_data, color='C' + str(j), marker='o', label="#Tests: " + str(test_number), s=5)

    # Compute the line of best fit
    random_selection_coverage_data.append(worst_coverage)
    random_selection_coverage_data.append(best_coverage)
    random_selection_crash_data.append(worst_case_crashes)
    random_selection_crash_data.append(best_case_crashes)
    m, b = np.polyfit(random_selection_coverage_data, random_selection_crash_data, 1)
    x_range = np.arange(0, best_coverage)
    plt.plot(x_range, m*x_range + b, c='C' + str(j))


first_legend = plt.legend(loc='upper left')
ax = plt.gca().add_artist(first_legend)
greed = plt.scatter([], [], color='black', marker='*', s=40, label='Greedy Selection')
rand = plt.scatter([], [], color='black', marker='o', s=20, label='Random Selection')
second_legend = plt.legend(handles=[greed, rand], loc='lower right')
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of Crashes")
plt.title("Greedy Sample Size: " + str(greedy_sample_size))

plt.show()