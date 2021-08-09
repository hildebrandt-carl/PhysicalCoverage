import sys
import glob
import argparse
import scipy.stats
import multiprocessing

from time import sleep
from copy import copy

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from general.environment_configurations import RSRConfig
from general.environment_configurations import HighwayKinematics

from scipy.optimize import curve_fit

def line_objective(x, a, b, c):
    return (a * x) + (b * x**2) + c

# Used to generated a random selection of tests
def random_select(number_of_tests):
    global traces
    global crashes
    global feasible_RSR_set
    global unique_crashes_set

    # Generate the indices for the random tests cases
    local_state = np.random.RandomState()
    indices = local_state.choice(traces.shape[0], size=number_of_tests, replace=False)

    # Get the coverage and crash set
    seen_RSR_set = set()
    seen_crash_set = set()

    # Go through each of the different tests
    for i in indices:
        # Get the vectors
        vectors = traces[i]
        crash = crashes[i]

        # Go through the traces and compute coverage
        for v in vectors:

            # Make sure that this is a scene (not a nan or inf or -1)
            if (np.isnan(v).any() == False) and (np.isinf(v).any() == False) and (np.less(v, 0).any() == False):
                seen_RSR_set.add(tuple(v))

        # Check if there was a crash and if there was count it
        if np.isnan(crash) == False:
            seen_crash_set.add(crash)

    # Compute the coverage and the crash percentage
    coverage_percentage = float(len(seen_RSR_set)) / len(feasible_RSR_set)
    crash_percentage =  float(len(seen_crash_set)) / len(unique_crashes_set)

    return [coverage_percentage, crash_percentage]


def coverage_computation(index, seen_RSR_set):
    global traces

    current_RSR = set()
    # Get the coverage for that trace
    for v in traces[index]:

        # Make sure that this is a scene (not a nan or inf or -1)
        if (np.isnan(v).any() == False) and (np.isinf(v).any() == False) and (np.less(v, 0).any() == False):
            current_RSR.add(tuple(v))

    # Get the new coverage
    coverage_set = (seen_RSR_set | current_RSR)
    coverage_percentage = float(len(coverage_set)) / len(feasible_RSR_set)

    return [coverage_set, coverage_percentage, index]

# Used to generated a random selection of tests
def greedy_select(test_suit_size, selection_type, greedy_sample_size):
    global traces
    global crashes
    global feasible_RSR_set
    global unique_crashes_set

    # Get all the available indices
    available_indices = set(np.arange(traces.shape[0]))

    # Get the coverage and crash set
    seen_RSR_set = set()
    seen_crash_set = set()

    # For each of the tests in the test suit
    for _ in range(test_suit_size):

        # Randomly select greedy_sample_size traces
        local_state = np.random.RandomState()
        randomly_selected_indices = local_state.choice(list(available_indices), size=min(greedy_sample_size, len(available_indices)), replace=False)

        results = []
        for index in randomly_selected_indices:
            r = coverage_computation(index, seen_RSR_set)
            results.append(r)

        # Turn the results into coverage and index
        results = np.array(results)
        results = np.transpose(results)
        coverage_sets = results[0, :]
        coverage_percentages = results[1, :]
        indices = results[2, :]

        # Pick the best or the worst coverage
        if selection_type.upper() == "MIN":
            selected_index = indices[np.argmin(coverage_percentages)]
            selected_percentage = coverage_percentages[np.argmin(coverage_percentages)]
            selected_coverage_set = coverage_sets[np.argmin(coverage_percentages)]
        elif selection_type.upper() == "MAX":
            selected_index = indices[np.argmax(coverage_percentages)]
            selected_percentage = coverage_percentages[np.argmax(coverage_percentages)]
            selected_coverage_set = coverage_sets[np.argmax(coverage_percentages)]

        # Remove the index from available indices
        available_indices.remove(selected_index)
        seen_RSR_set = selected_coverage_set

        # Check for crashes in this trace
        crash = crashes[selected_index]
        if np.isnan(crash) == False:
            seen_crash_set.add(crash)


    # Compute the coverage and the crash percentage
    coverage_percentage = float(len(seen_RSR_set)) / len(feasible_RSR_set)
    crash_percentage =  float(len(seen_crash_set)) / len(unique_crashes_set)

    return [coverage_percentage, crash_percentage]


# multiple core
def greedy_selection(cores, test_suite_sizes, selection_type, greedy_sample_size):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for s in test_suite_sizes:
        test_suit_size = int(np.round(s, 0))
        jobs.append(pool.apply_async(greedy_select, args=([test_suit_size, selection_type, greedy_sample_size])))
    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())
    # Its 8pm the pool is closed
    pool.close() 

    # Get the results
    results = np.array(results)
    results = np.transpose(results)
    greedy_coverage_percentages = results[0, :]
    greedy_crash_percentages = results[1, :]

    return greedy_coverage_percentages, greedy_crash_percentages


# multiple core
def random_selection(cores, test_suite_sizes):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for s in test_suite_sizes:
        test_suit_size = int(np.round(s, 0))
        jobs.append(pool.apply_async(random_select, args=([test_suit_size])))
    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())
    # Its 8pm the pool is closed
    pool.close() 

    # Get the results
    results = np.array(results)
    results = np.transpose(results)
    random_coverage_percentages = results[0, :]
    random_crash_percentages = results[1, :]

    return random_coverage_percentages, random_crash_percentages


# Use the tests_per_test_suite
def determine_test_suit_sizes(total_samples):
    increment = 0.01
    test_suit_size_percentage = np.arange(increment, 1.000001, increment)
    test_suit_sizes = test_suit_size_percentage * total_samples
    return test_suit_sizes


parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
parser.add_argument('--beam_count',     type=int, default=3,    help="The number of beams you want to consider")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = ""
load_name += "_s" + str(new_steering_angle) 
load_name += "_b" + str('*') 
load_name += "_d" + str(new_max_distance) 
load_name += "_a" + str(new_accuracy)
load_name += "_t" + str(args.total_samples)
load_name += ".npy"

# Get the file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/randomly_generated/processed/' + str(args.total_samples) + "/"
trace_file = glob.glob(base_path + "traces_*_b{}_*".format(args.beam_count))
crash_file = glob.glob(base_path + "crash_*_b{}_*".format(args.beam_count))

# Get the feasible vectors
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/feasibility/processed/'
feasible_file = glob.glob(base_path + "*_b{}.npy".format(args.beam_count))

# Check we have files
assert(len(trace_file) == 1)
assert(len(crash_file) == 1)
assert(len(feasible_file) == 1)
trace_file = trace_file[0]
crash_file = crash_file[0]
feasible_file = feasible_file[0]

# Get the test suit sizes
test_suit_sizes = determine_test_suit_sizes(args.total_samples)
greedy_sample_size = 5

# Load the traces
global traces
global crashes
traces = np.load(trace_file)
crashes = np.load(crash_file)

# Create the feasible set
feasible_traces = np.load(feasible_file)
global feasible_RSR_set
feasible_RSR_set = set()
for scene in feasible_traces:
    feasible_RSR_set.add(tuple(scene))

# Create the crash unique set
global unique_crashes_set
unique_crashes_set = set()
for crash in crashes:
    if np.isnan(crash) == False:
        unique_crashes_set.add(crash)

# Create the figure
plt.figure(1)

# For each of the different beams
print("Generating random tests")
random_coverage_percentages, random_crash_percentages = random_selection(args.cores, test_suit_sizes)
plt.scatter(random_coverage_percentages, random_crash_percentages, c="C0", label="random", s=2)
popt, _ = curve_fit(line_objective, random_coverage_percentages, random_crash_percentages)
a, b, c = popt
x_line = np.arange(min(random_coverage_percentages), max(random_coverage_percentages), 0.01)
y_line = line_objective(x_line, a, b, c)
plt.plot(x_line, y_line, '--', color='C0')

print("Generating best case greedy tests")
best_coverage_percentages, best_crash_percentages = greedy_selection(args.cores, test_suit_sizes, "max", greedy_sample_size)
plt.scatter(best_coverage_percentages, best_crash_percentages, c="C1", s=2, label="Greedy Best")
popt, _ = curve_fit(line_objective, best_coverage_percentages, best_crash_percentages)
a, b, c = popt
x_line = np.arange(min(best_coverage_percentages), max(best_coverage_percentages), 0.01)
y_line = line_objective(x_line, a, b, c)
plt.plot(x_line, y_line, '--', color='C1')

print("Generating worst case greedy tests")
worst_coverage_percentages, worst_crash_percentages = greedy_selection(args.cores, test_suit_sizes, "min", greedy_sample_size)
plt.scatter(worst_coverage_percentages, worst_crash_percentages, c="C2", s=2, label="Greedy Worst")
popt, _ = curve_fit(line_objective, worst_coverage_percentages, worst_crash_percentages)
a, b, c = popt
x_line = np.arange(min(worst_coverage_percentages), max(worst_coverage_percentages), 0.01)
y_line = line_objective(x_line, a, b, c)
plt.plot(x_line, y_line, '--', color='C2')

plt.legend()
plt.ylabel("Crashes (%)")
plt.xlabel("Coverage (%)")
plt.show()