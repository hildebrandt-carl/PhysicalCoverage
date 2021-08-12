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
base_directory = str(path[:path.rfind("/test_selection")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from general.environment_configurations import RSRConfig
from general.environment_configurations import HighwayKinematics


# Used to generated a random selection of tests
def random_selection(number_of_tests):
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


# Used to generated a random selection of tests
def greedy_selection(number_of_tests, greedy_sample_size, selection_type="Min"):
    global traces
    global crashes
    global feasible_RSR_set
    global unique_crashes_set

    # Get all the available indices
    available_indices = set(np.arange(traces.shape[0]))

    # Get the coverage and crash set
    seen_RSR_set = set()
    seen_crash_set = set()

    # For each test
    for _ in range(number_of_tests):

        # Randomly select greedy_sample_size traces
        local_state = np.random.RandomState()
        randomly_selected_indices = local_state.choice(list(available_indices), size=greedy_sample_size, replace=False)

        # Holds the final coverage set for each selected index
        coverage_array = []
        coverage_RSR_array = []

        # Greedily select the best or worst one
        for index in randomly_selected_indices:

            # Holds the current RSR
            current_RSR = set()

            # Get the coverage for that trace
            for v in traces[index]:

                # Make sure that this is a scene (not a nan or inf or -1)
                if (np.isnan(v).any() == False) and (np.isinf(v).any() == False) and (np.less(v, 0).any() == False):
                    current_RSR.add(tuple(v))

            # Save the coverage
            coverage_array.append(len(current_RSR | seen_RSR_set))
            coverage_RSR_array.append(copy(current_RSR))

        # Find the best or worst coverage
        if selection_type.upper() == "MIN":
            selected = np.argmin(coverage_array)
        elif selection_type.upper() == "MAX":
            selected = np.argmax(coverage_array)

        # Remove the selected index from available processing
        available_indices.remove(randomly_selected_indices[selected])


        # Update the coverage
        seen_RSR_set = seen_RSR_set | coverage_RSR_array[selected]

        # Updated the crashes
        crash = crashes[randomly_selected_indices[selected]]
        if np.isnan(crash) == False:
            seen_crash_set.add(crash)

    # Compute the coverage and the crash percentage
    coverage_percentage = float(len(seen_RSR_set)) / len(feasible_RSR_set)
    crash_percentage =  float(len(seen_crash_set)) / len(unique_crashes_set)
    # print("")

    return [coverage_percentage, crash_percentage]


# multiple core
def random_selection_multiple_core(cores, total_test_suits, test_suit_size):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for _ in range(total_test_suits):
        jobs.append(pool.apply_async(random_selection, args=([test_suit_size])))
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


def greedy_selection_multiple_core(cores, total_test_suits, test_suit_size, greedy_sample_sizes, greedy_type):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for _ in range(int(np.round(total_test_suits / len(greedy_sample_sizes), 0))):
        for greedy_size in greedy_sample_sizes:
            jobs.append(pool.apply_async(greedy_selection, args=([test_suit_size, greedy_size, greedy_type])))
    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())
    # # Its 8pm the pool is closed
    pool.close() 

    # Get the results
    results = np.array(results)
    results = np.transpose(results)
    best_coverage_percentages = results[0, :]
    best_crash_percentages = results[1, :]

    return best_coverage_percentages, best_crash_percentages


# Use the tests_per_test_suite
def determine_test_suit_sizes(total_samples):
    test_suit_size_percentage = np.array([0.01, 0.025, 0.05]) 
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
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/processed/' + str(args.total_samples) + "/"
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
total_test_suits = 100
greedy_sample_sizes = [2, 3, 4, 5, 10]

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
for s in range(len(test_suit_sizes)):
    # Get the test suit size
    test_suit_size = int(np.round(test_suit_sizes[s], 0))

    # Computing the test suit size
    print("Computing test suits of size: {}".format(test_suit_size))

    print("Generating random tests")
    random_coverage_percentages, random_crash_percentages = random_selection_multiple_core(args.cores, total_test_suits, test_suit_size)
    plt.scatter(random_coverage_percentages, random_crash_percentages, c="C" + str(s), label=str(test_suit_size), s=2)

    print("Generating best case greedy tests")
    best_coverage_percentages, best_crash_percentages = greedy_selection_multiple_core(args.cores, total_test_suits, test_suit_size, greedy_sample_sizes, "max")
    plt.scatter(best_coverage_percentages, best_crash_percentages, c="C" + str(s), s=2)

    print("Generating worst case greedy tests")
    worst_coverage_percentages, worst_crash_percentages = greedy_selection_multiple_core(args.cores, total_test_suits, test_suit_size, greedy_sample_sizes, "max")
    plt.scatter(worst_coverage_percentages, worst_crash_percentages, c="C" + str(s), s=2)

    print("Computing the line of best fit")
    # Get all the data
    x = np.concatenate([random_coverage_percentages, best_coverage_percentages, worst_coverage_percentages])
    y = np.concatenate([random_crash_percentages, best_crash_percentages, worst_crash_percentages])

    # Compute the line of best fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    x_range = np.arange(np.min(x), np.max(x), 0.01)

    # Generate the label for the regression line
    lb = str(np.round(slope,2)) +"x+" + str(int(np.round(intercept,0)))
    if intercept < 0:
        lb = str(np.round(slope,2)) +"x" + str(int(np.round(intercept,0)))

    # Plot the line of best fit
    plt.plot(x_range, slope*x_range + intercept, c='C' + str(s), label=lb)



# reorder the legend
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
indices = [3,0,4,1,5,2]
new_handles = list(np.array(handles)[indices])
new_labels = list(np.array(labels)[indices])

plt.legend(new_handles, new_labels, markerscale=2, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=len(test_suit_sizes), handletextpad=0.1)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("Crashes (%)")
plt.xlabel("Coverage (%)")
plt.show()