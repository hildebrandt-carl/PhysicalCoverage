import sys
import glob
import argparse
import scipy.stats
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from general.environment_configurations import RSRConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics

from scipy.optimize import curve_fit

def line_objective(x, a, b, c):
    return (a * x) + (b * x**2) + c

def line_of_best_fit(x, y):
    popt, _ = curve_fit(line_objective, x, y)
    a, b, c = popt
    x_line = np.arange(min(x), max(x), 0.01)
    y_line = line_objective(x_line, a, b, c)
    return x_line, y_line

# Used to generated a random selection of tests
def random_select(number_of_tests):
    global traces
    global crashes
    global stalls
    global feasible_RSR_set
    global unique_failure_set

    # Generate the indices for the random tests cases
    local_state = np.random.RandomState()
    indices = local_state.choice(traces.shape[0], size=number_of_tests, replace=False)

    # Get the coverage and failure set
    seen_RSR_set = set()
    seen_failure_set = set()

    # Go through each of the different tests
    for i in indices:
        # Get the vectors
        vectors = traces[i]
        crash = crashes[i]
        stall = stalls[i]

        # Go through the traces and compute coverage
        for v in vectors:

            # Make sure that this is a scene (not a nan or inf or -1)
            if (np.isnan(v).any() == False) and (np.isinf(v).any() == False) and (np.less(v, 0).any() == False):
                seen_RSR_set.add(tuple(v))

        # Check if there was a crash and if there was count it
        for c in crash:
            if ~np.isinf(c):
                seen_failure_set.add(c)

        # Check if there was a stall and if there was count it
        for s in stall:
            if ~np.isinf(s):
                seen_failure_set.add(s)

    # Compute the coverage and the crash percentage
    coverage_percentage = float(len(seen_RSR_set)) / len(feasible_RSR_set)
    failure_percentage =  float(len(seen_failure_set)) / len(unique_failure_set)
    failures_found = len(seen_failure_set)

    return [coverage_percentage, failures_found, number_of_tests]

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
    # coverage_percentage = float(len(coverage_set)) / len(feasible_RSR_set)
    crash_count = len(coverage_set)

    return [coverage_set, crash_count, index]

# Used to generated a random selection of tests
def greedy_select(test_suite_size, selection_type, greedy_sample_size):
    global traces
    global crashes
    global stalls
    global feasible_RSR_set
    global unique_failure_set

    # Get all the available indices
    available_indices = set(np.arange(traces.shape[0]))

    # Get the coverage and crash set
    seen_RSR_set = set()
    seen_failure_set = set()

    # For each of the tests in the test suite
    for _ in range(test_suite_size):

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
        for c in crash:
            if ~np.isinf(c):
                seen_failure_set.add(c)

        # Check for stalls in this trace
        stall = stalls[selected_index]
        for s in stall:
            if ~np.isinf(s):
                seen_failure_set.add(s)

    # Compute the coverage and the crash percentage
    coverage_percentage = float(len(seen_RSR_set)) / len(feasible_RSR_set)
    # failure_percentage =  float(len(seenseen_failure_set_crash_set)) / len(unique_crashes_set)
    failure_count = len(seen_failure_set)

    return [coverage_percentage, failure_count, test_suite_size]

# multiple core
def greedy_selection(cores, test_suite_sizes, selection_type, greedy_sample_size):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for s in test_suite_sizes:
        test_suite_size = int(np.round(s, 0))
        jobs.append(pool.apply_async(greedy_select, args=([test_suite_size, selection_type, greedy_sample_size])))
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
    greedy_crash_count          = results[1, :]
    result_test_suite_size       = results[2, :]

    return greedy_coverage_percentages, greedy_crash_count, result_test_suite_size

# multiple core
def random_selection(cores, test_suite_sizes):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for s in test_suite_sizes:
        test_suite_size = int(np.round(s, 0))
        jobs.append(pool.apply_async(random_select, args=([test_suite_size])))
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
    random_crash_count          = results[1, :]
    result_test_suite_size       = results[2, :]

    return random_coverage_percentages, random_crash_count, result_test_suite_size

# Use the tests_per_test_suite
def determine_test_suite_sizes(number_of_tests):
    increment = 0.0001
    test_suite_size_percentage = np.arange(increment, 0.0100001, increment)
    test_suite_sizes = test_suite_size_percentage * number_of_tests
    return test_suite_sizes

# Declare the greedy sample size
greedy_sample_size = 100

# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--number_of_tests',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',     type=str, default="",   help="linear/center_close/center_mid")
parser.add_argument('--RRS_number',       type=int, default=3,    help="The number of beams you want to consider")
parser.add_argument('--scenario',         type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',            type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
NG = BeamNGKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
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
base_path = '/media/carl/DataDrive/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
print(base_path + "traces_*_b{}_*".format(args.RRS_number))
trace_file = glob.glob(base_path + "traces_*_b{}_*".format(args.RRS_number))
crash_file = glob.glob(base_path + "crash_*_b{}_*".format(args.RRS_number))
stall_file = glob.glob(base_path + "stall_*_b{}_*".format(args.RRS_number))

# Get the feasible vectors
base_path = '/media/carl/DataDrive/PhysicalCoverageData/{}/feasibility/processed/{}/'.format(args.scenario, args.distribution)
feasible_file = glob.glob(base_path + "*_b{}.npy".format(args.RRS_number))

# Check we have files
assert(len(trace_file) == 1)
assert(len(crash_file) == 1)
assert(len(stall_file) == 1)
assert(len(feasible_file) == 1)

# Get the file name out of the array
trace_file = trace_file[0]
crash_file = crash_file[0]
stall_file = stall_file[0]
feasible_file = feasible_file[0]

# Get the test suite sizes
test_suite_sizes = determine_test_suite_sizes(args.number_of_tests)
print("Considered test suite sizes: {}".format(test_suite_sizes))

# Load the traces
global traces
global crashes
global stalls
traces  = np.load(trace_file)
crashes = np.load(crash_file)
stalls  = np.load(stall_file)

# Create the feasible set
feasible_traces = np.load(feasible_file)
global feasible_RSR_set
feasible_RSR_set = set()
for scene in feasible_traces:
    feasible_RSR_set.add(tuple(scene))

# Create the failure unique set
global unique_failure_set
unique_failure_set = set()
for crash in crashes:
    for c in crash:
        if ~np.isinf(c):
            unique_failure_set.add(c)
for stall in stalls:
    for s in stall:
        if ~np.isinf(s):
            unique_failure_set.add(s)

# Create the figure
plt.figure(1)

# For each of the different beams
print("Generating random tests")
random_coverage_percentages, random_crash_count, random_test_suite_size = random_selection(args.cores, test_suite_sizes)
plt.scatter(random_test_suite_size, random_crash_count, c="C0", marker="s", label="Random", s=5)
x_line, y_line = line_of_best_fit(random_test_suite_size, random_crash_count)
plt.plot(x_line, y_line, '--', color="C0")

print("Generating best case greedy tests")
best_coverage_percentages, best_crash_count, best_test_suite_size = greedy_selection(args.cores, test_suite_sizes, "max", greedy_sample_size)
plt.scatter(best_test_suite_size, best_crash_count, c="C1", marker="o", s=6, label="Greedy Best")
x_line, y_line = line_of_best_fit(best_test_suite_size, best_crash_count)
plt.plot(x_line, y_line, '--', color="C1")

print("Generating worst case greedy tests")
worst_coverage_percentages, worst_crash_count, worst_test_suite_size = greedy_selection(args.cores, test_suite_sizes, "min", greedy_sample_size)
plt.scatter(worst_test_suite_size, worst_crash_count, c="C2", marker="^", s=7, label="Greedy Worst")
x_line, y_line = line_of_best_fit(worst_test_suite_size, worst_crash_count)
plt.plot(x_line, y_line, '--', color="C2")

if args.scenario == "highway":
    increment = 50
if args.scenario == "beamng":
    increment = 10

plt.plot([], [] ,'--', color="black",label="Line of Best Fit")

plt.legend(markerscale=5)
plt.title(args.distribution)
# plt.xticks(np.arange(0, np.max(random_test_suite_size) + 0.01,  args.number_of_tests/100))
plt.yticks(np.arange(0, np.max(best_crash_count) + 0.01, increment))
plt.ylabel("Unique Failures")
plt.xlabel("Test Suite Size")
# Round the x ticks to 3 places
# plt.ticklabel_format(style='sci', axis='x', scilimits=(2,3))
plt.grid(alpha=0.5)
plt.show()