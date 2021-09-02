import os
import ast
import sys
import glob
import json
import shutil
import argparse
import multiprocessing

from time import sleep

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr
from general_functions import order_by_beam
from general_functions import get_beam_numbers
from general_functions import get_ignored_code_coverage_lines

from general.environment_configurations import RSRConfig
from general.environment_configurations import HighwayKinematics

# Get the coverage on a random test suit 
def coverage_on_random_test_suit(suit_size):
    global traces
    global feasible_RSR_set

    # Randomly generate the indices for this test suit
    local_state = np.random.RandomState()
    indices = local_state.choice(len(traces), suit_size, replace=False) 

    # Used to compute the coverage for this trace
    Unique_RSR = set()

    # Go through each of the indices
    for index in indices:
        # Get the trace
        trace = traces[index]

        # Add it to the RSR set
        for scene in trace:
            # Get the current scene
            s = tuple(scene)

            # Make sure that this is a scene (not a nan or inf or -1)
            if (np.isnan(scene).any() == False) and (np.isinf(scene).any() == False) and (np.less(scene, 0).any() == False):
                Unique_RSR.add(tuple(s))

                # Give a warning if a vector is found that is not feasible
                if s not in feasible_RSR_set:
                    print("Infeasible vector found: {}".format(scene))

    # Compute coverage
    coverage = float(len(Unique_RSR)) / len(feasible_RSR_set)

    return coverage

# Get the coverage on a random test suit 
def coverage_on_random_test_suit_no_crashes(suit_size):
    global traces
    global crashes
    global feasible_RSR_set

    # Randomly generate the indices for this test suit
    local_state = np.random.RandomState()
    indices = local_state.choice(len(traces), (suit_size*2)-1, replace=False) 

    # Used to compute the coverage for this trace
    Unique_RSR = set()

    # Go through each of the indices
    processed = 0
    counter = 0
    while processed < suit_size:
        index = indices[counter]
        counter += 1
        # Get the trace
        trace = traces[index]
        crash = crashes[index]

        # If the trace is crashing ignore it
        if ~np.isinf(crash[0]):
            continue
        else:
            processed += 1

        # Add it to the RSR set
        for scene in trace:
            # Get the current scene
            s = tuple(scene)


            # Make sure that this is a scene (not a nan or inf or -1)
            if (np.isnan(scene).any() == False) and (np.isinf(scene).any() == False) and (np.less(scene, 0).any() == False):
                Unique_RSR.add(tuple(s))

                # Give a warning if a vector is found that is not feasible
                if s not in feasible_RSR_set:
                    print("Infeasible vector found: {}".format(scene))

    # Compute coverage
    coverage = float(len(Unique_RSR)) / len(feasible_RSR_set)

    return coverage

# Determine the test sizes for this plot
def determine_test_suit_sizes(total_tests):
    # We now need to sample tests of different sizes to create the plot
    percentage_of_all_tests = np.arange(0,50.0001, 5)
    percentage_of_all_tests[0] += 1
    test_sizes = []

    for p in percentage_of_all_tests:

        # Compute what the size is
        t = (p/100.0) * total_tests
        test_sizes.append(int(np.round(t,0)))

    return test_sizes

# Get the coverage on a random test suit 
def code_coverage_on_random_test_suit(suit_size, job_number):
    global code_coverage_file_names
    global ignored_lines

    # Randomly select suit_size files:
    local_state = np.random.RandomState()
    indices = local_state.choice(len(code_coverage_file_names), suit_size, replace=False) 

    # Create the lines and random lines
    lines_coverage = set()
    all_possible_lines = set()

    # For each file
    for i in indices:

        # Get the file
        file_name = code_coverage_file_names[i]
        f = open(file_name, "r")

        # Read the file
        for line in f: 
            if "Lines covered:" in line:
                covered_l = ast.literal_eval(line[15:])

            if "Total lines covered:" in line:
                total_covered_l = int(line[21:])
                assert(len(covered_l) == total_covered_l)

            if "All Lines:" in line:
                all_l = ast.literal_eval(line[11:])

            if "Total Lines:" in line:
                total_all_l = int(line[13:])
                assert(len(all_l) == total_all_l)

        # Close the file
        f.close()

        # Keep track of the lines covered
        lines_coverage = lines_coverage | set(covered_l)

        # Get all the possible lines
        if len(all_possible_lines) <= 0:
            all_possible_lines = set(all_l)
        else:
            assert(all_possible_lines == set(all_l))

    # Remove the ignored lines from all possible lines and lines covered
    lines_coverage = lines_coverage - ignored_lines
    all_possible_lines = all_possible_lines - ignored_lines

    # Compute the coverage
    coverage = float(len(lines_coverage)) / len(all_possible_lines)

    return coverage

# Get the coverage on a random test suit 
def code_coverage_on_random_test_suit_no_crashes(suit_size, job_number):
    global code_coverage_file_names
    global ignored_lines

    # Randomly generate the indices for this test suit
    local_state = np.random.RandomState()
    indices = local_state.choice(len(code_coverage_file_names), (suit_size*2)-1, replace=False) 

    # Create the lines and random lines
    lines_coverage = set()
    all_possible_lines = set()

    processed = 0
    counter = 0
    # For each file
    while processed < suit_size:

        # Get the file
        index = indices[counter]
        file_name = code_coverage_file_names[index]
        counter += 1
        f = open(file_name, "r")

        # Init the variable we care about
        number_of_crashes = -1
        covered_l = []
        all_l = []
        total_covered_l = 0
        total_all_l=  0

        # Read the file
        for line in f: 
            if "Lines covered:" in line:
                covered_l = ast.literal_eval(line[15:])

            if "Total lines covered:" in line:
                total_covered_l = int(line[21:])
                assert(len(covered_l) == total_covered_l)

            if "All Lines:" in line:
                all_l = ast.literal_eval(line[11:])

            if "Total Lines:" in line:
                total_all_l = int(line[13:])
                assert(len(all_l) == total_all_l)

            if "Total physical crashes: " in line:
                number_of_crashes = int(line[24:])

        # Close the file
        f.close()

        # Check if there is a crash
        if number_of_crashes != 0:
            continue
        else:
            processed += 1

        # Keep track of the lines covered
        lines_coverage = lines_coverage | set(covered_l)

        # Get all the possible lines
        if len(all_possible_lines) <= 0:
            all_possible_lines = set(all_l)
        else:
            assert(all_possible_lines == set(all_l))

    # Remove the ignored lines from all possible lines and lines covered
    lines_coverage = lines_coverage - ignored_lines
    all_possible_lines = all_possible_lines - ignored_lines

    # Compute the coverage
    coverage = float(len(lines_coverage)) / len(all_possible_lines)

    return coverage


parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
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
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/physical_coverage/processed/' + str(args.total_samples) + "/"
trace_file_names = glob.glob(base_path + "traces_*")
crash_file_names = glob.glob(base_path + "crash_*")

# Get the feasible vectors
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/feasibility/processed/'
feasible_file_names = glob.glob(base_path + "*.npy")

# Get the code coverage
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/code_coverage/raw/'
global code_coverage_file_names
code_coverage_file_names = glob.glob(base_path + "*/*.txt")\

global ignored_lines
ignored_lines = get_ignored_code_coverage_lines(args.scenario)

# Make sure you have all the files you need
assert(len(trace_file_names) > 1)
assert(len(crash_file_names) > 1)
assert(len(feasible_file_names) > 1)
assert(len(code_coverage_file_names) > 1)

# Get the beam numbers
trace_beam_numbers = get_beam_numbers(trace_file_names)
crash_beam_numbers = get_beam_numbers(crash_file_names)
feasibility_beam_numbers = get_beam_numbers(feasible_file_names)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(trace_beam_numbers) | set(crash_beam_numbers) | set(feasibility_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
trace_file_names = order_by_beam(trace_file_names, beam_numbers)
crash_file_names = order_by_beam(crash_file_names, beam_numbers)
feasible_file_names = order_by_beam(feasible_file_names, beam_numbers)

# Get the test suit sizes
test_suit_sizes = determine_test_suit_sizes(args.total_samples)

# Create the output figure
plt.figure(1)

# Compute the total coverage for tests of different sizes
total_test_suits = 10

# Used to save each of the different coverage metrics so that we can compute the correlation between that and the crash data
all_coverage_data = []

# For each of the different beams
for i in range(len(beam_numbers)):

    # Get the beam number and files we are currently considering
    beam_number = beam_numbers[i]
    trace_file = trace_file_names[i]
    crash_file = crash_file_names[i]
    feasibility_file = feasible_file_names[i]

    if beam_number > 6:
        continue

    print("Processing beams: {}".format(beam_numbers[i]))

    # Skip if any of the files are blank
    if trace_file == "" or crash_file == "" or feasibility_file == "":
        print(feasibility_file)
        print(crash_file)
        print(trace_file)
        print("\nWarning: Could not find one of the files for beam number: {}".format(beam_number))
        continue

    # Load the feasibility file
    feasible_traces = np.load(feasibility_file)
    
    # Create the feasible set
    global feasible_RSR_set
    feasible_RSR_set = set()
    for scene in feasible_traces:
        feasible_RSR_set.add(tuple(scene))

    # Load the traces
    global traces
    global crashes
    traces = np.load(trace_file)
    crashes = np.load(crash_file)

    # Create the crash unique set
    global unique_crashes_set
    unique_crashes_set = set()
    for k in range(crashes.shape[0]):
        for l in range(crashes.shape[1]):
            crash = crashes[k][l]
            if ~np.isinf(crash):
                unique_crashes_set.add(crash)

    # Create the average line
    average_coverage = []
    average_coverage_no_crashes = []

    # Keep a list of all results
    all_results = []
    
    # Go through each of the different test suit sizes
    for suit_size in test_suit_sizes:
        print("Processing test suit size: {}".format(suit_size))

        # Create the pool for parallel processing
        pool =  multiprocessing.Pool(processes=args.cores)

        # Call our function total_test_suites times
        jobs = []
        for _ in range(total_test_suits):
            jobs.append(pool.apply_async(coverage_on_random_test_suit, args=([suit_size])))

        # Get the results
        results = []
        for job in tqdm(jobs):
            results.append(job.get())

        # Its 8pm the pool is closed
        pool.close() 

        # Get the average coverage for this test suit size
        average_coverage.append(np.average(results))

        # Keep track of all the results
        all_results.append(np.average(results))

        # Plot the data
        plt.scatter(np.full(len(results), suit_size), results, marker='o', c="C{}".format(i), s=0.5)

        # Create the pool for parallel processing
        pool =  multiprocessing.Pool(processes=args.cores)

        # Call our function total_test_suites times
        jobs = []
        for _ in range(total_test_suits):
            jobs.append(pool.apply_async(coverage_on_random_test_suit_no_crashes, args=([suit_size])))

        # Get the results
        results = []
        for job in tqdm(jobs):
            results.append(job.get())

        # Its 8pm the pool is closed
        pool.close() 

        # Get the average coverage for this test suit size
        average_coverage_no_crashes.append(np.average(results))

        # Plot the data
        plt.scatter(np.full(len(results), suit_size), results, marker='*', c="C{}".format(i), s=0.5)


    # Save the results for correlation computation later
    all_coverage_data.append(all_results)

    # Plot the average test suit coverage
    plt.plot(test_suit_sizes, average_coverage, c="C{}".format(i), label="RSR{}".format(beam_number))

    # Plot the average test suit coverage
    plt.plot(test_suit_sizes, average_coverage_no_crashes, c="C{}".format(i), linestyle="--")

# -----------------------------
# Compute the code coverage
# -----------------------------
print("Processing code coverage")
average_code_coverage = []
average_code_coverage_no_crashes = []
for suit_size in test_suit_sizes:
    print("Processing test suit size: {}".format(suit_size))
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=args.cores)

    # Call our function total_test_suites times
    jobs = []
    for i in range(total_test_suits):
        jobs.append(pool.apply_async(code_coverage_on_random_test_suit, args=([suit_size, i])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Get the average code coverage for this test suit size
    average_code_coverage.append(np.average(results))

    plt.scatter(np.full(len(results), suit_size), results, marker='o', c="black", s=0.5)

    # Its 8pm the pool is closed
    pool.close() 

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=args.cores)

    # Call our function total_test_suites times
    jobs = []
    for i in range(total_test_suits):
        jobs.append(pool.apply_async(code_coverage_on_random_test_suit_no_crashes, args=([suit_size, i])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Get the average code coverage for this test suit size
    average_code_coverage_no_crashes.append(np.average(results))

    plt.scatter(np.full(len(results), suit_size), results, marker='*', c="black", s=0.5)

    # Its 8pm the pool is closed
    pool.close() 

# Plot the average test suit coverage
plt.plot(test_suit_sizes, average_code_coverage, c="black", label="CC")

# Plot the average test suit coverage
plt.plot(test_suit_sizes, average_code_coverage_no_crashes, c="black", linestyle="--")

ax = plt.gca()

# Shrink current axis's height by 5% on the top
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])

import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

# Place the first legend
num_cols = 4
handles, labels = ax.get_legend_handles_labels()
legend1 = ax.legend(flip(handles, num_cols), flip(labels, num_cols), loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=num_cols)

# Place the second legend
legend2 = ax.legend(["With and Without Crashes", "Without Crashes"] , loc=0)

# Set the labels and limits
ax.set_xlabel("Test suit size")
ax.set_ylabel("Coverage (%)")
ax.set_ylim([-0.05, 1.05])

# Readd legend 1 before showing
plt.gca().add_artist(legend1)
plt.show()