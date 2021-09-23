import ast
import glob
import argparse
import multiprocessing

import numpy as np
from random import sample

from tqdm import tqdm
from prettytable import PrettyTable

from general_functions import order_by_beam
from general_functions import get_beam_numbers
from general_functions import get_ignored_code_coverage_lines

def compute_physical_coverage_hash(index):

    # Get the file
    trace = traces[index]
    crash = crashes[index]

    # Init the variables
    coverage_hash = None
    number_of_crashes = 0

    # Hash the RSR vectors
    flattened_trace = tuple(list(trace.reshape(-1)))
    coverage_hash = hash(flattened_trace) 

    # Count the number of crashes
    for c in crash:
        if ~np.isinf(c):
            number_of_crashes += 1

    return [coverage_hash, number_of_crashes]

def compute_code_coverage_hash(index):
    global code_coverage_file_names

    coverage_hash = 0
    number_of_crashes = 0

    # Get the code coverage file
    code_coverage_file = code_coverage_file_names[index]
    f = open(code_coverage_file, "r")

    all_lines_coverage = set()

    # Read the file
    for line in f: 
        if "Lines covered:" in line:
            covered_l = ast.literal_eval(line[15:])

        if "Total lines covered:" in line:
            total_covered_l = int(line[21:])
            assert(len(covered_l) == total_covered_l)

        if "Total physical crashes: " in line:
            number_of_crashes = int(line[24:])

    # Close the file
    f.close()

    all_lines_coverage = set(covered_l) - ignored_lines

    # Get the coverage hash
    all_lines_coverage = tuple(sorted(list(all_lines_coverage)))
    coverage_hash = hash(all_lines_coverage)

    return [coverage_hash, number_of_crashes]

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = "*.npy"

# Get the file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/physical_coverage/processed/' + str(args.total_samples) + "/"
trace_file_names = glob.glob(base_path + "traces_*.npy")
crash_file_names = glob.glob(base_path + "crash_*.npy")

# Get the code coverage
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/code_coverage/raw/'
global code_coverage_file_names
code_coverage_file_names = glob.glob(base_path + "*/*.txt")

# Make sure we have enough samples
assert(len(trace_file_names) >= 1)
assert(len(crash_file_names) >= 1)
assert(len(code_coverage_file_names) >= 1)

# Select args.total_samples total code coverage files
code_coverage_file_names = sample(code_coverage_file_names, args.total_samples)

global ignored_lines
ignored_lines = get_ignored_code_coverage_lines(args.scenario)

# Get the beam numbers
trace_beam_numbers = get_beam_numbers(trace_file_names)
crash_beam_numbers = get_beam_numbers(crash_file_names)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(trace_beam_numbers) | set(crash_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
trace_file_names = order_by_beam(trace_file_names, beam_numbers)
crash_file_names = order_by_beam(crash_file_names, beam_numbers)

# Used to save the final results
final_results = {}

# For each of the different beams
for i in range(len(beam_numbers)):

    if beam_numbers[i] > 6:
        continue

    print("Processing RSR{}".format(beam_numbers[i]))

    # Get the beam number and files we are currently considering
    beam_number = beam_numbers[i]
    trace_file = trace_file_names[i]
    crash_file = crash_file_names[i]

    # Load the traces
    global traces
    global crashes
    traces = np.load(trace_file)
    crashes = np.load(crash_file)

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=args.cores)
    jobs = []

    # Go through each of the different test suite sizes
    for i in range(len(traces)):
        jobs.append(pool.apply_async(compute_physical_coverage_hash, args=([i])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    crashing_tests = 0
    non_crashing_tests = 0
    crash_count = 0

    # Put the results into the final results dict
    coverage_hashes = []
    crash_numbers = []
    for r in results:
        coverage_hashes.append(r[0])
        crash_numbers.append(r[1])
        crash_count += r[1]
        # If there was a crash
        if r[1] >= 1:
            crashing_tests += 1
        else:
            non_crashing_tests += 1
        
    final_results["RSR{}".format(beam_number)] = [coverage_hashes, crash_numbers, crashing_tests, non_crashing_tests, crash_count]

    # Its 8pm the pool is closed
    pool.close() 

# Create the pool for parallel processing
pool =  multiprocessing.Pool(processes=args.cores)
jobs = []

# Go through each of the different test suite sizes
print("Processing Code Coverage")

for i in range(len(code_coverage_file_names)):
    jobs.append(pool.apply_async(compute_code_coverage_hash, args=([i])))

# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

crashing_tests = 0
non_crashing_tests = 0
crash_count = 0

# Put the results into the final results dict
coverage_hashes = []
crash_numbers = []
for r in results:
    coverage_hashes.append(r[0])
    crash_numbers.append(r[1])
    crash_count += r[1]
    # If there was a crash
    if r[1] >= 1:
        crashing_tests += 1
    else:
        non_crashing_tests += 1

    
final_results["CC"] = [coverage_hashes, crash_numbers, crashing_tests, non_crashing_tests, crash_count]

# Its 8pm the pool is closed
pool.close() 

# Create the output table
t = PrettyTable()
t.field_names = ["Coverage Type", "T" , "P T", "F T", "Unique T", "Not Unique T", "Unique P T", "Unique F T", "Not Unique P T", "Not Unique F T"]
s = PrettyTable()
s.field_names = ["Coverage Type", "S", "Unique S", "Not Unique S", "Unique S P", "Unique S F", "Not Unique S All P", "Not Unique S All F", "Not Unique S PF"]

# Print the results out
for key in sorted(final_results):

    print("Processing: {}".format(key))

    # Get the data
    data = final_results[key]
    coverage_hashes = np.array(data[0]).reshape(-1)
    crash_results = np.array(data[1]).reshape(-1)
    crashing_tests = int(data[2])
    non_crashing_tests = int(data[3])
    crash_count = int(data[4])

    debug_count = 0

    # Find the unique crashes
    unique_values, unique_indexes, counts = np.unique(coverage_hashes, return_inverse=True, return_counts=True)

    # Count the number of unique and not unique values
    total_tests = len(coverage_hashes)
    total_unique_tests = len(unique_values[counts == 1])
    total_non_unique_tests = np.sum(counts[counts > 1])
    total_non_unique_signatures = len(unique_values[counts > 1])

    total_signatures = len(unique_values)
    total_unique_signatures = np.sum(counts[counts == 1])

    # Count the number of passing and failing and mixed results
    not_unique_sig_all_passing_count = 0
    not_unique_sig_all_failing_count = 0
    not_unique_sig_mixed_count = 0
    unique_tests_failing = 0
    unique_tests_passing = 0
    not_unique_tests_failing = 0 
    not_unique_tests_passing = 0
    passing_test_count = 0
    failing_test_count = 0

    failing_test_count = np.sum(crash_results)
    passing_test_count = len(crash_results) - np.sum(crash_results)

    # For each of the values that aren't unique
    for i in range(len(unique_values)):
        test_outputs = crash_results[np.where(unique_indexes == i)]

        # Number of failing tests
        z_count = np.count_nonzero(test_outputs)

        # If there is more than 1 output we know that it wasn't a unique coverage signature
        if len(test_outputs) > 1:
            
            # Number of tests
            t_count = len(test_outputs)

            # Count all passing
            if z_count == 0:
                not_unique_sig_all_passing_count += 1
                not_unique_tests_passing += len(test_outputs)

            # Count all failing
            elif z_count == t_count:
                not_unique_sig_all_failing_count += 1
                not_unique_tests_failing += len(test_outputs)

            # Count mix
            elif 0 < z_count < t_count:
                not_unique_sig_mixed_count += 1
                not_unique_tests_passing += len(test_outputs) - z_count
                not_unique_tests_failing += z_count

            # Else
            else:
                print("Error 1")
                exit()

        # We know this test is unique
        else:

            z_count = np.count_nonzero(test_outputs)

            if z_count == 0:
                unique_tests_passing += 1
            elif z_count == 1:
                unique_tests_failing += 1

            # Else
            else:
                print("Error 2")
                exit()

    # Check your math son
    assert(total_unique_tests + total_non_unique_tests == total_tests)
    assert(not_unique_sig_all_passing_count + not_unique_sig_all_failing_count + not_unique_sig_mixed_count == total_non_unique_signatures)

    # Print the output
    print("Not unique: {}".format(total_unique_tests))
    print("Unique: {}".format(total_non_unique_tests))
    
    t.add_row([key, total_tests, passing_test_count, failing_test_count, total_unique_tests, total_non_unique_tests, unique_tests_passing, unique_tests_failing, not_unique_tests_passing, not_unique_tests_failing])
    s.add_row([key, total_signatures, total_unique_signatures, total_non_unique_signatures, unique_tests_passing, unique_tests_failing, not_unique_sig_all_passing_count, not_unique_sig_all_failing_count, not_unique_sig_mixed_count])


# Display the table
print("\n\n\n")
print("Note: The code coverage will only match the RSR values if you are using the whole test set, as the random samples were selected at different times")
print("Tests")
print(t)
print("Signatures")
print(s)

print("Definitions of each field:")
print("T: The total number of tests executed")
print("P T: The total number of tests executed that were passing (no crashes)")
print("F T: The total number of tests executed that were failing (crashes at least once)")
print("Unique T: The total number of tests that have a unique signature (i.e. there is only 1 test with that signature)")
print("Not Unique T: The total number of tests that have do not have a unique signature (i.e. there is more than 1 test with that signature)")
print("Unique P T: The number of unique tests (The only test with that signature) that are passing")
print("Unique F T: The number of unique tests (The only test with that signature) that are failing")
print("Not Unique P T: The number of not unique tests (A test that shares a signature with at least 1 other test) that are passing")
print("Not Unique F T: The number of not unique tests (A test that shares a signature with at least 1 other test) that are failing")

print("S: The total number of signatures generated over all tests")
print("Unique S: The total number of signatures that were only found by 1 test")
print("Not Unique S: The total number of signautres that were found in 2 or more tests")
print("Unique S P: Of the signatures that were only found by 1 tests, how many were passing tests")
print("Unique S F: Of the signatures that were only found by 1 tests, how many were failing tests")
print("Not Unique S All P: Of the signatures that were only found in 2 or more tests, how many of those signatures had all passing tests")
print("Not Unique S All F: Of the signatures that were only found in 2 or more tests, how many of those signatures had all failing tests")
print("Not Unique S PF: Of the signatures that were only found in 2 or more tests, how many of those signatures had both passing and failing tests")