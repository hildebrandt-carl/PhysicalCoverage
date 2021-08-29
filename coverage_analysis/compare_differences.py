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
trace_file_names = glob.glob(base_path + "traces_*")
crash_file_names = glob.glob(base_path + "crash_*")

# Get the code coverage
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/code_coverage/raw/'
global code_coverage_file_names
code_coverage_file_names = glob.glob(base_path + "*/*.txt")

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

    # Go through each of the different test suit sizes
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

# Go through each of the different test suit sizes
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
x = PrettyTable()
x.field_names = ["Coverage Type", "# Unique", "# Not Unique", "# Same Cov, Same Out", "# Same Cov, Diff Out", "# None Crashing Tests", "# Crashing Tests", "# Crashes", "# Tests"]

# Print the results out
for key in final_results:
    same_cov_diff_output_count = 0
    same_cov_same_output_count = 0

    print("Processing: {}".format(key))

    # Get the data
    data = final_results[key]
    coverage_hashes = np.array(data[0]).reshape(-1)
    crash_numbers = np.array(data[1]).reshape(-1)
    crashing_tests = int(data[2])
    non_crashing_tests = int(data[3])
    crash_count = int(data[4])

    # Used to keep track of which signatures I have already checked
    not_unique = set()
    not_unique_count = 0
    
    debug_count = 0

    # Find the percentage of unique coverage signatures
    unique_signatures = set()
    for i in range(len(coverage_hashes)):
        # Get the coverage signature, and crash number
        cov = coverage_hashes[i]
        out = crash_numbers[i]
        # See if the crash signature is unique
        if cov not in unique_signatures:
            unique_signatures.add(cov)

        # For the ones that aren't unique, determine if the have the same of different output
        else:
            not_unique_count += 1
            if cov not in not_unique:
                not_unique.add(cov)
                # Find all the matches in the original data
                index_matches = np.argwhere(coverage_hashes==cov).reshape(-1)
                
                # Get the output for these indices
                matched_output = crash_numbers[index_matches]

                # Find what the majority is (mostly 1's):
                if np.sum(matched_output) >= (len(matched_output) / 2):
                    # Find out how many are 1's
                    same_cov_same_output_count += np.count_nonzero(matched_output)
                    same_cov_diff_output_count += len(matched_output) - np.count_nonzero(matched_output)
                # (mostly 0's)
                else:
                    # Find out how many are 0's
                    same_cov_same_output_count += len(matched_output) - np.count_nonzero(matched_output)
                    same_cov_diff_output_count += np.count_nonzero(matched_output)

    # We need to add on the not unique values that were originally thought unique:
    not_unique_count += len(not_unique)
    # We need to remove the non_unique_signatures that were originally thought unique
    unique_signatures = unique_signatures - not_unique

    # Check your math son
    assert(not_unique_count + len(unique_signatures) == args.total_samples)

    # Print the output
    print("Not unique: {}".format(not_unique_count))
    print("Unique: {}".format(len(unique_signatures)))
    print("same_cov_same_output_count: {}".format(same_cov_same_output_count))
    print("same_cov_diff_output_count: {}".format(same_cov_diff_output_count))
    
    # Check the math holds out
    total_tests = len(unique_signatures) + same_cov_same_output_count + same_cov_diff_output_count
    print("Non Crashing Tests: {}".format(total_tests))
    print("Crashing Tests: {}".format(total_tests))
    print("total_tests: {}".format(total_tests))
    print('--------')

    crashing_tests = int(data[2])
    non_crashing_tests = int(data[3])
    crash_count = int(data[4])

    x.add_row([key, len(unique_signatures), not_unique_count, same_cov_same_output_count, same_cov_diff_output_count, non_crashing_tests, crashing_tests, crash_count, args.total_samples])

# Display the table
print("\n\n\n")
print("Note: The code coverage will only match the RSR values if you are using the whole test set, as the random samples were selected at different times")
print(x)