import ast
import glob
import copy
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from random import sample

from tqdm import tqdm
from collections import Counter
from prettytable import PrettyTable

from general_functions import order_by_beam
from general_functions import get_beam_numbers
from general_functions import get_ignored_code_coverage_lines

def compute_trace_signature_and_crash(index):
    global traces
    global crashes

    # Get the trace and crash data
    old_trace = traces[index]
    crash = crashes[index]

    trace = copy.deepcopy(old_trace)

    # Remove the center beam
    if np.shape(trace)[1] % 2 != 0:
        mid = int(np.shape(trace)[1] / 2)
        trace[:,mid] = 0
        if index == 0:
            print("Odd RSR, removing beam: {}".format(mid))

    # Init the trace signature and crash detected variable 
    trace_signature  = set()
    crash_detected   = False

    # The signature for the trace is the set of all RSR signatures
    for sig in trace:
        trace_signature.add(tuple(sig))

    # Create the hash of the signature for each comparison
    trace_hash = hash(tuple(sorted(trace_signature)))

    # Check if this trace had a crash
    crash_detected = not np.isinf(crash).all()

    return [trace_hash, crash_detected]


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

# Create the output table
t = PrettyTable()
t.field_names = ["Coverage Type", "Total Signatures" , "Single Test Signatures", "Multitest Signatures", "Consistent Multitest Signatures", "Inconsistent Multitest Signatures", "Percentage Consistent"]

# Loop through each of the files and compute both an RSR signature as well as determine if there was a crash
for beam_number in beam_numbers:
    print("Processing RSR{}".format(beam_number))
    key = "RSR{}".format(beam_number)

    # Get the trace and crash files
    global traces
    traces  = np.load(trace_file_names[beam_number-1])
    global crashes
    crashes = np.load(crash_file_names[beam_number-1])

    # Get the total number of tests
    total_number_of_tests = traces.shape[0]

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                =  multiprocessing.Pool(processes=total_processors)

    # Call our function on each test in the trace
    jobs = []
    for random_test_index in range(total_number_of_tests):
        jobs.append(pool.apply_async(compute_trace_signature_and_crash, args=([random_test_index])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        signature, crash_detected = r

        # Collect all the signatures
        all_signatures[i] = signature

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Final signatures holds the list of all signatures
    # Count of signatures holds the list intergers representing how many times each signature was seen

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Print the info
    print("Total signatures: {}".format(total_signatures_count))
    print("Total single test signatures: {}".format(single_test_signatures_count))
    print("Total multi test signatures: {}".format(multi_test_signatures_count))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

    # Add to the final table
    t.add_row([key, total_signatures_count, single_test_signatures_count, multi_test_signatures_count, "-", "-", "-"])

# Display the table
print(t)


