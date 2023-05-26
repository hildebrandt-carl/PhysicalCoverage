import re
import sys
import glob
import random
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/analysis/research_questions")])
sys.path.append(base_directory)

from utils.line_coverage_configuration import clean_branch_data
from utils.line_coverage_configuration import get_code_coverage
from utils.line_coverage_configuration import get_ignored_lines
from utils.line_coverage_configuration import get_ignored_branches
from utils.file_functions import get_beam_number_from_file
from utils.file_functions import order_files_by_beam_number

# multiple core
def random_selection(cores, test_suite_size, number_of_test_suites):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for i in range(number_of_test_suites):
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
    line_coverage_percentage    = results[0, :]
    branch_coverage_percentage  = results[1, :]
    unique_crash_count          = results[2, :]

    return line_coverage_percentage, branch_coverage_percentage, unique_crash_count

# Used to generated a random selection of tests
def random_select(number_of_tests):
    global lines_covered_per_test
    global branches_covered_per_test
    global code_coverage_denomiator
    global branch_coverage_denominator
    global unique_failure_set
    global crashes
    global stalls

    # Generate the indices for the random tests cases
    local_state = np.random.RandomState()
    indices = local_state.choice(lines_covered_per_test.shape[0], size=number_of_tests, replace=False)

    # Get the coverage and failure set
    line_coverage_set = set()
    branch_coverage_set = set()
    seen_failure_set = set()

    # Go through each of the different tests
    for i in indices:
        # Get the vectors
        l_cov = lines_covered_per_test[i]
        b_cov = branches_covered_per_test[i]
        crash = crashes[i]
        stall = stalls[i]

        # Add the line and branch coverage
        line_coverage_set   = line_coverage_set | l_cov
        branch_coverage_set = branch_coverage_set | b_cov

        # Check if there was a crash and if there was count it
        for c in crash:
            if c is not None:
                seen_failure_set.add(c)

        # Check if there was a stall and if there was count it
        for s in stall:
            if s is not None:
                seen_failure_set.add(s)

    # Compute the coverage and the crash percentage
    line_coverage_percentage    = (float(len(line_coverage_set)) / code_coverage_denomiator) * 100
    branch_coverage_percentage  = (float(len(branch_coverage_set)) / branch_coverage_denominator) * 100
    failures_found              = len(seen_failure_set)
    all_failures                = len(unique_failure_set)
    failure_percentage          = float(failures_found / all_failures) * 100

    return [line_coverage_percentage, branch_coverage_percentage, failure_percentage]


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',             type=str, default="/mnt/extradrive3/PhysicalCoverageData",       help="The location and name of the datafolder")
parser.add_argument('--number_of_test_suites', type=int, default=10,                                            help="The number of random test suites created")
parser.add_argument('--number_of_tests',       type=int, default=-1,                                            help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',          type=str, default="",                                            help="center_full/center_close")
parser.add_argument('--scenario',              type=str, default="",                                            help="beamng/highway")
parser.add_argument('--cores',                 type=int, default=4,                                             help="number of available cores")
parser.add_argument('--RRS',                   type=int, default=10,                                            help="Which RRS number you want to compute a correlation for")
args = parser.parse_args()

# Checking the distribution
if not (args.distribution == "center_full" or args.distribution == "center_close"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '{}/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.data_path, args.scenario, args.distribution, args.number_of_tests)
crash_file_names = glob.glob(base_path + "crash_*")
stall_file_names = glob.glob(base_path + "stall_*")

# Get the code coverage
base_path = '{}/{}/random_tests/code_coverage/processed/{}/'.format(args.data_path, args.scenario, args.number_of_tests)
global code_coverage_file_names
code_coverage_file_names = glob.glob(base_path + "*/*.txt")

# Holds the lines and branches covered per test
global lines_covered_per_test
global branches_covered_per_test
lines_covered_per_test      = np.full(args.number_of_tests, None, dtype="object")
branches_covered_per_test   = np.full(args.number_of_tests, None, dtype="object")

# Holds the denomiator for the code and branch coverage
global code_coverage_denomiator
global branch_coverage_denominator
code_coverage_denomiator = 0
branch_coverage_denominator = 0

# Select args.number_of_tests total code coverage files
assert(len(code_coverage_file_names) == args.number_of_tests)

global ignored_lines
global ignored_branches
ignored_lines       = set(get_ignored_lines(args.scenario))
ignored_branches    = set(get_ignored_branches(args.scenario))

# Get the feasible vectors
base_path = '{}/{}/feasibility/processed/{}/'.format(args.data_path, args.scenario, args.distribution)
feasible_file_names = glob.glob(base_path + "*.npy")

# Get the RRS numbers
crash_RRS_numbers = get_beam_number_from_file(crash_file_names)
stall_RRS_numbers = get_beam_number_from_file(stall_file_names)
feasibility_RRS_numbers = get_beam_number_from_file(feasible_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(crash_RRS_numbers) | set(stall_RRS_numbers) | set(feasibility_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
crash_file_names = order_files_by_beam_number(crash_file_names, RRS_numbers)
stall_file_names = order_files_by_beam_number(stall_file_names, RRS_numbers)
feasible_file_names = order_files_by_beam_number(feasible_file_names, RRS_numbers)

# Assume we are only doing RRS 10
i = args.RRS - 1

# Get the beam number and files we are currently considering
RRS_number = RRS_numbers[i]
crash_file = crash_file_names[i]
stall_file = stall_file_names[i]

# Select the correct crash files
global stalls
global crashes
stalls = np.load(stall_file, allow_pickle=True)
crashes = np.load(crash_file, allow_pickle=True)

# Create the failure unique set
global unique_failure_set
unique_failure_set = set()
for crash in crashes:
    for c in crash:
        if c is not None:
            unique_failure_set.add(c)
for stall in stalls:
    for s in stall:
        if s is not None:
            unique_failure_set.add(s)

# Get the total number of tests
total_tests = len(code_coverage_file_names)

# Keep track of all the different sets
all_lines_set           = set()
all_branches_set        = set()

# Start the multiprocessing
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
for f in code_coverage_file_names:
    jobs.append(pool.apply_async(get_code_coverage, args=([f])))

# Get the results
counter = 0
for job in tqdm(jobs):
    result = job.get()

    # Expand the data
    lines_covered           = result[0]
    all_lines               = result[1]
    branches_covered        = result[2]
    all_branches            = result[3]

    # Make sure converting to a set was done correctly
    assert(len(lines_covered)       == len(set(lines_covered)))
    assert(len(all_lines)           == len(set(all_lines)))
    assert(len(branches_covered)    == len(set(branches_covered)))
    assert(len(all_branches)        == len(set(all_branches)))

    # Save them to the overall sets
    lines_covered_set       = set(lines_covered)
    branches_covered_set    = set(branches_covered)
    all_lines_set           = all_lines_set         | set(all_lines)
    all_branches_set        = all_branches_set      | set(all_branches)

    # Clean the branch data
    if args.scenario == "highway":
        all_branches_set_clean, branches_covered_set_clean = clean_branch_data(all_branches_set, branches_covered_set)
    else:
        all_branches_set_clean = all_branches_set
        branches_covered_set_clean = branches_covered_set

    # Remove the ignored lines
    lines_covered_set           -= ignored_lines
    all_lines_set               -= ignored_lines
    all_branches_set_clean      -= ignored_branches
    branches_covered_set_clean  -= ignored_branches

    # Make sure it all makes sense
    assert(len(all_lines_set) == len(all_lines_set | lines_covered_set))
    assert(len(all_branches_set_clean) == len(all_branches_set_clean | branches_covered_set_clean))

    # Save the coverage
    lines_covered_per_test[counter]      = lines_covered_set
    branches_covered_per_test[counter]    = branches_covered_set_clean
    counter += 1

# Close the pool
pool.close()

# Create the total files set
code_coverage_denomiator    = len(all_lines_set)
branch_coverage_denominator = len(all_branches_set_clean)

test_suit_sizes = [10, 50, 100, 500, 1000, 5000]

f_l = open("line-{}.txt".format(args.scenario), "w")
f_b = open("branch-{}.txt".format(args.scenario), "w")

# Compute the correlation
for j, test_suite_size in enumerate(test_suit_sizes):
    print("Computing {} test suites of size {}".format(args.number_of_test_suites, test_suite_size))
    # Create random test suites
    results = random_selection(cores=args.cores,
                               test_suite_size=test_suite_size,
                               number_of_test_suites=args.number_of_test_suites)

    # Get the results
    line_coverage_percentage    = results[0]
    branch_coverage_percentage  = results[1]
    unique_crash_count          = results[2]

    # Compute the correlation
    l_r = stats.pearsonr(line_coverage_percentage, unique_crash_count)
    l_r_value = round(l_r[0], 4)
    l_p_value = round(l_r[1], 4)

    b_r = stats.pearsonr(branch_coverage_percentage, unique_crash_count)
    b_r_value = round(b_r[0], 4)
    b_p_value = round(b_r[1], 4)

    f_l.write("Test suite size: {}\n".format(test_suite_size))
    f_l.write("Line Trajectory Coverage             R value: {} - P value: {}\n".format(l_r_value, l_p_value))
    f_l.write("Line Average coverage: {}\n".format(np.average(line_coverage_percentage)))
    f_l.write("---------------------------------------------\n")

    f_b.write("Test suite size: {}\n".format(test_suite_size))
    f_b.write("Branch Trajectory Coverage coverage R value: {} - P value: {}\n".format(b_r_value, b_p_value))
    f_b.write("Branch Average coverage: {}\n".format(np.average(branch_coverage_percentage)))
    f_b.write("---------------------------------------------\n")

    print("Line coverage   R value: {} - P value: {}".format(l_r_value, l_p_value))
    print("Branch coverage R value: {} - P value: {}".format(b_r_value, b_p_value))
    print("Line Average coverage: {}".format(np.average(line_coverage_percentage)))
    print("Branch Average coverage: {}".format(np.average(branch_coverage_percentage)))

    # Plot the results
    plt.figure("Line")
    plt.scatter(line_coverage_percentage, unique_crash_count, color="C{}".format(j), label="Size: {} - Correlation: {}".format(test_suite_size, l_r_value))
    plt.figure("Branch")
    plt.scatter(branch_coverage_percentage, unique_crash_count, color="C{}".format(j), label="Size: {} - Correlation: {}".format(test_suite_size, b_r_value))
    print("---------------------------------------------")

f_l.close()
f_b.close()

plt.figure("Line")
plt.xlabel("Line Coverage (%)")
plt.ylabel("Unique Failures (%)")
plt.title("Line Coverage: {}".format(args.scenario))
plt.xlim([-5,100])
plt.ylim([-5,100])
plt.legend()

plt.figure("Branch")
plt.xlabel("Branch Coverage (%)")
plt.ylabel("Unique Failures (%)")
plt.title("Branch Coverage: {}".format(args.scenario))
plt.xlim([-5,100])
plt.ylim([-5,100])
plt.legend()

plt.show()