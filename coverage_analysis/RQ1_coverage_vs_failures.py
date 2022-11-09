import os
import sys
import ast
import glob
import hashlib
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

from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number
from general.environment_configurations import RRSConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics
from general.line_coverage_configuration import clean_branch_data
from general.line_coverage_configuration import get_code_coverage
from general.line_coverage_configuration import get_ignored_lines
from general.line_coverage_configuration import get_ignored_branches

def preprocessing_code_coverage_on_random_test_suite(num_cores):
    global lines_covered_per_test
    global branches_covered_per_test
    global code_coverage_file_names
    global code_coverage_denomiator
    global branch_coverage_denominator

    # Get the denominoators
    coverage_data = get_code_coverage(code_coverage_file_names[0])
    all_lines           = coverage_data[1]
    all_branches        = coverage_data[3]

    all_lines_set           = set(all_lines)
    all_branches_set        = set(all_branches)

    # Make sure the conversion worked
    assert(len(all_lines)           == len(all_lines_set))
    assert(len(all_branches)        == len(all_branches_set))

    # Remove the ignored lines
    all_lines_set -= ignored_lines
    all_branches_set -= ignored_branches

    # print("all_lines_set: {}".format(all_lines_set))
    # print("all_branches_set: {}".format(all_branches_set))

    # Get the denominator
    code_coverage_denomiator    = len(all_lines_set)
    branch_coverage_denominator = len(all_branches_set) 

    # This computes the line and branch numbers per file and stores them in the arrays
    # preprocessing
    total_number_of_tests = len(code_coverage_file_names)

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                = multiprocessing.Pool(processes=total_processors)

    print("Processing line and branch coverage")
    # Call our function on each file
    jobs = []
    for i in range(total_number_of_tests):
        jobs.append(pool.apply_async(get_line_branch_coverage, args=([i])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    l = set()
    b = set()

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        lines_covered_set, branches_covered_set = r

        l = l | lines_covered_set
        b = b | branches_covered_set

        # Collect all the signatures
        lines_covered_per_test[i]       = lines_covered_set
        branches_covered_per_test[i]    = branches_covered_set

    # print("Lines Covered = {}".format(l))
    # print("Branches Covered = {}".format(b))
    return True

def generate_code_coverage_on_random_test_suite(num_cores, number_test_suites):
   
    # Do preprocessing
    preprocessing_code_coverage_on_random_test_suite(num_cores)

    # Will hold the output
    line_coverage   = np.zeros((number_test_suites, len(code_coverage_file_names)))
    branch_coverage = np.zeros((number_test_suites, len(code_coverage_file_names)))

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=num_cores)
    jobs = []
    print("Generating random test suites")

    # Generate random test suites
    for i in range(number_test_suites):
        # Generate a random test suite
        jobs.append(pool.apply_async(random_test_suite_code_coverage))
        
    # Get the results:
    for i, job in enumerate(tqdm(jobs)):
        # Get the coverage
        results = job.get()
        line_coverage[i]    = results[0] 
        branch_coverage[i]  = results[1]

    # Close the pool
    pool.close()

    return line_coverage, branch_coverage

def get_line_branch_coverage(index):
    global lines_covered_per_test
    global branches_covered_per_test
    global code_coverage_file_names
    global ignored_lines
    global ignored_branches

    coverage_data = get_code_coverage(code_coverage_file_names[index])

    lines_covered           = coverage_data[0]
    branches_covered        = coverage_data[2]

    # Break up the branches
    new_branches_covered = set()
    for b in branches_covered:
        if "_" in b:
            new_branches_covered.add(b[:b.find("_")])
            new_branches_covered.add(b[b.find("_")+1:])
        else:
            new_branches_covered.add(b)

    branches_covered = new_branches_covered

    lines_covered_set       = set(lines_covered)
    branches_covered_set    = set(branches_covered)

    # Make sure the conversion worked
    assert(len(lines_covered)           == len(lines_covered_set))
    assert(len(branches_covered)        == len(branches_covered_set))

    # Remove the ignored lines
    lines_covered_set -= ignored_lines
    branches_covered_set -= ignored_branches

    return [lines_covered_set, branches_covered_set]

def random_test_suite_code_coverage():
    global lines_covered_per_test
    global branches_covered_per_test
    global code_coverage_denomiator
    global branch_coverage_denominator

    # Create the output
    code_coverage_output = np.zeros(len(code_coverage_file_names))
    branch_coverage_output = np.zeros(len(code_coverage_file_names))

    # Randomly generate the indices for this test suite
    local_state = np.random.RandomState()
    indices = local_state.choice(len(code_coverage_file_names), len(code_coverage_file_names), replace=False) 

    # Holds the current code and line coverage
    current_code_coverage = set()
    current_branch_coverage = set()

    # Go through each of the indices
    for i, index in enumerate(indices):

        # Get the coverage file
        line_coverage_set = lines_covered_per_test[index]
        branch_coverage_set = branches_covered_per_test[index]

        # Add this to the current code and branch coverage
        current_code_coverage   = current_code_coverage | line_coverage_set
        current_branch_coverage = current_branch_coverage | branch_coverage_set

        # Compute coverage
        code_coverage_output[i]     = (float(len(current_code_coverage)) / code_coverage_denomiator) * 100
        branch_coverage_output[i]   = (float(len(current_branch_coverage)) / branch_coverage_denominator) * 100

    return code_coverage_output, branch_coverage_output

# Creates a list of test signatures from a set of traces (returns a list of sets)
def create_list_of_test_signatures(num_cores):
    global traces

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=num_cores)
    jobs = []

    # Used to hold the RRS signature for each test
    signatures = []

    # Get the RRS set for each of the different tests
    for i, _ in enumerate(traces):
        # Compute the signatures for each trace
        jobs.append(pool.apply_async(compute_test_signature, args=([i])))
        
    # Get the results:
    for job in tqdm(jobs):
        signatures.append(job.get())

    # Close the pool
    pool.close()

    # Return the signatures
    return signatures

# Compute the test signature given the index of the test (returns a set)
def compute_test_signature(index):
    global traces

    # Create the RRS signature
    RRS_signature = set()   

    # Go through each scene and add it to the RRS set
    for scene in traces[index]:
        # Get the current scene
        s = tuple(scene)

        # Make sure that this is a scene (not a nan or inf or -1)
        if (np.isnan(scene).any() == False) and (np.isinf(scene).any() == False) and (np.less(scene, 0).any() == False):
            RRS_signature.add(tuple(s))



            # Give a warning if a vector is found that is not feasible
            if s not in feasible_RRS_set:
                print("Warning: Infeasible vector found: {}".format(scene))
                for v in feasible_RRS_set:
                    print(v)
                    print("----------")
    # Return the RRS signature
    return RRS_signature

# Get the coverage on a number_test_suites random test suites (returns a numpy array of coverage (coverage vs test suite size)) 
def generate_coverage_on_random_test_suites(num_cores, number_test_suites):
    global test_signatures

    # Create the output
    coverage = np.zeros((number_test_suites, len(test_signatures)))

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=num_cores)
    jobs = []

    # Generate random test suites
    for i in range(number_test_suites):
        # Generate a random test suite
        jobs.append(pool.apply_async(random_test_suite))
        
    # Get the results:
    for i, job in enumerate(tqdm(jobs)):
        # Get the coverage
        coverage[i] = job.get()
    # Close the pool
    pool.close()

    return coverage

# Get the number of crashes on a number_test_suites random test suites (returns a numpy array of coverage (crashes vs test suite size)) 
def compute_failures_on_random_test_suites(num_cores, number_test_suites):
    global crashes
    global stalls

    # Sanity check
    assert(len(crashes) == len(stalls))

    # Create the output
    failures = np.zeros((number_test_suites, len(crashes)))

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=num_cores)
    jobs = []

    # Generate random test suites
    for i in range(number_test_suites):
        # Generate a random test suite
        jobs.append(pool.apply_async(random_test_suite_failures))
        
    # Get the results:
    for i, job in enumerate(tqdm(jobs)):
        # Get the coverage
        failures[i] = job.get()
    # Close the pool
    pool.close()

    return failures

# Computes the coverage on a random test suite
def random_test_suite_failures():
    global crashes
    global stalls
    global total_failures

    # Create the output
    output = np.zeros(len(crashes))

    # Randomly generate the indices for this test suite
    local_state = np.random.RandomState()
    indices = local_state.choice(len(crashes), len(crashes), replace=False) 

    # Used to hold the seen RRS
    seen_failures = set()

    # Go through each of the indices
    for i, index in enumerate(indices):
        
        # Add each crash to the seen_failures set
        for c in crashes[index]:
            if c is not None:
                seen_failures.add(c)
            else:
                break

        # Add each crash to the seen_failures set
        for s in stalls[index]:
            if s is not None:
                seen_failures.add(s)
            else:
                break

        # Failure percentage over all tests
        output[i] = (len(seen_failures) / total_failures) * 100

    return output

# Computes the coverage on a random test suite
def random_test_suite():
    global test_signatures
    global feasible_RRS_set

    # Compute the denominator for the coverage
    denominator = len(feasible_RRS_set)

    # Create the output
    output = np.zeros(len(test_signatures))

    # Randomly generate the indices for this test suite
    local_state = np.random.RandomState()
    indices = local_state.choice(len(test_signatures), len(test_signatures), replace=False) 

    # Used to hold the seen RRS
    seen_RRS = set()

    # Go through each of the indices
    for i, index in enumerate(indices):
        # Get the trace
        RRS_sig = test_signatures[index]
        # Add this to the seen RRS set
        seen_RRS = seen_RRS | RRS_sig
        # Compute coverage
        output[i] = (float(len(seen_RRS)) / denominator) * 100

    return output

# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/media/carl/DataDrive/PhysicalCoverageData",     help="The location and name of the datafolder")
parser.add_argument('--random_test_suites', type=int, default=10,                                               help="The number of random line samples used")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",                                               help="beamng/highway")
parser.add_argument('--cores',              type=int, default=4,                                                help="number of available cores")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
NG = BeamNGKinematics()
RRS = RRSConfig()

# Save the kinematics and RRS parameters
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
base_path = '{}/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.data_path, args.scenario, args.distribution, args.number_of_tests)
trace_file_names = glob.glob(base_path + "traces_*")
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
trace_RRS_numbers = get_beam_number_from_file(trace_file_names)
crash_RRS_numbers = get_beam_number_from_file(crash_file_names)
stall_RRS_numbers = get_beam_number_from_file(stall_file_names)
feasibility_RRS_numbers = get_beam_number_from_file(feasible_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(trace_RRS_numbers) | set(crash_RRS_numbers) | set(stall_RRS_numbers) | set(feasibility_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
trace_file_names = order_files_by_beam_number(trace_file_names, RRS_numbers)
crash_file_names = order_files_by_beam_number(crash_file_names, RRS_numbers)
stall_file_names = order_files_by_beam_number(stall_file_names, RRS_numbers)
feasible_file_names = order_files_by_beam_number(feasible_file_names, RRS_numbers)

# Create the output figure
plt.figure(1)

# Compute the code coverage
computed_line_coverage, computed_branch_coverage = generate_code_coverage_on_random_test_suite(num_cores=args.cores,
                                                                                               number_test_suites=args.random_test_suites)

average_line_coverage = np.average(computed_line_coverage, axis=0)
line_coverage_upper_bound = np.max(computed_line_coverage, axis=0)
line_coverage_lower_bound = np.min(computed_line_coverage, axis=0)
average_branch_coverage = np.average(computed_branch_coverage, axis=0)
branch_coverage_upper_bound = np.max(computed_branch_coverage, axis=0)
branch_coverage_lower_bound = np.min(computed_branch_coverage, axis=0)

x = np.arange(0, np.shape(computed_line_coverage)[1])
# Plot the results
plt.fill_between(x, line_coverage_lower_bound, line_coverage_upper_bound, alpha=0.2, color="black") #this is the shaded error
plt.plot(x, average_line_coverage, c="black", label="Line Cov", linestyle="dashed") #this is the line itself
plt.fill_between(x, branch_coverage_lower_bound, branch_coverage_upper_bound, alpha=0.2, color="black") #this is the shaded error
plt.plot(x, average_branch_coverage, c="black", label="Branch Cov", linestyle="dotted") #this is the line itself

# For each of the different RRS_numbers
for i in range(len(RRS_numbers)):
    print("Processing RRS: {}".format(RRS_numbers[i]))

    # Get the beam number and files we are currently considering
    RRS_number = RRS_numbers[i]
    trace_file = trace_file_names[i]
    crash_file = crash_file_names[i]
    stall_file = stall_file_names[i]
    feasibility_file = feasible_file_names[i]

    # Skip if any of the files are blank
    if trace_file == "" or crash_file == "" or stall_file == "" or feasibility_file == "":
        print(feasibility_file)
        print(crash_file)
        print(stall_file)
        print(trace_file)
        print("\nWarning: Could not find one of the files for RRS number: {}".format(RRS_number))
        continue

    # Load the feasibility file
    feasible_traces = np.load(feasibility_file)

    # Create the feasible set
    global feasible_RRS_set
    feasible_RRS_set = set()
    for scene in feasible_traces:
        feasible_RRS_set.add(tuple(scene))

    # Load the traces
    global traces
    traces = np.load(trace_file)

    # Do the preprocessing
    global test_signatures
    test_signatures = create_list_of_test_signatures(num_cores=args.cores)

    # Create the output figure
    plt.figure(1)

    print("Generating random test suites")
    computed_coverage = generate_coverage_on_random_test_suites(num_cores=args.cores, 
                                                                number_test_suites=args.random_test_suites)

    # Compute the results
    average_coverage = np.average(computed_coverage, axis=0)
    upper_bound = np.max(computed_coverage, axis=0)
    lower_bound = np.min(computed_coverage, axis=0)
    x = np.arange(0, np.shape(computed_coverage)[1])

    # Plot the results
    plt.fill_between(x, lower_bound, upper_bound, alpha=0.2, color="C{}".format(i)) #this is the shaded error
    plt.plot(x, average_coverage, c="C{}".format(i), label="$\Psi_{" + str(RRS_number) + "}$") #this is the line itself

# Load the stall and crash file
global stalls
global crashes

stalls = np.load(stall_file, allow_pickle=True)
crashes = np.load(crash_file, allow_pickle=True)

# Compute the total number of failures
global total_failures
total_failures = 0
for stall in stalls:
    for s in stall:
        if s is not None:
            total_failures += 1
        else:
            break
for crash in crashes:
    for c in crash:
        if c is not None:
            total_failures += 1
        else:
            break

# Add crashes
print("Computing failures")
computed_failures = compute_failures_on_random_test_suites(num_cores=args.cores, 
                                                           number_test_suites=args.random_test_suites)

# Compute the results
average_failures    = np.average(computed_failures, axis=0)
upper_bound         = np.max(computed_failures, axis=0)
lower_bound         = np.min(computed_failures, axis=0)
x                   = np.arange(0, np.shape(computed_failures)[1])

# Plot the results on a second axis
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot the results
ax2.fill_between(x, lower_bound, upper_bound, alpha=0.2, color="black") #this is the shaded error
ax2.plot(x, average_failures, color="black", label="Unique Failures", linestyle="solid") #this is the line itself

plt.title(args.distribution)
ax1.grid(alpha=0.5)
ax1.set_yticks(np.arange(0, 100.01, step=5))
ax1.set_xlabel("Number of tests")
ax1.set_ylabel("Coverage (%)")
ax2.set_ylabel("Failures (%)")
ax2.set_yticks(np.arange(0, 100.01, step=5))
ax1.legend(ncol=6)
ax2.legend(loc="lower center", ncol=3)

plt.show()