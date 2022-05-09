import os
import sys
import ast
import glob
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

# Creates a list of test signatures from a set of random_traces (returns a list of sets)
def create_list_of_test_signatures(num_cores):
    global random_traces

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=num_cores)
    jobs = []

    # Used to hold the RRS signature for each test
    signatures = []

    # Get the RRS set for each of the different tests
    for i, _ in enumerate(random_traces):
        # Compute the signatures for each trace
        jobs.append(pool.apply_async(compute_test_signature, args=([i, True])))
        
    # Get the results:
    for job in tqdm(jobs):
        signatures.append(job.get())

    # Close the pool
    pool.close()

    # Return the signatures
    return signatures

# Compute the test signature given the index of the test (returns a set)
def compute_test_signature(index, random=True):
    global random_traces
    global generated_traces

    # Create the RRS signature
    RRS_signature = set()   

    if random:
        current_traces = random_traces
    else:
        current_traces = generated_traces

    # Go through each scene and add it to the RRS set
    for scene in current_traces[index]:
        # Get the current scene
        s = tuple(scene)

        # Make sure that this is a scene (not a nan or inf or -1)
        if (np.isnan(scene).any() == False) and (np.isinf(scene).any() == False) and (np.less(scene, 0).any() == False):
            
            # Give a warning if a vector is found that is not feasible
            if s not in feasible_RRS_set:
                print("Warning: Infeasible vector found: {}".format(scene))
                for v in feasible_RRS_set:
                    print(v)
                    print("----------")
            else:
                RRS_signature.add(tuple(s))
                
    # Return the RRS signature
    return RRS_signature

# Get the coverage on a number_test_suites random test suites (returns a numpy array of coverage (coverage vs test suite size)) 
def generate_coverage_on_random_test_suites(num_cores, number_test_suites):
    global test_signatures

    # ----------------------------------------------
    # First compute coverage for random tests suites
    # ----------------------------------------------
    # Create the output
    random_coverage = np.zeros((number_test_suites, len(test_signatures)))

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
        random_coverage[i] = job.get()
    # Close the pool
    pool.close()

    # ----------------------------------------------
    # Now compute coverage for generated tests suites
    # ----------------------------------------------
    # Create the output
    generated_coverage = generated_test_suite(num_cores)

    return random_coverage, generated_coverage

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

# Computes the coverage for a generated test suite
def generated_test_suite(num_cores):
    global random_traces
    global generated_traces
    global feasible_RRS_set

    print("Computing generated tests")

    # Compute the denominator for the coverage
    denominator = len(feasible_RRS_set)

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=num_cores)
    jobs = []

    # Used to hold the RRS signature for each test
    signatures = []

    # Get the RRS set for each of the different tests
    for i, _ in enumerate(random_traces):
        # Compute the signatures for each trace
        jobs.append(pool.apply_async(compute_test_signature, args=([i, True])))
        
    # Get the results:
    for job in tqdm(jobs):
        signatures.append(job.get())

    # Close the pool
    pool.close()

    # Compute the set of seen signatures
    seen_RRS = set()
    for sig in signatures:
        for s in sig:
            seen_RRS.add(s)

    output = np.zeros(len(generated_traces))

    # See how many signatures are added based on the new generated tests
    for i in range(len(generated_traces)):
        sig = compute_test_signature(index=i,random=False)
        for s in sig:
            seen_RRS.add(s)
        output[i] = (float(len(seen_RRS)) / denominator) * 100

    return output


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--random_test_suites', type=int, default=10,   help="The number of random line samples used")
parser.add_argument('--number_of_tests',    type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",   help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',              type=int, default=4,    help="number of available cores")
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
base_path = '/media/carl/DataDrive/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
random_trace_file_names = glob.glob(base_path + "traces_*")
crash_file_names        = glob.glob(base_path + "crash_*")
stall_file_names        = glob.glob(base_path + "stall_*")

# Get the feasible vectors
base_path = '/media/carl/DataDrive/PhysicalCoverageData/{}/feasibility/processed/{}/'.format(args.scenario, args.distribution)
feasible_file_names = glob.glob(base_path + "*.npy")

# Get the generated test file names
base_path = '/media/carl/DataDrive/PhysicalCoverageData/{}/generated_tests/{}/physical_coverage/processed/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
generated_trace_file_names = glob.glob(base_path + "traces_*")

# Get the RRS numbers
random_trace_RRS_numbers    = get_beam_number_from_file(random_trace_file_names)
generated_trace_RRS_numbers = get_beam_number_from_file(generated_trace_file_names)
crash_RRS_numbers           = get_beam_number_from_file(crash_file_names)
stall_RRS_numbers           = get_beam_number_from_file(stall_file_names)

feasibility_RRS_numbers = get_beam_number_from_file(feasible_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(random_trace_RRS_numbers) | set(generated_trace_RRS_numbers) | set(crash_RRS_numbers) | set(stall_RRS_numbers) | set(feasibility_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
random_trace_file_names         = order_files_by_beam_number(random_trace_file_names, RRS_numbers)
generated_trace_file_names      = order_files_by_beam_number(generated_trace_file_names, RRS_numbers)
crash_file_names                = order_files_by_beam_number(crash_file_names, RRS_numbers)
stall_file_names                = order_files_by_beam_number(stall_file_names, RRS_numbers)
feasible_file_names             = order_files_by_beam_number(feasible_file_names, RRS_numbers)

# Create the output figure
plt.figure(1)

# For each of the different beams
for i in range(len(RRS_numbers)):
    print("Processing RRS: {}".format(RRS_numbers[i]))

    # Get the beam number and files we are currently considering
    RRS_number              = RRS_numbers[i]
    random_trace_file       = random_trace_file_names[i]
    generated_trace_file    = generated_trace_file_names[i]
    crash_file              = crash_file_names[i]
    stall_file              = stall_file_names[i]
    feasibility_file        = feasible_file_names[i]

    # Skip if any of the files are blank
    if random_trace_file == "" or generated_trace_file == "" or crash_file == "" or stall_file == "" or feasibility_file == "":
        print(feasibility_file)
        print(generated_trace_file)
        print(crash_file)
        print(stall_file)
        print(random_trace_file)
        print("\nWarning: Could not find one of the files for RRS number: {}".format(RRS_number))
        continue

    # Load the feasibility file
    feasible_traces = np.load(feasibility_file)

    # Create the feasible set
    global feasible_RRS_set
    feasible_RRS_set = set()
    for scene in feasible_traces:
        feasible_RRS_set.add(tuple(scene))

    # Load the random_traces
    global random_traces
    random_traces = np.load(random_trace_file)

    # Load the generated_traces
    global generated_traces
    generated_traces = np.load(generated_trace_file)

    # Do the preprocessing
    global test_signatures
    test_signatures = create_list_of_test_signatures(num_cores=args.cores)

    # Create the output figure
    plt.figure(1)

    print("Generating random test suites")
    random_coverage, generated_coverage = generate_coverage_on_random_test_suites(num_cores=args.cores, 
                                                                                  number_test_suites=args.random_test_suites)

    # Compute the results
    average_coverage    = np.average(random_coverage, axis=0)

    # Link the average to the generated
    generated_coverage = np.hstack((average_coverage[-1], generated_coverage))

    # Compute the other metrics
    upper_bound         = np.max(random_coverage, axis=0)
    lower_bound         = np.min(random_coverage, axis=0)

    # Generate the x arrays
    x_random            = np.arange(0, np.shape(random_coverage)[1])
    x_generated         = np.arange(0, np.shape(generated_coverage)[0]) + x_random[-1]

    # Plot the results
    plt.fill_between(x_random, lower_bound, upper_bound, alpha=0.2, color="C{}".format(i)) #this is the shaded error
    plt.plot(x_random, average_coverage, c="C{}".format(i), label="RRS{}".format(RRS_number)) #this is the line itself
    plt.plot(x_generated, generated_coverage, c="C{}".format(i), linewidth=3) #this is the line itself

plt.axvline(args.number_of_tests, c="red", linestyle='--')


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
ax2.plot(x, average_failures, color="black", label="Failures", linestyle="--") #this is the line itself

plt.title(args.distribution)
ax1.grid(alpha=0.5)
ax1.set_yticks(np.arange(0, 100.01, step=5))
ax1.set_xlabel("Number of tests")
ax1.set_ylabel("Coverage (%)")
ax2.set_ylabel("Unique Failures (%)")
ax2.set_yticks(np.arange(0, 100.01, step=5))
ax1.legend()
ax2.legend(loc="lower center")

plt.show()