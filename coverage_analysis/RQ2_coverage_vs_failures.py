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
from general.environment_configurations import RSRConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics

# Creates a list of test signatures from a set of traces (returns a list of sets)
def create_list_of_test_signatures(num_cores):
    global traces

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=num_cores)
    jobs = []

    # Used to hold the RRS signature for each test
    signatures = []

    # Get the RSR set for each of the different tests
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

    # Go through each scene and add it to the RSR set
    for scene in traces[index]:
        # Get the current scene
        s = tuple(scene)

        # Make sure that this is a scene (not a nan or inf or -1)
        if (np.isnan(scene).any() == False) and (np.isinf(scene).any() == False) and (np.less(scene, 0).any() == False):
            RRS_signature.add(tuple(s))

            # Give a warning if a vector is found that is not feasible
            if s not in feasible_RRS_set:
                print("Warning: Infeasible vector found: {}".format(scene))
    
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
parser.add_argument('--random_test_suites', type=int, default=10,   help="The number of random line samples used")
parser.add_argument('--number_of_tests',    type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",   help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',              type=int, default=4,    help="number of available cores")
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
trace_file_names = glob.glob(base_path + "traces_*")
crash_file_names = glob.glob(base_path + "crash_*")
stall_file_names = glob.glob(base_path + "stall_*")

# Get the feasible vectors
base_path = '/media/carl/DataDrive/PhysicalCoverageData/{}/feasibility/processed/{}/'.format(args.scenario, args.distribution)
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

# For each of the different beams
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
    global stalls
    global crashes
    traces = np.load(trace_file)
    stalls = np.load(stall_file)
    crashes = np.load(crash_file)

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
    plt.plot(x, average_coverage, c="C{}".format(i)) #this is the line itself

    # Create the legend
    plt.plot([], c="C{}".format(i), label="RRS{}".format(RRS_number))
    

plt.grid(alpha=0.5)
plt.title(args.distribution)
plt.yticks(np.arange(0, 100.01, step=5))
plt.xlabel("Number of tests")
plt.ylabel("Coverage / Failure (%)")
plt.legend()
plt.show()