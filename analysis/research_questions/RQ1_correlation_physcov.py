import sys
import glob
import hashlib
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/analysis/research_questions")])
sys.path.append(base_directory)

from tqdm import tqdm
from scipy import stats
from matplotlib_venn import venn2

import numpy as np
import matplotlib.pyplot as plt

from utils.file_functions import get_beam_number_from_file
from utils.file_functions import order_files_by_beam_number
from utils.environment_configurations import RRSConfig
from utils.environment_configurations import BeamNGKinematics
from utils.environment_configurations import HighwayKinematics

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
    coverage_percentages = results[0, :]
    unique_crash_count   = results[1, :]

    return coverage_percentages, unique_crash_count

# Used to generated a random selection of tests
def random_select(number_of_tests):
    global traces
    global crashes
    global stalls
    global feasible_RRS_set
    global unique_failure_set

    # Generate the indices for the random tests cases
    local_state = np.random.RandomState()
    indices = local_state.choice(traces.shape[0], size=number_of_tests, replace=False)

    # Get the coverage and failure set
    seen_RRS_set = set()
    seen_failure_set = set()

    # Go through each of the different tests
    for i in indices:
        # Get the vectors
        vectors = traces[i]
        crash = crashes[i]
        stall = stalls[i]

        # Go through the traces and compute coverage
        for v in vectors:

            # Make sure that this is a scene (not a nan or inf or -1 or None)
            if v is not None:
                if (np.isnan(v).any() == False) and (np.isinf(v).any() == False) and (np.less(v, 0).any() == False):
                    seen_RRS_set.add(tuple(v))

        # Check if there was a crash and if there was count it
        for c in crash:
            if c is not None:
                seen_failure_set.add(c)

        # Check if there was a stall and if there was count it
        for s in stall:
            if s is not None:
                seen_failure_set.add(s)

    # Compute the coverage and the crash percentage
    coverage_percentage = (float(len(seen_RRS_set)) / len(feasible_RRS_set)) * 100
    failures_found              = len(seen_failure_set)
    all_failures                = len(unique_failure_set)
    failure_percentage          = float(failures_found / all_failures) * 100

    return [coverage_percentage, failure_percentage]

# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',             type=str, default="/mnt/extradrive3/PhysicalCoverageData",  help="The location and name of the datafolder")
parser.add_argument('--number_of_test_suites', type=int, default=10,                                            help="The number of random test suites created")
parser.add_argument('--number_of_tests',       type=int, default=-1,                                            help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',          type=str, default="",                                            help="center_close/center_full")
parser.add_argument('--scenario',              type=str, default="",                                            help="beamng/highway")
parser.add_argument('--cores',                 type=int, default=4,                                             help="number of available cores")
parser.add_argument('--RRS',                   type=int, default=10,                                            help="Which RRS number you want to compute a correlation for")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
BK = BeamNGKinematics()
RRS = RRSConfig()

# Save the kinematics and RRS parameters
if args.scenario == "highway":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "beamng":
    new_steering_angle  = BK.steering_angle
    new_max_distance    = BK.max_velocity
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
if not (args.distribution == "center_close" or args.distribution == "center_full"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '{}/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.data_path, args.scenario, args.distribution, args.number_of_tests)
trace_file_names = glob.glob(base_path + "traces_*")
crash_file_names = glob.glob(base_path + "crash_*")
stall_file_names = glob.glob(base_path + "stall_*")

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

# Select a specific RRS
i = args.RRS - 1

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
    exit()

# Load the traces
global traces
traces = np.load(trace_file)

# Load the stall and crash file
global stalls
global crashes
stalls = np.load(stall_file, allow_pickle=True)
crashes = np.load(crash_file, allow_pickle=True)

# Load the feasibility file
feasible_traces = np.load(feasibility_file)

# Create the feasible set
global feasible_RRS_set
feasible_RRS_set = set()
for scene in feasible_traces:
    feasible_RRS_set.add(tuple(scene))

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

# Compute the denominator for the coverage
denominator = len(feasible_RRS_set)

total_tests = len(traces)

test_suit_sizes = [10, 50, 100, 500, 1000]

# Compute the correlation
for j, test_suite_size in enumerate(test_suit_sizes):
    print("Computing {} test suites of size {}".format(args.number_of_test_suites, test_suite_size))
    # Create random test suites
    coverage_percentages, unique_crash_count = random_selection(cores=args.cores,
                                                                test_suite_size=test_suite_size,
                                                                number_of_test_suites=args.number_of_test_suites)

    # Compute the correlation
    r = stats.pearsonr(coverage_percentages, unique_crash_count)
    r_value = round(r[0], 4)
    p_value = round(r[1], 4)
    print("R value: {} - P value: {}".format(r_value, p_value))

    # Plot the results
    plt.scatter(coverage_percentages, unique_crash_count, color="C{}".format(j), label="Size: {} - Correlation: {}".format(test_suite_size, r_value))
    print("---------------------------------------------")

plt.xlabel("Physical Coverage (%)")
plt.ylabel("Unique Failures (%)")
plt.legend()
plt.title("PhysCov - RRS {}: {}".format(args.RRS, args.scenario))
plt.xlim([-5,100])
plt.ylim([-5,100])
plt.show()
