import glob
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from count_unique_crashes import count_unique_crashes_from_file, crash_hasher

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics

def unique_vector_config(scenario, number_of_seconds):
    if scenario == "highway":
        hash_size = 4 * number_of_seconds
    elif scenario == "beamng":
        hash_size = 4 * number_of_seconds
    else:
        exit()
    return hash_size


def compute_coverage(trace_indices, scenario, feasible_vectors):

    # Count the number of unique vectors seen for the combined data
    total_crashes         = 0
    total_unique_crashes  = 0

    # Count the number of unique vectors seen for each of the different combinations of data
    unique_vectors_seen_set                 = set()
    unique_feasible_vector_seen_set         = set()
    unique_crashes_seen_set                 = set()

    # For each of the traces
    for index in trace_indices:
        global traces
        trace = traces[index]
        
        # See if there was a crash
        if np.isnan(trace).any():
            total_crashes += 1

            # Check if the crash is unique
            hash_size = unique_vector_config(scenario, number_of_seconds=1)
            crash_hash = crash_hasher(trace, hash_size)
            # Check we only get 1 answer
            assert(len(crash_hash) == 1)
            crash_hash = crash_hash[0]
            # Count and add it
            if crash_hash not in unique_crashes_seen_set:
                total_unique_crashes += 1
                unique_crashes_seen_set.add(crash_hash)

        # For each vector in the trace
        for vector in trace:
            # If this vector contains nan it means it crashed (and so we can ignore it, this traces crash was already counted)
            if np.isnan(vector).any():
                continue

            # If this vector contains inf it means that the trace was extended to match sizes with the maximum sized trace
            if np.isinf(vector).any():
                continue

            # If this vector contains any -1 it means it needed to be expanded to fit
            if np.all(vector==-1):
                continue
            
            # Check if it is unique
            unique_vectors_seen_set.add(tuple(vector))

            # Check if any of the vectors are infeasible
            if tuple(vector) not in feasible_vectors:
                continue

            unique_feasible_vector_seen_set.add(tuple(vector))


    # Used for the accumulative graph
    unique_vector_count                          = len(unique_vectors_seen_set)
    unique_feasible_vector_count                 = len(unique_feasible_vector_seen_set)

    # Return the data
    return [unique_vector_count, unique_feasible_vector_count, total_crashes, total_unique_crashes]


def compute_coverage_per_file(file_names, base_path, test_suite_sizes, test_suite_samples, scenario):

    for i in range(len(file_names)):

        # Get the file
        f = file_names[i]
        current_beam_count = i+1
        assert("b" + str(current_beam_count) in f)

        # Get the file names
        global traces
        traces = np.load(base_path + "traces" + f)

        # Turn the feasible vectors into a set
        fname = '../../PhysicalCoverageData/' + str(scenario) +'/feasibility/processed/FeasibleVectors_b' + str(current_beam_count) + ".npy"
        feasible_vectors = list(np.load(fname))
        feasible_vectors_set = set()
        for vector in feasible_vectors:
            feasible_vectors_set.add(tuple(vector))
        assert(len(feasible_vectors_set) == len(feasible_vectors))
            
        for suite_size in test_suite_sizes:

            # Randomly pick tests of the correct sample size
            indices = np.random.choice(len(traces), suite_size, replace=False) 

            # Compute the coverage for that test suit
            results = compute_coverage(indices, scenario, feasible_vectors_set)

            # Get the results
            unique_vector_count             = results[0]
            unique_feasible_vector_count    = results[1]
            total_crashes                   = results[2]
            total_unique_crashes            = results[3]

            print(total_crashes)


## WAS HERE ABOUT TO COMPUTE SHIT HERE

total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
for number_of_tests in tests_per_test_suite:
    for i in range(total_test_suites):
        jobs.append(pool.apply_async(random_selection, args=([number_of_tests])))

# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())


def determine_test_suite_sizes(total_tests):
    # We now need to sample tests of different sizes to create the plot
    percentage_of_all_tests = [1, 2.5, 5, 10, 25, 50]
    test_sizes = []

    for p in percentage_of_all_tests:

        # Compute what the size is
        t = (p/100.0) * total_tests
        test_sizes.append(int(np.round(t,0)))

    return test_sizes

def determine_total_crashes(file_names, scenario, total_samples, cores):
    # Compute how many unique crashes there are
    print("----------------------------------")
    print("---------Computing Crashes--------")
    print("----------------------------------")

    total_crashes = np.full(len(file_names), 0)
    unique_crashes = np.full(len(file_names), 0)

    for i in range(len(file_names)):
        # get the file
        f = file_names[i]

        # Count the number of crashes
        total, unique = count_unique_crashes_from_file(f, scenario, total_samples, cores)

        # Make sure we are working on the correct file
        assert(("b" + str(i+1)) in f)

        # Save the data
        total_crashes[i] = total
        unique_crashes[i] = unique

    return total_crashes, unique_crashes



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
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/processed/' + str(args.total_samples) + "/"
trace_file_names = glob.glob(base_path + "traces_" + args.scenario + load_name)
file_names = []
for f in trace_file_names:
    name = f.replace(base_path + "traces", "")
    file_names.append(name)

# Sort the names
file_names.sort()
print("Files: " + str(file_names))
print("Loading Complete")

# Run the program
total_crashes, unique_crashes = determine_total_crashes(file_names, args.scenario, args.total_samples, args.cores)
test_suite_sizes = determine_test_suite_sizes(args.total_samples)


# Compute the coverage
compute_coverage_per_file(file_names, base_path, test_suite_sizes, 10, args.scenario)























