import random 
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm


def random_selection(number_of_tests):
    global traces

    # Generate the test indices
    local_state = np.random.RandomState()
    indices = local_state.choice(traces.shape[0], number_of_tests, replace=False)
    # Init variables
    coverage = 0
    number_of_crashes = 0
    unique_vectors_seen = []
    # For the X test cases
    for random_i in indices:
        # Get the filename
        vectors = traces[random_i]
        # Look for a crash
        if np.isnan(vectors).any():
            number_of_crashes += 1
        # Check to see if any of the vectors are new
        for v in vectors:
            if not np.isnan(v).any():
                unique = isUnique(v, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(v)
        
        # Compute coverage
        coverage = (len(unique_vectors_seen) / float(total_possible_observations)) * 100
        
    # Return the data
    return [number_of_tests, coverage, number_of_crashes]

def isUnique(vector, unique_vectors_seen):
    # Return false if the vector contains Nan
    if np.isnan(vector).any():
        return False
    # Assume True
    unique = True
    for v2 in unique_vectors_seen:
        # If we have seen this vector break out of this loop
        if np.array_equal(vector, v2):
            unique = False
            break
    return unique


parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,    help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=5,     help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',   type=int, default=30,    help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,     help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--greedy_sample',  type=int, default=50,    help="The unumber of samples considered by the greedy search")
parser.add_argument('--scenario',       type=str, default="",    help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,     help="The number of CPU cores available")
args = parser.parse_args()

new_steering_angle  = args.steering_angle
new_total_lines     = args.beam_count
new_max_distance    = args.max_distance
new_accuracy        = args.accuracy
greedy_sample_size  = args.greedy_sample

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Total beams:\t\t" + str(new_total_lines))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))
print("Greedy Sample Size:\t" + str(greedy_sample_size))

# Compute total possible values using the above
unique_observations_per_cell = (new_max_distance / float(new_accuracy)) + 1.0
total_possible_observations = pow(unique_observations_per_cell, new_total_lines)

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = ""
load_name += "_s" + str(new_steering_angle) 
load_name += "_b" + str(new_total_lines) 
load_name += "_d" + str(new_max_distance) 
load_name += "_a" + str(new_accuracy)
load_name += "_t" + str(args.total_samples)
load_name += ".npy"

# Get the file names
base_path = None
if args.scenario == "beamng":
    base_path = '../../PhysicalCoverageData/beamng/numpy_data/'
elif args.scenario == "highway":
    base_path = '../../PhysicalCoverageData/highway/numpy_data/' + str(args.total_samples) + "/"
else:
    exit()

print("Loading: " + load_name)
traces = np.load(base_path + "traces_" + args.scenario + load_name)

print("----------------------------------")
print("--------Crashes vs Coverage-------")
print("----------------------------------")

total_test_suites = 10000
tests_per_test_suite = [50, 100, 250, 500, 1000]

# Create a pool with x processes
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

print("Done!")

# Sort the results based on number of test suites
final_coverage          = np.zeros((len(tests_per_test_suite), total_test_suites))
final_number_crashes    = np.zeros((len(tests_per_test_suite), total_test_suites))
position_counter        = np.zeros(len(tests_per_test_suite), dtype=int)
for r in results:
    # Get the row
    ind = tests_per_test_suite.index(r[0])
    # Save in the correct position
    final_coverage[ind, position_counter[ind]] = r[1]
    final_number_crashes[ind, position_counter[ind]] = r[2]
    position_counter[ind] += 1

# Save the results
save_name = "../results/rq3_"
print("Saving to: " + str(save_name))

np.save(save_name + "coverage_" + str(args.scenario), final_coverage)
np.save(save_name + "crashes_" + str(args.scenario), final_number_crashes)

final_coverage          = np.load(save_name + "coverage_" + str(args.scenario) + ".npy")
final_number_crashes    = np.load(save_name + "crashes_" + str(args.scenario) + ".npy")

# Run the plotting code
exec(compile(open("rq3_plot.py", "rb").read(), "rq3_plot.py", 'exec'))