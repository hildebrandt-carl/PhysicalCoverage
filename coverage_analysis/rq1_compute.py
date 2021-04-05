import time
import random 
import argparse
import multiprocessing

import numpy as np


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom')

def myround(x, base=5):
    return round(base * round(x/base), 5)

def generate_random_test_suite_coverage(process_number):
    global traces 

    print("Starting: " + str(process_number))
    # Randomly select how many tests to include
    number_traces = random.randint(1, min(traces.shape[0], 10000))
    # Randomly select indices to include
    selected_indices = random.sample(set(np.arange(traces.shape[0])), number_traces)
    # Init variables
    coverage = 0
    crashes = 0
    unique_vectors_seen = []
    # Compute the coverage for this test set
    for i in selected_indices:
        # Get the trace
        trace = traces[i]
        # Get all unique vectors from this trace
        for vector in trace:
            if not np.isnan(vector).any():
                unique = isUnique(vector, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(vector)
        # Check if there was a crash
        if np.isnan(trace).any():
            crashes += 1
    # Compute the coverage
    coverage = (len(unique_vectors_seen) / float(total_possible_observations)) * 100
    # Return the data
    print("Finished: " + str(process_number))
    return [coverage, crashes, number_traces]

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
parser.add_argument('--steering_angle',             type=int, default=30,    help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',                 type=int, default=5,     help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',               type=int, default=30,    help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',                   type=int, default=5,     help="What each vector is rounded to")
parser.add_argument('--total_samples',              type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',                   type=str, default="",    help="beamng/highway")
parser.add_argument('--total_random_test_suites',   type=int, default=1000,  help="Total random test suites to be generated")
parser.add_argument('--cores',                      type=int, default=4,     help="number of available cores")
args = parser.parse_args()

new_steering_angle  = args.steering_angle
new_total_lines     = args.beam_count
new_max_distance    = args.max_distance
new_accuracy        = args.accuracy
total_test_suites   = args.total_random_test_suites

min_tests_per_group = 500
interval_size = 1

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Total beams:\t\t" + str(new_total_lines))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

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

# Create a pool with x processes
total_processors = int(args.cores)
pool =  multiprocessing.Pool(total_processors)

# Call our function total_test_suites times
result_object = []
for i in range(total_test_suites):
    result_object.append(pool.apply_async(generate_random_test_suite_coverage, args=([i])))

# Get the results
results = [r.get() for r in result_object]
results = np.array(results)

# Close the pool
pool.close()

# Save the results
np.save("rq1" + str(args.scenario), results)
results = np.load("r11" + str(args.scenario))

# Run the plotting code
exec(compile(open("rq1_plot.py", "rb").read(), "rq1_plot.py", 'exec'))