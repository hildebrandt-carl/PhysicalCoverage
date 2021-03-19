import glob
import random 
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import argparse
import copy
import multiprocessing
import time
import itertools

def crashing_vs_noncrashing(print_pos):
    global traces 

    only_crashing_vectors = 0

    # Compute all unique values for tests with crashes
    unique_vectors = []
    crashing_test_count = 0
    for i in tqdm(range(traces.shape[0]), position=print_pos):
        trace = traces[i]
        # Only compute traces without a crash
        if np.isnan(trace).any():
            crashing_test_count += 1
            # For each vector in the trace
            for vector in trace:
                # We want to consider vectors before Nan
                if not np.isnan(vector).any():
                    unique = isUnique(vector, unique_vectors)
                    if unique:
                        unique_vectors.append(vector)

    # Save how many vectros were seen when crashing
    only_crashing_vectors = len(unique_vectors)

    # Compute all unique values for tests without crashes
    non_crashing_test_count = 0
    for i in tqdm(range(traces.shape[0]), position=print_pos):
        trace = traces[i]
        # Only compute traces without a crash
        if not np.isnan(trace).any():
            non_crashing_test_count += 1
            # For each vector in the trace
            for vector in trace:
                unique = isUnique(vector, unique_vectors)
                if unique:
                    unique_vectors.append(vector)

    print("\n\n\n\n\n\n\n--------------------------------------------------------------")
    print("Crashing tests: " + str(crashing_test_count) + "/" + str(traces.shape[0]))
    print("Crashing unique vectors: " + str(only_crashing_vectors) + "/" + str(total_possible_observations))
    print("Non-crashing tests: " + str(non_crashing_test_count) + "/" + str(traces.shape[0]))
    print("Non-crashing unique vectors (including non crashing): " + str(len(unique_vectors)) + "/" + str(total_possible_observations))
    print("--------------------------------------------------------------\n\n\n\n\n\n\n")

    return None


def noncrashing_vs_crashing(print_pos):
    global traces 

    only_noncrashing_vectors = 0

    # Compute all unique values for tests without crashes
    unique_vectors = []
    non_crashing_test_count = 0
    for i in tqdm(range(traces.shape[0]), position=print_pos):
        trace = traces[i]
        # Only compute traces without a crash
        if not np.isnan(trace).any():
            non_crashing_test_count += 1
            # For each vector in the trace
            for vector in trace:
                unique = isUnique(vector, unique_vectors)
                if unique:
                    unique_vectors.append(vector)

    only_noncrashing_vectors = len(unique_vectors)

    # Compute all unique values for tests with crashes
    crashing_test_count = 0
    for i in tqdm(range(traces.shape[0]), position=print_pos):
        trace = traces[i]
        # Only compute traces without a crash
        if np.isnan(trace).any():
            crashing_test_count += 1
            # For each vector in the trace
            for vector in trace:
                # We want to consider vectors before Nan
                if not np.isnan(vector).any():
                    unique = isUnique(vector, unique_vectors)
                    if unique:
                        unique_vectors.append(vector)

    print("\n\n\n\n\n\n\n--------------------------------------------------------------")
    print("Non crashing tests: " + str(non_crashing_test_count) + "/" + str(traces.shape[0]))
    print("Non crashing unique vectors: " + str(only_noncrashing_vectors) + "/" + str(total_possible_observations))
    print("Crashing tests: " + str(crashing_test_count) + "/" + str(traces.shape[0]))
    print("Crashing unique vectors (including non crashing): " + str(len(unique_vectors)) + "/" + str(total_possible_observations))
    print("--------------------------------------------------------------\n\n\n\n\n\n\n")

    return None

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
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
args = parser.parse_args()

new_steering_angle  = args.steering_angle
new_total_lines     = args.beam_count
new_max_distance    = args.max_distance
new_accuracy        = args.accuracy

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

total_processors = 32
pool =  multiprocessing.Pool(total_processors)

# Call our functions in two different threads
result_object = []
result_object.append(pool.apply_async(crashing_vs_noncrashing, args=([1])))
result_object.append(pool.apply_async(noncrashing_vs_crashing, args=([2])))

# Get the results
results = [r.get() for r in result_object]
results = np.array(results)

# Close the pool
pool.close()
