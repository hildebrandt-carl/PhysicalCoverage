import copy
import random 
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from rq3_configuration import compute_crash_hash, unique_vector_config


def crash_hasher(trace_number, hash_size):
    global traces

    # Determine if there is a crash
    trace = traces[trace_number]

    # Used to hold the last vectors before a crash
    last_seen_vectors = np.zeros((hash_size, trace[0].shape[0]))

    # Create the hash
    hash_value = np.nan

    # If there is no crash return none
    if not np.isnan(trace).any():
        return [np.nan]
    # Else return the hash
    else:
        hash_value = compute_crash_hash(trace, hash_size)

    return [hash_value]


parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,    help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=5,     help="The number of beams used to vectorized the reachable set")
parser.add_argument('--max_distance',   type=int, default=30,    help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,     help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",    help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,     help="The number of CPU cores available")
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
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/numpy_data/' + str(args.total_samples) + "/"

print("Loading: " + load_name)
traces = np.load(base_path + "traces_" + args.scenario + load_name)

print("----------------------------------")
print("-----Computing Unique Crashes-----")
print("----------------------------------")

hash_size = unique_vector_config(args.scenario, number_of_seconds=1)

# Create a pool with x processes
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

jobs = []
# For all the different test suite sizes
for trace_i in range(len(traces)):
    jobs.append(pool.apply_async(crash_hasher, args=(trace_i, hash_size)))
    
# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

print("Done computing all the hash functions")

# Get the crash data
print("Computing unique crash values")
print("")
results = np.array(results).reshape(-1)
total_crashes = results[np.logical_not(np.isnan(results))]
print("Total crash count: " + str(total_crashes.shape[0]))

# Count the unique elements in an array
unique = np.unique(total_crashes)
print("Total unique crashes count: " + str(unique.shape[0]))