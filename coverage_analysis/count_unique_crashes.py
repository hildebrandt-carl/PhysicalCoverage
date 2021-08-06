import copy
import random 
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from test_selection_config import compute_crash_hash, unique_vector_config

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics

def count_unique_crashes_from_file(file_name, scenario, total_samples, cores):

    # Get the file names
    base_path = '../../PhysicalCoverageData/' + str(scenario) +'/processed/' + str(total_samples) + "/"
    traces = np.load(base_path + "traces" + file_name)

    # Count the crashes
    total_crashes, unique_crashes = count_unique_crashes(traces, scenario, cores)
    return total_crashes, unique_crashes

def count_unique_crashes(traces, scenario, cores):

    hash_size = unique_vector_config(scenario, number_of_seconds=1)

    # Create a pool with x processes
    total_processors = int(cores)
    pool =  multiprocessing.Pool(processes=total_processors)

    jobs = []
    # For all the different test suite sizes
    for trace in traces:
        jobs.append(pool.apply_async(crash_hasher, args=(trace, hash_size)))
        
    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Get the crash data
    results = np.array(results).reshape(-1)
    total_crashes = results[np.logical_not(np.isnan(results))]

    unique = np.unique(total_crashes)
    return total_crashes.shape[0], unique.shape[0]


def crash_hasher(trace, hash_size):

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
