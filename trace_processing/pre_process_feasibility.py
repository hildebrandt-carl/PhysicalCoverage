import os
import sys
import glob
import copy
import argparse
import multiprocessing

import numpy as np
import pandas as pd

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/trace_processing")])
sys.path.append(base_directory)

from tqdm import tqdm
from tabulate import tabulate
from general.environment_configurations import RSRConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics

from pre_process_functions import processFileFeasibility

def beamng_handler(beam, new_steering_angle, new_max_distance, new_accuracy):

    # Compute the new_total_lines
    new_total_lines = int(beam)

    # Compute all vectors
    feasible_vectors = set()
    tmp_vec = np.full(new_total_lines, new_max_distance)
    while tmp_vec[-1] >= new_accuracy:
        # Add the current vector
        feasible_vectors.add(tuple(tmp_vec))

        # Remove the value from the first position
        tmp_vec[0] -= new_accuracy

        # make sure it doesn't go below new_accuracy
        for i in range(len(tmp_vec) - 1):
            if tmp_vec[i] < new_accuracy:
                tmp_vec[i] = new_max_distance
                tmp_vec[i+1] -= new_accuracy

    # Get a list of all vectors
    feasible_vectors = np.array(list(feasible_vectors))
    total_feasible_vectors = np.shape(feasible_vectors)[0]

    # All vectors are feasible
    total_all_vectors = total_feasible_vectors

    per = round((total_feasible_vectors/total_all_vectors) * 100, 4)

    print("----------------------------------")
    print("-----Beam Number {} Complete------".format(new_total_lines))
    print("----------------------------------")
    print("Determining Percentage Feasible")
    print("Total feasible vectors: {}".format(total_feasible_vectors))
    print("Total possible vectors: {}".format(total_feasible_vectors))
    print("Percentage of feasible space {}% ".format(per))

    save_path = '../output/beamng/feasibility/processed/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_location = save_path + "FeasibleVectors_b{}.npy".format(new_total_lines)
    print("Saving to: {}".format(save_location))
    print("\n\n")
    np.save(save_location, feasible_vectors)
    
    return True

def highway_handler(file_name, new_steering_angle, new_max_distance, new_accuracy):

    # Compute the new_total_lines
    beams = file_name[file_name.rfind('s')+1:file_name.rfind('.')]
    new_total_lines = int(beams)

    # Process the data
    f = open(file_name, "r")    
    vehicle_count, crash, test_vectors = processFileFeasibility(f, new_steering_angle, new_max_distance, new_total_lines, new_accuracy)
    f.close()

    # Create the set of feasible vectors
    feasible_vectors = set()

    # Go through and add each of the vectors
    for i in tqdm(range(len(test_vectors))):
        vec = test_vectors[i]
        tmp_vec = copy.deepcopy(vec)

        # Loop through all smaller vectors
        while tmp_vec[-1] >= new_accuracy:

            # Add the current vector
            feasible_vectors.add(tuple(tmp_vec))

            # Remove the value from the first position
            tmp_vec[0] -= new_accuracy

            # make sure it doesn't go below 0
            for i in range(len(tmp_vec) - 1):
                if tmp_vec[i] < new_accuracy:
                    tmp_vec[i] = vec[i]
                    tmp_vec[i+1] -= new_accuracy

    feasible_vectors = np.array(list(feasible_vectors))
    total_feasible_vectors = np.shape(feasible_vectors)[0]

    # Compute all vectors
    all_vectors = set()
    tmp_vec = np.full(new_total_lines, new_max_distance)
    while tmp_vec[-1] >= new_accuracy:
        # Add the current vector
        all_vectors.add(tuple(tmp_vec))

        # Remove the value from the first position
        tmp_vec[0] -= new_accuracy

        # make sure it doesn't go below new_accuracy
        for i in range(len(tmp_vec) - 1):
            if tmp_vec[i] < new_accuracy:
                tmp_vec[i] = new_max_distance
                tmp_vec[i+1] -= new_accuracy

    # Get a list of all vectors
    all_vectors = np.array(list(all_vectors))
    total_all_vectors = np.shape(all_vectors)[0]

    per = round((total_feasible_vectors/total_all_vectors) * 100, 4)

    print("----------------------------------")
    print("-----Beam Number {} Complete------".format(new_total_lines))
    print("----------------------------------")
    print("File name: {}".format(file_name))
    print("Determining Percentage Feasible")
    print("Total feasible vectors: {}".format(total_feasible_vectors))
    print("Total possible vectors: {}".format(total_all_vectors))
    print("Percentage of feasible space {}% ".format(per))

    save_path = '../output/highway/feasibility/processed/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_location = save_path + "FeasibleVectors_b{}.npy".format(new_total_lines)
    print("Saving to: {}".format(save_location))
    print("\n\n")
    np.save(save_location, feasible_vectors)
    
    return True

parser = argparse.ArgumentParser()
parser.add_argument('--scenario',       type=str, default="",    help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,     help="number of available cores")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
NG = BeamNGKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_accuracy            = RSR.accuracy
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
print("Vector accuracy:\t" + str(new_accuracy))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

# These are used to hold the final values
table_tfv = []
table_tpv = []
table_percentage = []
table_beam = []

# Create the pool of processors
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Handle highway
if args.scenario == "highway":
    file_names = glob.glob("../../PhysicalCoverageData/highway/feasibility/raw/feasible_vectors*.txt")
    file_names.sort()

    print("Files found: {}".format(file_names))

    # Call our function total_test_suites times
    jobs = []
    for file_name in file_names:
        jobs.append(pool.apply_async(highway_handler, args=([file_name, new_steering_angle, new_max_distance, new_accuracy])))

# Handle beamng
if args.scenario == "beamng":
    # Call our function for each beam
    beams = [1,2,3,4,5,6,7,8,9]
    jobs = []
    for beam in beams:
        jobs.append(pool.apply_async(beamng_handler, args=([beam, new_steering_angle, new_max_distance, new_accuracy])))


# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

print("Complete")

pool.close()