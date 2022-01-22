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
base_directory = str(path[:path.rfind("/preprocessing")])
sys.path.append(base_directory)

from tqdm import tqdm
from tabulate import tabulate
from general.environment_configurations import RSRConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics

from preprocess_functions import processFileFeasibility
from preprocess_functions import getFeasibleVectors

def beamng_handler(new_steering_angle, new_max_distance, RRS_number):

    # Get the new_total_lines
    new_total_lines = int(RRS_number)

    # All vectors are possible in beamng
    test_vectors = [np.full(new_total_lines, new_max_distance)]
    
    # Create the set of feasible vectors
    feasible_vectors = set()

    # Get all the vectors and all feasible vectors
    all_vectors, feasible_vectors = getFeasibleVectors(test_vectors, new_total_lines, "beamng")

    total_feasible_vectors = np.shape(feasible_vectors)[0]
    total_all_vectors = np.shape(all_vectors)[0]
    per = round((total_feasible_vectors/total_all_vectors) * 100, 4)

    print("----------------------------------")
    print("-----Beam Number {} Complete------".format(new_total_lines))
    print("----------------------------------")
    print("Determining Percentage Feasible")
    print("Total feasible vectors: {}".format(total_feasible_vectors))
    print("Total possible vectors: {}".format(total_all_vectors))
    print("Percentage of feasible space {}% ".format(per))

    save_path = '../output/beamng/feasibility/processed/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_location = save_path + "FeasibleVectors_b{}.npy".format(new_total_lines)
    print("Saving to: {}".format(save_location))
    print("\n\n")
    np.save(save_location, feasible_vectors)
    
    return True

def highway_handler(file_name, new_steering_angle, new_max_distance, RRS_number):

    # Get the new_total_lines
    new_total_lines = int(RRS_number)

    # Process the data
    f = open(file_name, "r")  
    vehicle_count, crash, test_vectors = processFileFeasibility(f, new_steering_angle, new_max_distance, new_total_lines, "highway")
    f.close()
    
    # Create the set of feasible vectors
    feasible_vectors = set()

    # Get all the vectors and all feasible vectors
    all_vectors, feasible_vectors = getFeasibleVectors(test_vectors, new_total_lines, "highway")

    total_feasible_vectors = np.shape(feasible_vectors)[0]
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
    feasibility_file = glob.glob("../../PhysicalCoverageData/highway/feasibility/raw/feasible_vectors.txt")
    assert(len(feasibility_file) == 1)
    print("file found: {}".format(feasibility_file[0]))

    # Call our function for each beam
    jobs = []
    for RRS_number in range(1,11):
        jobs.append(pool.apply_async(highway_handler, args=([feasibility_file[0], new_steering_angle, new_max_distance, RRS_number])))

# Handle beamng
if args.scenario == "beamng":
    # Call our function for each beam
    jobs = []
    for RRS_number in range(1,11):
        jobs.append(pool.apply_async(beamng_handler, args=([new_steering_angle, new_max_distance, RRS_number])))


# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

print("Complete")

pool.close()