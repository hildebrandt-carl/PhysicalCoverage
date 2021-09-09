
import os
import re
import sys
import glob
import math
import random 
import argparse

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/trace_processing")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from general.environment_configurations import RSRConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics

from general.crash_oracle import CrashOracle

from pre_process_functions import processFile
from pre_process_functions import countVectorsInFile


parser = argparse.ArgumentParser()
parser.add_argument('--beam_count',     type=int, default=4,    help="The number of beams used to vectorize the reachable set")
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
NG = BeamNGKinematics()
RSR = RSRConfig(beam_count = args.beam_count)
CO = CrashOracle(scenario=args.scenario)

# Save the kinematics and RSR parameters
new_total_lines         = RSR.beam_count
new_accuracy            = RSR.accuracy

if args.scenario == "highway_random":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "beamng_random":
    new_steering_angle  = NG.steering_angle
    new_max_distance    = NG.max_velocity
else:
    print("ERROR: Unknown scenario")
    exit()

# Get the total number of possible crashes per test
max_crashes_per_test = CO.max_possible_crashes
crash_base           = CO.base

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Total beams:\t\t" + str(new_total_lines))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

# Compute total possible values using the above
unique_observations_per_cell = (new_max_distance / float(new_accuracy))
total_possible_observations = pow(unique_observations_per_cell, new_total_lines)

print("----------------------------------")
print("----------Locating Files----------")
print("----------------------------------")

all_files = None
if args.scenario == "beamng_random":
    all_files = glob.glob("../../PhysicalCoverageData/beamng/random_tests/physical_coverage/raw/*/*.txt")
elif args.scenario == "highway_random":
    all_files = glob.glob("../../PhysicalCoverageData/highway/random_tests/physical_coverage/raw/*/*.txt")
elif args.scenario == "highway_generated":
    all_files = glob.glob("../../PhysicalCoverageData/highway/generated_tests/tests_single/raw/{}/{}_external_vehicles/*.txt".format(args.total_samples, new_total_lines))
else:
    exit()

total_files = len(all_files)
print("Total files found: " + str(total_files))

# Select all of the files
file_names = all_files

# If no files are found exit
if len(file_names) <= 0:
    print("No files found")
    exit()

# If you don't want all files, select a random portion of the files
if (args.scenario == "highway_random") or (args.scenario == "beamng_random"): 
    if args.scenario == "highway_random":
        folders = glob.glob("../../PhysicalCoverageData/highway/random_tests/physical_coverage/raw/*")
    elif args.scenario == "beamng_random":
        folders = glob.glob("../../PhysicalCoverageData/beamng/random_tests/physical_coverage/raw/*")

    files_per_folder = int(math.ceil(args.total_samples / len(folders)))

    # Need to set the seed or else you will be picking different tests for each different beam number  
    random.seed(10)
    print("There are {} categories, thus we need to select {} from each".format(len(folders), files_per_folder))
    print("")
    file_names = []
    for f in folders:
        print("Selecting {} random files from - {}".format(files_per_folder, f))
        all_files = glob.glob(f + "/*.txt")
        names = random.sample(all_files, files_per_folder)
        file_names.append(names)
if args.scenario == "highway_generated": 
    # You want to select all files here so do nothing
    pass

# Flatten the list
if len(np.shape(file_names)) > 1:
    file_names_flat = []
    for subl in file_names:
        for item in subl:
            file_names_flat.append(item)
    file_names = file_names_flat

# Get the file size
total_files = len(file_names)
print("Total files selected for processing: " + str(total_files))

print("----------------------------------")
print("--------Memory Requirements-------")
print("----------------------------------")
print(total_files)
print("Computing size of memory required")
# Open the first 1000 files to get an estimate of how many vectors in each file
vectors_per_file = np.zeros(min(total_files, 1000), dtype=int)
for i in tqdm(range(min(total_files, 1000))):
    # Get the filename
    file_name = file_names[i]

    # Process the file
    f = open(file_name, "r")    
    vector_count, crash = countVectorsInFile(f)
    f.close()

    # See how many vectors there are
    vectors_per_file[i] = vector_count

# Compute the average number of vectors per file
vec_per_file = np.max(vectors_per_file)

# Compute the estimated size of the numpy array
print("Computed vectors per file as: {}".format(vec_per_file))
dummy_array = np.zeros((1, vec_per_file, new_total_lines), dtype='float64')
memory_required = dummy_array.nbytes * total_files

# Convert to GB
memory_required = round(memory_required / 1024 / 1024 / 1024, 2)
print("Memory required: " + str(memory_required) + "GB")

print("----------------------------------")
print("---------Processing files---------")
print("----------------------------------")

# Create the numpy array 
reach_vectors = np.zeros((total_files, vec_per_file, new_total_lines), dtype='float64')
vehicles_per_trace = np.zeros(total_files, dtype=int)
time_per_trace = np.zeros(total_files, dtype='float64')
crash_hashes = np.zeros((total_files, max_crashes_per_test), dtype='float64')

# For each file
file_count = 0
for i in tqdm(range(total_files)):
    # Get the filename
    file_name = file_names[i]

    # Process the file
    f = open(file_name, "r")    
    vehicle_count, crash_count, test_vectors, simulation_time, incident_hashes = processFile(f, vec_per_file, new_total_lines, new_steering_angle, new_max_distance, new_total_lines, new_accuracy, max_crashes_per_test, crash_base, True)
    f.close()

    reach_vectors[i]        = test_vectors
    vehicles_per_trace[i]   = vehicle_count
    time_per_trace[i]       = simulation_time
    crash_hashes[i]         = incident_hashes

save_name = args.scenario
save_name += "_s" + str(new_steering_angle) 
save_name += "_b" + str(new_total_lines) 
save_name += "_d" + str(new_max_distance) 
save_name += "_a" + str(new_accuracy)
save_name += "_t" + str(total_files)
save_name += ".npy"
   
save_path = ""
if args.scenario == "beamng_random":
    save_path = "../output/beamng/random_tests/physical_coverage/processed/{}".format(args.total_samples)
elif args.scenario == "highway_random":
    save_path = "../output/highway/random_tests/physical_coverage/processed/{}".format(args.total_samples)
elif args.scenario == "highway_generated":
    save_path = "../output/highway/generated_tests/processed/{}/".format(args.total_samples)
else:
    print("Error")
    exit()

# Create the output directory if it doesn't exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Saving data")
np.save(save_path + '/traces_{}'.format(save_name), reach_vectors)
np.save(save_path + '/vehicles_{}'.format(save_name), vehicles_per_trace)
np.save(save_path + '/time_{}'.format(save_name), time_per_trace)
np.save(save_path + '/crash_hash_{}'.format(save_name), crash_hashes)
