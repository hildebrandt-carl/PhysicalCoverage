
import os
import re
import glob
import math
import random 
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics


def string_to_vector(vector_string):
    vector_str = vector_string[vector_string.find(": ")+3:-2]
    vector = np.fromstring(vector_str, dtype=float, sep=', ')
    return vector

# This function was written as if the data had a steering angle of 60, total lines of 30, max_distance of 30
def vector_conversion(vector, steering_angle, max_distance, total_lines):
    original_steering_angle = 60
    original_total_lines = 30
    original_max_distance = 30

    # Fix the vector to have to correct max_distance
    vector = np.clip(vector, 0, max_distance)

    # Get how many degrees between each line
    line_space = (original_steering_angle * 2) / float(original_total_lines - 1)

    # Get the starting lines angle
    left_index = int(len(vector) / 2)
    right_index = int(len(vector) / 2)
    current_steering_angle = 0
    if (original_total_lines % 2) == 0: 
        current_steering_angle = line_space / 2
        left_index -= 1

    # This is an overapproximation
    while current_steering_angle < steering_angle:
        left_index -= 1
        right_index += 1
        current_steering_angle += line_space

    # Get the corrected steering angle
    steering_angle_corrected_vector = vector[left_index:right_index+1]

    # Select the correct number of lines
    if len(steering_angle_corrected_vector) < total_lines:
        pass
        # print("Requested moer lines than we have, extrapolating")
    idx = np.round(np.linspace(0, len(steering_angle_corrected_vector) - 1, total_lines)).astype(int)
    final_vector = steering_angle_corrected_vector[idx]

    return final_vector

def getStep(vector, accuracy):
    return np.round(np.array(vector, dtype=float) / accuracy) * accuracy

def countVectorsInFile(f):
    vector_count = 0
    crash = False
    for line in f: 
        if "Vector: " in line:
            vector_count += 1
        if "Crash: True" in line:
            crash = True
    return vector_count, crash

def processFile(f, total_vectors, vector_size):
    test_vectors    = np.full((total_vectors, vector_size), np.inf, dtype='float64')
    crash           = False
    vehicle_count   = -1
    current_vector  = 0
    for line in f: 
        # Get the number of external vehicles
        if "External Vehicles: " in line:
            vehicle_count = int(line[line.find(": ")+2:])
        # Get each of the vectors
        if "Vector: " in line:
            vector_str = line[line.find(": ")+3:-2]
            vector = np.fromstring(vector_str, dtype=float, sep=', ')
            vector = vector_conversion(vector, new_steering_angle, new_max_distance, new_total_lines)
            vector = getStep(vector, new_accuracy)
            test_vectors[current_vector] = vector
            current_vector += 1
        # Look for crashes
        if "Crash: True" in line:
            crash = True
            # File the rest of the test vector up with np.nan
            while current_vector < test_vectors.shape[0]:
                test_vectors[current_vector] = np.full(test_vectors.shape[1], np.nan, dtype='float64')
                current_vector += 1

    return vehicle_count, crash, test_vectors

parser = argparse.ArgumentParser()

parser.add_argument('--beam_count',     type=int, default=4,    help="The number of beams used to vectorize the reachable set")
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig(beam_count = args.beam_count)

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_total_lines     = RSR.beam_count
new_accuracy        = RSR.accuracy

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
print("----------Locating Files----------")
print("----------------------------------")

all_files = None
if args.scenario == "beamng":
    all_files = glob.glob("../../PhysicalCoverageData/beamng/processed/*.txt")
elif args.scenario == "highway":
    all_files = glob.glob("../../PhysicalCoverageData/highway/raw/*/*.txt")
elif args.scenario == "highway_unseen":
    all_files = glob.glob("../../PhysicalCoverageData/highway/unseen/{}/results/{}_beams/*.txt".format(args.total_samples, new_total_lines))
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
if args.scenario != "highway_unseen":
    if args.total_samples != -1:

        # Only sample this way if using highway
        if args.scenario == "highway":
            folders = glob.glob("../../PhysicalCoverageData/highway/raw/*")
            files_per_folder = int(math.ceil(args.total_samples / len(folders)))
            print("There are {} categories, thus we need to select {} from each".format(len(folders), files_per_folder))
            print("")
            file_names = []
            for f in folders:
                print("Selecting {} random files from - {}".format(files_per_folder, f))
                all_files = glob.glob(f + "/*.txt")
                names = random.sample(all_files, files_per_folder)
                file_names.append(names)

        # Do this for the other scenarios
        else:
            print("Selecting {} random files".format(args.total_samples))
            file_names = random.sample(all_files, args.total_samples)


# Flatten the list
if len(np.shape(file_names)) > 1:
    file_names_flat = []
    for subl in file_names:
        for item in subl:
            file_names_flat.append(item)
    file_names = file_names_flat

# Make sure your list is the exact right size
print("")
if args.total_samples >= 1:
    if len(file_names) > args.total_samples:
        print("Currently there are {} files, cropping to {}".format(len(file_names), args.total_samples))
        file_names = file_names[0:args.total_samples]

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

# For each file
file_count = 0
for i in tqdm(range(total_files)):
    # Get the filename
    file_name = file_names[i]

    # Process the file
    f = open(file_name, "r")    
    vehicle_count, crash, test_vectors = processFile(f, vec_per_file, new_total_lines)
    f.close()

    reach_vectors[i] = test_vectors
    vehicles_per_trace[i] = vehicle_count

save_name = args.scenario
save_name += "_s" + str(new_steering_angle) 
save_name += "_b" + str(new_total_lines) 
save_name += "_d" + str(new_max_distance) 
save_name += "_a" + str(new_accuracy)
save_name += "_t" + str(total_files)
save_name += ".npy"

if args.scenario == "highway_unseen":
    total_files = args.total_samples
    
# Create the output directory if it doesn't exists
if not os.path.exists('output/{}'.format(total_files)):
    os.makedirs('output/{}'.format(total_files))

print("Saving data")
np.save("output/{}/traces_{}".format(total_files, save_name), reach_vectors)
np.save("output/{}/vehicles_{}".format(total_files, save_name), vehicles_per_trace)
