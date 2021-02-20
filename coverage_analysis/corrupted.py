import glob
import random 
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,   help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=4,    help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',   type=int, default=20,   help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,    help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
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
print("----------Locating Files----------")
print("----------------------------------")

# all_files = glob.glob("../beamng_output/*.txt")
all_files = glob.glob("../../PhysicalCoverageData/highway/*/*.txt")
total_files = len(all_files)
print("Total files found: " + str(total_files))

# Select all or part of the files
file_names = all_files
if args.total_samples != -1:
    file_names = random.sample(all_files, args.total_samples)

total_files = len(file_names)
print("Total files selected for processing: " + str(total_files))

print("----------------------------------")
print("----Locating Uncorrupt Example----")
print("----------------------------------")

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

print("----------------------------------")
print("---Checking for corrupted files---")
print("----------------------------------")

# Check if any of the files are corrupted
for i in tqdm(range(len(file_names))):
    # Get the filename
    file_name = file_names[i]

    # Open the file and count vectors
    f = open(file_name, "r")    
    vector_count, crash = countVectorsInFile(f)
    f.close()

    # Print the number of vectors
    if (vector_count < vec_per_file) and not crash:
        print("corrupted file: " + str(file_name))