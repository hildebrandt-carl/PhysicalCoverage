
import os
import glob
import copy
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from tabulate import tabulate
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

def processFile(f):
    test_vectors    = []
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
            test_vectors.append(np.array(vector))
            current_vector += 1
        # Look for crashes
        if "Crash: True" in line:
            crash = True

    test_vectors = np.array(test_vectors)

    return vehicle_count, crash, test_vectors


parser = argparse.ArgumentParser()
parser.add_argument('--scenario',               type=str, default="",    help="beamng/highway")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy

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

file_names = glob.glob("../../PhysicalCoverageData/highway/feasibility/raw/feasible_vectors*.txt")
file_names.sort()

print("Files found: {}".format(file_names))

print("Done")

for file_name in file_names:

    print("----------------------------------")
    print("---------Processing file----------")
    print("----------------------------------")
    print("File name: {}".format(file_name))
    print("")

    # Compute the new_total_lines
    beams = file_name[file_name.rfind('s')+1:file_name.rfind('.')]
    new_total_lines = int(beams)

    # if new_total_lines == 10:
    #     continue

    # Process the data
    f = open(file_name, "r")    
    vehicle_count, crash, test_vectors = processFile(f)
    f.close()

    print("Computing Feasible Vectors")

    # Create the set of feasible vectors
    feasible_vectors = set()

    # Go through and add each of the vectors
    for i in tqdm(range(len(test_vectors))):
        vec = test_vectors[i]
        tmp_vec = copy.deepcopy(vec)

        # Loop through all smaller vectors
        while tmp_vec[-1] >= 0:

            # Add the current vector
            feasible_vectors.add(tuple(tmp_vec))

            # Remove the value from the first position
            tmp_vec[0] -= new_accuracy

            # make sure it doesn't go below 0
            for i in range(len(tmp_vec) - 1):
                if tmp_vec[i] < 0:
                    tmp_vec[i] = vec[i]
                    tmp_vec[i+1] -= new_accuracy


    print("Determining Percentage Feasible")

    feasible_vectors = np.array(list(feasible_vectors))
    total_feasible_vectors = np.shape(feasible_vectors)[0]
    print("Total feasible vectors: {}".format(total_feasible_vectors))

    # Compute all vectors
    all_vectors = set()
    tmp_vec = np.full(new_total_lines, new_max_distance)
    while tmp_vec[-1] >= 0:

        # Add the current vector
        all_vectors.add(tuple(tmp_vec))

        # Remove the value from the first position
        tmp_vec[0] -= new_accuracy

        # make sure it doesn't go below 0
        for i in range(len(tmp_vec) - 1):
            if tmp_vec[i] < 0:
                tmp_vec[i] = new_max_distance
                tmp_vec[i+1] -= new_accuracy

    # Get a list of all vectors
    all_vectors = np.array(list(all_vectors))
    total_all_vectors = np.shape(all_vectors)[0]
    print("Total possible vectors: {}".format(total_all_vectors))
    per = round((total_feasible_vectors/total_all_vectors) * 100, 4)
    print("Percentage of feasible space {}% ".format(per))

    # Save this to the final table
    table_tfv.append(total_feasible_vectors)
    table_tpv.append(total_all_vectors)
    table_percentage.append(per)
    table_beam.append(new_total_lines)

    if not os.path.exists('output/processed'):
        os.makedirs('output/processed')

    save_location = "output/processed/FeasibleVectors_b{}.npy".format(new_total_lines)
    print("Saving to: {}".format(save_location))
    print("\n\n")
    np.save(save_location, feasible_vectors)


print("----------------------------------")
print("----------Final Results-----------")
print("----------------------------------")
print("\n\n")

# Create the tabulated data
headings = ['Beam Count', 'Total Possible Vectors', 'Total Feasible Vectors', 'Percentage of Feasible Vectors']
rows = []
for i in range(len(table_beam)):
    beam = table_beam[i]
    tfv = table_tfv[i]
    tpv = table_tpv[i]
    per = table_percentage[i]

    r = [beam, tpv, tfv, per]
    rows.append(r)
    
print(tabulate(rows, headers=headings))