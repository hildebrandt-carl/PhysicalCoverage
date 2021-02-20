import glob
import random 
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import argparse


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
        
    # print("Overapproximated the steering angle by: " + str(current_steering_angle - steering_angle))

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

def processFile(f):
    test_vectors = []
    crash = False
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
            test_vectors.append(vector)
        # Look for crashes
        if "Crash: True" in line:
            crash = True

    return vehicle_count, crash, test_vectors

def isUnique(vector, unique_vectors_seen):
    unique = True
    for v2 in unique_vectors_seen:
        # If we have seen this vector break out of this loop
        if np.array_equal(vector, v2):
            unique = False
            break
    return unique


# # Test code to see everything works
# a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# b = vector_conversion(a, 30, 30, 5)
# c = getStep(b, 5)
# print(a)
# print(b)
# print(c)

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

all_files = glob.glob("../beamng_output/*.txt")
# all_files = glob.glob("../data/*/*.txt")

# Select all or part of the files
file_names = all_files
if args.total_samples != -1:
    file_names = random.sample(all_files, args.total_samples)

# Sort the file names based on the total number of vehicles
# file_names = sorted(file_names, key = lambda x: int(re.split(r'\-|/', x)[2]))


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
    if (vector_count < 400) and not crash:
        print("corrupted file: " + str(file_name))


print("----------------------------------")
print("---------Processing files---------")
print("----------------------------------")

unique_vectors_seen = []
total_observations = 0
total_traces = 0
total_crashes = 0

accumulative_graph = []
acuumulative_graph_vehicle_count = []

# For each file
file_count = 0
for i in tqdm(range(len(file_names))):
    # Get the filename
    file_name = file_names[i]
    total_traces += 1

    # Process the file
    f = open(file_name, "r")    
    vehicle_count, crash, test_vectors = processFile(f)
    f.close()

    total_crashes += int(crash)

    # Update total observations
    total_observations += len(test_vectors)

    # Check to see if any of the vectors are new
    for v1 in test_vectors:
        unique = isUnique(v1, unique_vectors_seen)
        if unique:
            unique_vectors_seen.append(v1)

    # Used for the accumulative graph
    accumulative_graph.append(len(unique_vectors_seen))
    acuumulative_graph_vehicle_count.append(vehicle_count)


overall_coverage = round((len(unique_vectors_seen) / float(total_possible_observations)) * 100, 4)
crash_percentage = round(total_crashes / float(total_traces) * 100, 4)

print("Total traces considered:\t" + str(total_traces))
print("Total crashes:\t\t\t" + str(total_crashes))
print("Crash percentage:\t\t" + str(crash_percentage) + "%")
print("Total vectors considered:\t" + str(total_observations))
print("Total unique vectors seen:\t" + str(len(unique_vectors_seen)))
print("Total possible vectors:\t\t" + str(total_possible_observations))
print("Total coverage:\t\t\t" + str(overall_coverage) + "%")

# Get all the unique number of external vehicles
unique_vehicle_count = list(set(acuumulative_graph_vehicle_count))
unique_vehicle_count.sort()

# Convert to numpy arrays
accumulative_graph_coverage = (np.array(accumulative_graph) / total_possible_observations) * 100
acuumulative_graph_vehicle_count = np.array(acuumulative_graph_vehicle_count)

# Plotting the coverage per scenario            
plt.figure(1)
color_index = 0
previous_index = 0
for vc in unique_vehicle_count:
    scenario_indices = np.where(acuumulative_graph_vehicle_count == vc)[0]
    y_data = accumulative_graph_coverage[scenario_indices]
    x_data = np.arange(previous_index, len(y_data) + previous_index)
    previous_index += len(y_data) + 1 
    plt.scatter(x_data, y_data, color='C'+str(color_index), marker='o', label=str(vc) + " vehicle(s)", s=1)
    color_index += 1
plt.legend(loc='upper left')
plt.title("Reachable Set Coverage")
plt.xlabel("Scenario")
plt.ylabel("Reachable Set Coverage (%)")



print("----------------------------------")
print("-------Coverage as a metric-------")
print("----------------------------------")


# Coverage criteria in percentage
overall_coverage = overall_coverage - (overall_coverage * 0.50)
coverage_criteria = np.linspace(1, overall_coverage, 5)
coverage_criteria = getStep(coverage_criteria, 1)
coverage_criteria = coverage_criteria.astype(int)
output_data = {}

for c in coverage_criteria:
    output_data[str(c) + "_crashes"] = []
    output_data[str(c) + "_traces"] = []
    output_data[str(c) + "_coverage"] = []

for c in coverage_criteria:
    # Do 10 random samples
    print("Coverage criteria: " + str(c))
    for i in tqdm(np.arange(10)):

        # Shuffle the data
        random.shuffle(file_names)

        coverage = 0
        crashes_found = 0
        unique_vectors_seen = []

        # While coverage is less than 10%
        f_index = 0
        while (coverage < c) and (f_index < len(file_names)):
            # Get the filename
            file_name = file_names[f_index]
            f_index += 1

            # Process the file
            f = open(file_name, "r")    
            vehicle_count, crash, test_vectors = processFile(f)
            f.close()

            # Count how many crashes
            crashes_found += int(crash)

            # Check to see if any of the vectors are new
            for v1 in test_vectors:
                unique = isUnique(v1, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(v1)

            # Compute coverage
            coverage = (len(unique_vectors_seen) / float(total_possible_observations)) * 100

        # Save how many crashes required to get to that coverage criteria
        if f_index >= len(file_names):
            crashes_found = -1
        output_data[str(c) + "_crashes"].append(crashes_found)
        output_data[str(c) + "_traces"].append(f_index)
        output_data[str(c) + "_coverage"].append(round(coverage, 4))

# Print output
for c in coverage_criteria:
    print("-----------------------------")
    print("Coverage criteria: " + str(c) + "%")
    print("Test in test suite:\t" + str(output_data[str(c) + "_traces"]))
    print("Crashes in test suite:\t" + str(output_data[str(c) + "_crashes"]))
    print("Coverage in test suite:\t" + str(output_data[str(c) + "_coverage"]))

# Create the boxplot
data = []
labels = []
for key in output_data:
    if "_crashes" in key:
        label_name = key[0:key.rfind("_")]
        labels.append(label_name)
        data.append(output_data[key])
plt.figure(2)
plt.boxplot(data)
plt.xticks(np.arange(len(labels)) + 1, list(labels))
plt.xlabel("Coverage Criteria (%)")
plt.ylabel("Number of Crashes")



print("----------------------------------")
print("--------Crashes vs Coverage-------")
print("----------------------------------")

# Select 1000 test suites each with 100 test cases
# Plot the crashes vs coverage data

total_test_suites = 1000
tests_per_test_suite = [10, 25, 50, 100, 250, 500]

plt.figure(3)

for j in range(len(tests_per_test_suite)):
    # Get the number of tests
    test_number = tests_per_test_suite[j]

    coverage_data = []
    crash_data = []

    # For test suites
    print("Test suites with " + str(test_number) + " tests")
    for i in tqdm(np.arange(total_test_suites)):

        # Shuffle the data
        random.shuffle(file_names)

        # Init variables
        cov = 0
        crashes_found = 0
        unique_vectors_seen = []

        # For 100 test cases
        for f_index in range(test_number):
            # Get the filename
            file_name = file_names[f_index]

            # Process the file
            f = open(file_name, "r")    
            vehicle_count, crash, test_vectors = processFile(f)
            f.close()

            # Count how many crashes
            crashes_found += int(crash)

            # Check to see if any of the vectors are new
            for v1 in test_vectors:
                unique = isUnique(v1, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(v1)

            # Compute coverage
            cov = (len(unique_vectors_seen) / float(total_possible_observations)) * 100

        # Save the data for that test suite
        coverage_data.append(cov)
        crash_data.append(crashes_found)

    # Plot the data
    plt.scatter(coverage_data, crash_data, color='C' + str(j), marker='o', label="#Tests: " + str(test_number), s=1)

    # Compute the line of best fit
    m, b = np.polyfit(coverage_data, crash_data, 1)
    x_range = np.arange(0,100)
    plt.plot(x_range, m*x_range + b, c='C' + str(j))

plt.legend(loc='upper left')
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of Crashes")

plt.show()