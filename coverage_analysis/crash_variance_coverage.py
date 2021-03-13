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

def myround(x, base=5):
    return round(base * round(x/base), 5)

def generate_random_test_suite_coverage(process_number):
    global traces 

    print("Starting: " + str(process_number))
    # Randomly select how many tests to include
    number_traces = random.randint(1, traces.shape[0])
    # Randomly select indices to include
    selected_indices = random.sample(set(np.arange(traces.shape[0])), number_traces)
    # Init variables
    coverage = 0
    crashes = 0
    unique_vectors_seen = []
    # Compute the coverage for this test set
    for i in selected_indices:
        # Get the trace
        trace = traces[i]
        # Get all unique vectors from this trace
        for vector in trace:
            if not np.isnan(vector).any():
                unique = isUnique(vector, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(vector)
        # Check if there was a crash
        if np.isnan(trace).any():
            crashes += 1
    # Compute the coverage
    coverage = (len(unique_vectors_seen) / float(total_possible_observations)) * 100
    # Return the data
    print("Finished: " + str(process_number))
    return [coverage, crashes, number_traces]

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
parser.add_argument('--steering_angle',             type=int, default=30,    help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',                 type=int, default=5,     help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',               type=int, default=30,    help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',                   type=int, default=5,     help="What each vector is rounded to")
parser.add_argument('--total_samples',              type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',                   type=str, default="",    help="beamng/highway")
parser.add_argument('--total_random_test_suites',   type=int, default=1000,  help="Total random test suites to be generated")
args = parser.parse_args()

new_steering_angle  = args.steering_angle
new_total_lines     = args.beam_count
new_max_distance    = args.max_distance
new_accuracy        = args.accuracy
total_test_suites   = args.total_random_test_suites

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
traces = np.load(base_path + "traces" + load_name)

print("----------------------------------")
print("--------Crashes vs Coverage-------")
print("----------------------------------")

# Create a pool with x processes
total_processors = 64
pool =  multiprocessing.Pool(total_processors)

# Call our function total_test_suites times
result_object = []
for i in range(total_test_suites):
    result_object.append(pool.apply_async(generate_random_test_suite_coverage, args=([i])))

# Get the results
results = [r.get() for r in result_object]
results = np.array(results)
np.save("crash_variance_" + str(args.scenario), results)

# Close the pool
pool.close()

# Get the coverage / crashes / and number of tests
coverage = np.zeros(len(results))
crashes = np.zeros(len(results))
num_tests = np.zeros(len(results))
for i in range(len(results)):
    coverage[i] = results[i][0]
    crashes[i] = results[i][1]
    num_tests[i] = results[i][2]

# Plot the differnt relationships as a scatter plot
fig = plt.figure(1)
ax = fig.add_subplot(1, 3, 1)
ax.scatter(coverage, crashes, alpha=0.8, edgecolors='none', s=30)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of Crashes")
plt.title('Matplot scatter plot')
ax = fig.add_subplot(1, 3, 2)
ax.scatter(num_tests, crashes, alpha=0.8, edgecolors='none', s=30)
plt.xlabel("Number of tests in test suite")
plt.ylabel("Number of Crashes")
plt.title('Matplot scatter plot')
ax = fig.add_subplot(1, 3, 3)
ax.scatter(coverage, num_tests, alpha=0.8, edgecolors='none', s=30)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of tests in test suite")
plt.title('Matplot scatter plot')

# Create a box plot of the data
interval_size = 0.5
max_coverage = np.max(coverage) + interval_size
box_intervals = np.arange(0, max_coverage, interval_size)

# Create a dictionary to hold all the data
coverage_data = {}
crash_data = {}
for interval in box_intervals:
    coverage_data[str(myround(interval, interval_size))] = []
    crash_data[str(myround(interval, interval_size))] = []

# Break the data up into groups
for i in range(coverage.shape[0]):
    c = myround(coverage[i], interval_size)
    coverage_data[str(c)].append(coverage[i])
    crash_data[str(c)].append(crashes[i])
    
# Create box plot
plot_data = []
label_data = []
for key in crash_data:
    plot_data.append(crash_data[key])
    label_data.append(key)

# Creat the box plot
fig = plt.figure(2, figsize =(10, 7)) 
# Creating plot 
bp = plt.boxplot(plot_data) 
plt.xticks(np.arange(1, len(label_data) + 1), label_data)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Crashes")
# show plot 
plt.show() 
