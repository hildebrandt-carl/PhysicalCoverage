import scipy.stats
import random 
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from prettytable import PrettyTable


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,    help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=5,     help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',   type=int, default=30,    help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,     help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--greedy_sample',  type=int, default=50,    help="The unumber of samples considered by the greedy search")
parser.add_argument('--scenario',       type=str, default="",    help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,     help="The number of CPU cores available")
args = parser.parse_args()

new_steering_angle  = args.steering_angle
new_total_lines     = args.beam_count
new_max_distance    = args.max_distance
new_accuracy        = args.accuracy
greedy_sample_size  = args.greedy_sample

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

# Load the results
save_name = "../results/rq2_"
final_coverage          = np.load(save_name + "coverage_" + str(args.scenario) + ".npy")
final_number_crashes    = np.load(save_name + "crashes_" + str(args.scenario) + ".npy")

# Load the traces
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
traces = np.load(base_path + "traces_" + args.scenario + load_name)

# Compute the number of crashes
print("Counting the number of crashes")
total_crashes = 0
for t in traces:
    if np.isnan(t).any():
        total_crashes += 1
print("Total Crashes: " + str(total_crashes))


print("----------------------------------")
print("----------Plotting Data-----------")
print("----------------------------------")


# Use the tests_per_test_suite
if args.scenario == "beamng":
    total_random_test_suites = 1000
    test_suite_size = [100, 500, 1000]
    total_greedy_test_suites = 100
    greedy_sample_sizes = [2, 3, 4, 5, 10]
elif args.scenario == "highway":
    total_random_test_suites = 1000
    test_suite_size = [100, 500, 1000, 5000, 10000]
    total_greedy_test_suites = 100
    greedy_sample_sizes = [2, 3, 4, 5, 10]
else:
    exit()

# For all the data plot it
plt.figure(args.scenario)

# Create a table with the output
print("Results:")
t = PrettyTable(['#Test Suites', '#Test', 'Pearson Correlation', 'P-value', "r-squared", "P-Value", "Standard Error"])

color_index = 0
for ind in range(final_coverage.shape[0]):
    # Expand the data
    random_selection_coverage_data  = final_coverage[ind,:]
    random_selection_crash_data     = final_number_crashes[ind,:]
    test_number                     = test_suite_size[ind]

    # Convert crash data to a percentage
    random_selection_crash_percentage = (random_selection_crash_data / total_crashes) * 100

    # Set the x and y data
    x = random_selection_coverage_data
    y = random_selection_crash_percentage

    # Plot the data
    plt.scatter(x, y, color='C' + str(color_index), marker='o', label="#Tests: " + str(test_number), s=2, alpha=1)

    # Compute Pearson Correlation
    r, p = scipy.stats.pearsonr(x, y)

    # Compute the line of best fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    x_range = np.arange(np.min(x), np.max(x), 0.1)
    plt.plot(x_range, slope*x_range + intercept, c='C' + str(color_index))

    # Display data
    t.add_row([len(x), test_number, np.round(r, 4), np.round(p, 4), np.round(r_value, 4), np.round(p_value, 4), np.round(std_err,4)])
    
    # keep track of the color we are plotting
    color_index += 1

# Print the table
print(t)

# Plot the data
plt.title(args.scenario)
plt.xlabel("Physical Coverage  (%)")
plt.ylabel("Total Crashes Found (%)")
plt.legend()
plt.show()



