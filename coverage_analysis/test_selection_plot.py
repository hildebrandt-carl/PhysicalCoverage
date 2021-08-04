import scipy.stats
import random 
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from prettytable import PrettyTable

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics
from test_selection_config import plot_config, unique_vector_config, compute_crash_hash


def crash_hasher(trace_number, hash_size):
    global traces

    # Determine if there is a crash
    trace = traces[trace_number]

    # Create the hash
    hash_value = np.nan

    # If there is no crash return none
    if not np.isnan(trace).any():
        return [np.nan]
    # Else return the hash
    else:
        hash_value = compute_crash_hash(trace, hash_size)

    return [hash_value]

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--beam_count',     type=int, default=5,     help="The number of beams used to vectorized the reachable set")
parser.add_argument('--total_samples',  type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",    help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,     help="The number of CPU cores available")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig(beam_count=args.beam_count)

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy
new_total_lines     = RSR.beam_count

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

# Load the results
save_name = "../results/rq3_"
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
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/processed/' + str(args.total_samples) + "/"
   
print("Loading: " + load_name)
traces = np.load(base_path + "traces_" + args.scenario + load_name)

# Get the hash_size
hash_size = unique_vector_config(args.scenario, number_of_seconds=1)

# Create a pool with x processes
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

jobs = []
# For all the different test suite sizes
for trace_i in range(len(traces)):
    jobs.append(pool.apply_async(crash_hasher, args=(trace_i, hash_size)))
    
# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

print("Done computing all the hash functions")

# Get the crash data
print("Computing unique crash values")
print("")
results = np.array(results).reshape(-1)
total_crashes = results[np.logical_not(np.isnan(results))]
print("Total crash count: " + str(total_crashes.shape[0]))

# Count the unique elements in an array
unique = np.unique(total_crashes)
print("Total unique crashes count: " + str(unique.shape[0]))
total_crashes = unique.shape[0]

print("----------------------------------")
print("----------Plotting Data-----------")
print("----------------------------------")

# Get the configuration
total_random_test_suites, test_suite_size, total_greedy_test_suites, greedy_sample_sizes = plot_config(args.scenario)

# For all the data plot it
plt.figure(args.scenario)

# Create a table with the output
print("Results:")
t = PrettyTable(['#Test Suites', '#Test', 'Slope', 'Intercept', "r-squared", "P-Value", "Standard Error"])

# Proxy for additional label
plt.plot([], [], ' ', label="Tests per test suite")
plt.plot([], [], ' ', label="Regression line")

color_index = 0
for ind in range(final_coverage.shape[0]):
    # Expand the data
    random_selection_coverage_data  = final_coverage[ind,:]
    random_selection_crash_data     = final_number_crashes[ind,:]
    test_number                     = test_suite_size[ind]

    # Convert crash data to a percentage
    random_selection_crash_percentage = (random_selection_crash_data / total_crashes) * 100
    # random_selection_crash_percentage = (random_selection_crash_data / test_number) * 100

    # Set the x and y data
    x = random_selection_coverage_data
    y = random_selection_crash_percentage

    # Compute Pearson Correlation
    # r, p = scipy.stats.pearsonr(x, y)

    # Compute the line of best fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    x_range = np.arange(np.min(x), np.max(x), 0.1)

    # Generate the label for the regression line
    lb = str(np.round(slope,2)) +"x+" + str(int(np.round(intercept,0)))
    if intercept < 0:
        lb = str(np.round(slope,2)) +"x" + str(int(np.round(intercept,0)))

    # Plot the line of best fit
    plt.plot(x_range, slope*x_range + intercept, c='C' + str(color_index), label=lb)

    # Plot the data
    plt.scatter(x, y, color='C' + str(color_index), marker='o', label=str(test_number), s=2, alpha=1)
    
    # Display data
    t_row = []
    t_row.append(len(x))
    t_row.append(test_number)
    t_row.append(np.round(slope, 4))
    t_row.append(np.round(intercept, 4))
    t_row.append(np.round(r_value, 4))
    t_row.append(np.round(p_value, 4))
    t_row.append(np.round(std_err, 4))
    t.add_row(t_row)
    
    # keep track of the color we are plotting
    color_index += 1

# Print the table
print(t)

# Get the legend handels so you can order them
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
new_handles = [handles[0], handles[1]]
new_labels = [labels[0], labels[1]]

top_row = 2
bottom_row = 2 + len(test_suite_size)
for i in range(len(test_suite_size)):
    new_handles.append(handles[bottom_row])
    new_handles.append(handles[top_row])
    new_labels.append(labels[bottom_row])
    new_labels.append(labels[top_row])
    top_row += 1
    bottom_row += 1

# Plot the data
plt.xlabel("Physical Coverage  (%)")
plt.ylabel("Total Unique Crashes Found (%)")
# plt.legend(markerscale=5)
plt.legend(new_handles, new_labels, markerscale=5, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=len(test_suite_size) + 1, handletextpad=0.1)
# plt.tight_layout()
plt.show()




