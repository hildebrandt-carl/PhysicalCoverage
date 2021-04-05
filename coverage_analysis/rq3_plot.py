import random 
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt


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

# Load the results
save_name = "../results/rq3_"
final_coverage          = np.load(save_name + "coverage_" + str(args.scenario) + ".npy")
final_number_crashes    = np.load(save_name + "crashes_" + str(args.scenario) + ".npy")

# Use the tests_per_test_suite
tests_per_test_suite = [50, 100, 250, 500, 1000]

# For all the data plot it
plt.figure(args.scenario)

color_index = 0
for ind in range(final_coverage.shape[0]):
    # Expand the data
    random_selection_coverage_data  = final_coverage[ind,:]
    random_selection_crash_data     = final_number_crashes[ind,:]
    test_number                     = tests_per_test_suite[ind]

    # Plot the data
    plt.scatter(random_selection_coverage_data, random_selection_crash_data, color='C' + str(color_index), marker='o', label="#Tests: " + str(test_number), s=5)

    # # Compute the line of best fit
    # m, b = np.polyfit(random_selection_coverage_data, random_selection_crash_data, 1)
    # x_range = np.arange(worst_coverage, best_coverage, 0.1)
    # plt.plot(x_range, m*x_range + b, c='C' + str(color_index))
    
    # keep track of the color we are plotting
    color_index += 1

plt.legend()
plt.show()