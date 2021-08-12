import argparse

import numpy as np
import glob as glob
import matplotlib.pyplot as plt

from general_functions import order_by_beam
from general_functions import get_beam_numbers

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_tests',    type=int, default=1,      help="The number of tests used while computing coverage")
parser.add_argument('--scenario',           type=str, default="",     help="beamng/highway")
args = parser.parse_args()

# Get the base path
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/processed/' + str(args.number_of_tests) + "/"

# Find all the crash files
crash_files = glob.glob(base_path + "crash*.npy")
time_files = glob.glob(base_path + "time*.npy")

assert(len(crash_files) > 1)
assert(len(crash_files) == len(time_files))

# Get all the beam numbers
crash_beam_numbers = get_beam_numbers(crash_files)
time_beam_numbers = get_beam_numbers(time_files)

# Get the beam numbers that we have
beam_numbers = list(set(crash_beam_numbers) | set(time_beam_numbers))

# Sort the files according to beam number
crash_files = order_by_beam(crash_files, beam_numbers)
time_files = order_by_beam(time_files, beam_numbers)

# Creating the matplotlib
plt.figure(1)

# Select what beam number to plot
plot_beam_number = 5

# Print processing information
beam_number = beam_numbers[plot_beam_number]
print("Processing beam: {}".format(plot_beam_number))

# Load the files
crash_f = crash_files[plot_beam_number]
time_f = time_files[plot_beam_number]

# Used to keep track of how the total crashes and unique crashes changed over time
time_of_crash = []
time_of_unique_crash = []

# Load the data
all_crash_data = np.load(crash_f)
all_time_data = np.load(time_f)

# Used to find the unique values
unique_crash_data = set()

# Make sure that the data is correct
assert(len(all_crash_data) == len(all_time_data))

# Process the data
for j in range(len(all_crash_data)):

    # Get the crash and time
    crash_vector = all_crash_data[j]
    test_time = all_time_data[j]

    # If it is a crash
    if not np.isnan(crash_vector):

        # Add to the total crashes
        time_of_crash.append(test_time)

        # Check if it is unique
        if crash_vector not in unique_crash_data:
            unique_crash_data.add(crash_vector)
            time_of_unique_crash.append(test_time)


print("Total traces: {}".format(len(all_crash_data)))
print("Total crashes: {}".format(len(time_of_crash)))
print("Total unique crashes: {}".format(len(unique_crash_data)))
print("Crash percentage: {}%".format((len(time_of_crash) / len(all_crash_data)) * 100))

# Plot the data
time_of_unique_crash = sorted(time_of_unique_crash)
time_of_crash = sorted(time_of_crash)

unique_percentage_before = (np.arange(len(time_of_unique_crash)) / len(time_of_unique_crash)) * 100
total_percentage_before = (np.arange(len(time_of_crash)) / len(time_of_crash)) * 100
plt.scatter(time_of_unique_crash, unique_percentage_before, s=2, c="C0")
plt.scatter(time_of_crash, total_percentage_before, s=2, c="C1")

plt.xticks(np.arange(0, 25.01, 1.0))
plt.yticks(np.arange(0, 100.01, 10))
plt.xlabel("Time before crash")
plt.ylabel("Percentage of crashes")
plt.grid()
plt.show()

