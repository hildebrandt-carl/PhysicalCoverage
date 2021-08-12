import argparse

import numpy as np
import glob as glob
import matplotlib.pyplot as plt

from general_functions import order_by_beam
from general_functions import get_beam_numbers

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_tests',    type=int, default=1,      help="The number of tests used while computing coverage")
parser.add_argument('--scenario',           type=str, default="",     help="beamng/highway")
parser.add_argument('--ordered',            action='store_true')
args = parser.parse_args()

# Get the base path
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/processed/' + str(args.number_of_tests) + "/"

# Find all the crash files
all_files = glob.glob(base_path + "crash*.npy")
print(base_path)
assert(len(all_files) > 1)

# Get all the beam numbers
beam_numbers = get_beam_numbers(all_files)
beam_numbers = sorted(beam_numbers)

# Sort the files according to beam number
all_files = order_by_beam(all_files, beam_numbers)

# Make sure all crash data is the same
crash_data = None
for i in range(len(beam_numbers)):
    beam_number = beam_numbers[i]
    print("Loading beam number: {}".format(beam_number))
    f = all_files[i]

    # Load the data 
    if crash_data is None:
        crash_data = np.load(f)
    else:
        tmp_crash_data = np.load(f)
        assert(np.allclose(crash_data, tmp_crash_data, equal_nan=True))

# Creating the matplotlib
plt.figure(1)

# Randomly shuffle the data
if not args.ordered:
    np.random.shuffle(crash_data)

print("Processing")
# Used to keep track of how the total crashes and unique crashes changed over time
accumulative_total_crashes = []
accumulative_unique_crashes = []
total_crash_count = 0
unique_crash_count = 0

# Used to find the unique values
unique_crash_data = set()

# Process the data
for d in crash_data:
    # If it is a crash
    if not np.isnan(d):

        # Add to the total crashes
        total_crash_count += 1

        # Check if it is unique
        if d not in unique_crash_data:
            unique_crash_data.add(d)
            unique_crash_count += 1

    # Add the counts to the accumulative graph
    accumulative_total_crashes.append(total_crash_count)
    accumulative_unique_crashes.append(unique_crash_count)

# Print the final information
assert(total_crash_count == crash_data.shape[0] - np.count_nonzero(np.isnan(crash_data)))
assert(unique_crash_count == len(unique_crash_data))
print("Total crashes: {}".format(total_crash_count))
print("Unique crashes: {}".format(unique_crash_count))

# If its not ordered show where the number of external vehicles changed
if args.ordered:
    interval = int(np.round(crash_data.shape[0] / 10.0, 0))
    line_positions = np.arange(0, crash_data.shape[0] +0.1, interval)
    for l in line_positions:
        plt.axvline(x=l, color='red', linestyle='--')

# Plot the data
plt.plot(accumulative_total_crashes, linestyle="--", color='C' + str(i), label="All crashes")
plt.plot(accumulative_unique_crashes, linestyle="-", color='C' + str(i), label="Unique crashes")

plt.legend()
plt.title("Total crashes compared to unique crashes")
plt.xlabel("Randomly Generated Tests")
plt.ylabel("Total Crashes")
plt.ticklabel_format(style='plain')
plt.grid()
plt.show()