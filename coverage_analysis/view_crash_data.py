import argparse

import numpy as np
import glob as glob
import matplotlib.pyplot as plt

from general_functions import get_beam_numbers

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_tests',    type=int, default=1,      help="The number of tests used while computing coverage")
parser.add_argument('--scenario',           type=str, default="",     help="beamng/highway")
args = parser.parse_args()

# Get the base path
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/randomly_generated/processed/' + str(args.number_of_tests) + "/"

# Find all the crash files
all_files = glob.glob(base_path + "crash*.npy")

# Get all the beam numbers
beam_numbers = get_beam_numbers(all_files)

# Sort the files according to beam number
beam_numbers, all_files = zip(*sorted(zip(beam_numbers, all_files)))

# Creating the matplotlib
plt.figure(1)

# Got through each file:
for i in range(len(beam_numbers)):
    # Print processing information
    beam_number = beam_numbers[i]
    f = all_files[i]
    print("Processing beam{}: {}".format(beam_number, f[f.rfind("/")+1:]))

    # Used to keep track of how the total crashes and unique crashes changed over time
    accumulative_total_crashes = []
    accumulative_unique_crashes = []
    total_crash_count = 0
    unique_crash_count = 0

    # Load the data
    all_crash_data = np.load(f)

    # Used to find the unique values
    unique_crash_data = set()

    # Process the data
    for d in all_crash_data:
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
    assert(total_crash_count == all_crash_data.shape[0] - np.count_nonzero(np.isnan(all_crash_data)))
    assert(unique_crash_count == len(unique_crash_data))
    print("Total crashes: {}".format(total_crash_count))
    print("Unique crashes: {}".format(unique_crash_count))

    # Plot the data
    plt.plot(accumulative_total_crashes, linestyle="--", color='C' + str(i), label=str(i) + " beams")
    plt.plot(accumulative_unique_crashes, linestyle="-", color='C' + str(i))

plt.legend()
plt.title("Total crashes compared to unique crashes")
plt.xlabel("Randomly Generated Tests")
plt.ylabel("Total Crashes")
plt.grid()
plt.show()