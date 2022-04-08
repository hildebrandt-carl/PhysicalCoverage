import sys
import argparse

import numpy as np
import glob as glob
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number

# Used to shuffle two arrays together
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_tests', type=int, default=1,  help="The number of tests used while computing coverage")
parser.add_argument('--distribution',    type=str, default="",   help="linear/center_close/center_mid")
parser.add_argument('--scenario',        type=str, default="", help="beamng/highway")
parser.add_argument('--ordered',         action='store_true')
args = parser.parse_args()

# Checking the distribution
if not (args.distribution == "linear" or args.distribution == "center_close" or args.distribution == "center_mid"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '/media/carl/DataDrive/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
crash_files = glob.glob(base_path + "crash_*")
stall_files = glob.glob(base_path + "stall_*")



# Check that you have found the files
assert(len(crash_files) >= 1)
assert(len(stall_files) >= 1)

# Get all the beam numbers
crash_RRS_numbers = get_beam_number_from_file(crash_files)
stall_RRS_numbers = get_beam_number_from_file(stall_files)
RRS_numbers = list(set(crash_RRS_numbers) | set(stall_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the files according to RRS number
crash_files = order_files_by_beam_number(crash_files, RRS_numbers)
stall_files = order_files_by_beam_number(stall_files, RRS_numbers)

# Load the data
crash_data = np.load(crash_files[0])
stall_data = np.load(stall_files[0], allow_pickle=True)

# All the data for all RSR should be exactly the same
# Check that this is the case
for i in range(len(crash_files)-1):
    # Load the other data
    c_data = np.load(crash_files[i])
    s_data = np.load(stall_files[i], allow_pickle=True)

    # Check that they are the same
    assert(np.all(crash_data==c_data))
    assert(np.all(stall_data==s_data))

# Make sure that they are the same size
assert(np.shape(crash_data)[0] == np.shape(stall_data)[0])

# Creating the matplotlib
plt.figure(1)

# Randomly shuffle the data
if not args.ordered:
    crash_data, stall_data = unison_shuffled_copies(crash_data, stall_data)

# Used for plotting
accumulative_total_crashes = np.zeros(args.number_of_tests)
accumulative_unique_crashes = np.zeros(args.number_of_tests)
accumulative_total_stalls = np.zeros(args.number_of_tests)
accumulative_unique_stalls = np.zeros(args.number_of_tests)
accumulative_total_failures = np.zeros(args.number_of_tests)
accumulative_unique_failures = np.zeros(args.number_of_tests)

# Used to find the unique values
unique_crash_data = set()
unique_stall_data = set()
unique_failure_data = set()

# Keep track of the overall tests success rate
tests_with_no_failures = 0
tests_with_failures = 0
tests_with_no_crashes = 0
tests_with_crashes = 0
tests_with_no_stalls = 0
tests_with_stalls = 0

total_crash_count = 0
total_stall_count = 0
total_failure_count = 0
unique_crash_count = 0
unique_stall_count = 0
unique_failure_count = 0

# Go through each of the tests
for i in tqdm(range(crash_data.shape[0])):

    # This will throw an error we need to check for None
    contains_crash = ~(np.isnan(crash_data[i]).all())
    contains_stall = ~(np.isnan(stall_data[i]).all())

    # Check if this trace had a crash or a stall
    if contains_crash and contains_stall:
        tests_with_failures += 1
        tests_with_crashes += 1
        tests_with_stalls += 1
    elif contains_crash:
        tests_with_failures += 1
        tests_with_crashes += 1
        tests_with_no_stalls += 1
    elif contains_stall:
        tests_with_failures += 1
        tests_with_stalls += 1
        tests_with_no_crashes += 1
    else:
        tests_with_no_failures += 1
        tests_with_no_crashes += 1
        tests_with_no_stalls += 1

    # Go through each of the possible crashes and stalls in the test
    for j in range(crash_data.shape[1]):
        # Get the crash data
        c = crash_data[i][j]
        # Get the stall data
        s = stall_data[i][j]

        # If it is a crash
        if c is not None:
            # Add to the total crashes
            total_crash_count += 1
            total_failure_count += 1
            # Check if it is a unique crash
            if c not in unique_crash_data:
                unique_crash_data.add(c)
                unique_crash_count += 1
            # Check if it is a unique failure
            if c not in unique_failure_data:
                unique_failure_data.add(c)
                unique_failure_count += 1

        # If it is a stall
        if s is not None:
            # Add to the total crashes
            total_stall_count += 1
            total_failure_count += 1
            # Check if it is unique
            if s not in unique_stall_data:
                unique_stall_data.add(s)
                unique_stall_count += 1
            # Check if it is a unique failure
            if s not in unique_failure_data:
                unique_failure_data.add(s)
                unique_failure_count += 1

        # Add the counts to the accumulative graph
        accumulative_total_crashes[i]   = total_crash_count
        accumulative_unique_crashes[i]  = unique_crash_count
        accumulative_total_stalls[i]    = total_stall_count
        accumulative_unique_stalls[i]   = unique_stall_count
        accumulative_total_failures[i]  = total_failure_count
        accumulative_unique_failures[i] = unique_failure_count

total_runs = args.number_of_tests
print("Total runs: {}".format(total_runs))
print("")
print("Total runs with no failures\t: {} - {}% success rate".format(tests_with_no_failures, (np.round((tests_with_no_failures / total_runs) * 100,2))))
print("Total runs with no crashes\t: {} - {}% success rate".format(tests_with_no_crashes, (np.round((tests_with_no_crashes / total_runs) * 100, 2))))
print("Total runs with no stalls\t: {} - {}% success rate".format(tests_with_no_stalls, (np.round((tests_with_no_stalls / total_runs) * 100, 2))))
print("")
print("Total runs with failures\t: {} - {}% failure rate".format(tests_with_failures, (np.round((tests_with_failures / total_runs) * 100, 2))))
print("Total runs with crashes\t\t: {} - {}% failure rate".format(tests_with_crashes, (np.round((tests_with_crashes / total_runs) * 100, 2))))
print("Total runs with stalls\t\t: {} - {}% failure rate".format(tests_with_stalls, (np.round((tests_with_stalls / total_runs) * 100, 2))))

print("-------------------------")

print("Total failures: {}".format(total_failure_count))
print("Total unique failures: {}".format(unique_failure_count))
print("")
print("Total crashes: {}".format(total_crash_count))
print("Total unique crashes: {}".format(unique_crash_count))
print("")
print("Total stalls: {}".format(total_stall_count))
print("Total unique stalls: {}".format(unique_stall_count))

print("-------------------------")

# Plot the data
plt.plot(accumulative_total_crashes, linestyle="--", color='C0', label="All crashes")
plt.plot(accumulative_unique_crashes, linestyle="-", color='C0', label="Unique crashes")

plt.plot(accumulative_total_stalls, linestyle="--", color='C1', label="All stalls")
plt.plot(accumulative_unique_stalls, linestyle="-", color='C1', label="Unique stalls")

plt.plot(accumulative_total_failures, linestyle="--", color='C2', label="All failures")
plt.plot(accumulative_unique_failures, linestyle="-", color='C2', label="Unique failures")

plt.legend()
plt.xlabel("Randomly Generated Tests")
plt.ylabel("Total count")
plt.ticklabel_format(style='plain')
plt.grid(alpha=0.5)
plt.show()