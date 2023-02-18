import sys
import glob
import argparse

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np

from tqdm import tqdm

from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",     help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="center_close/center_full")
parser.add_argument('--scenario',           type=str, default="",                                               help="beamng/highway")
args = parser.parse_args()

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = "*.npy"

# Checking the distribution
if not (args.distribution == "center_full" or args.distribution == "center_close"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '/mnt/extradrive3/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
crash_file_names = glob.glob(base_path + "crash_*.npy")

# Make sure we have enough samples
assert(len(crash_file_names) >= 1)

# Get the beam numbers
crash_beam_numbers = get_beam_number_from_file(crash_file_names)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(crash_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
crash_file_names = order_files_by_beam_number(crash_file_names, beam_numbers)

print("Note for each RRS they should be the same!!!!!!!!")
# Loop through each of the files and compute if there was a crash
for beam_number in beam_numbers:
    print("\nProcessing RRS{}".format(beam_number))
    key = "RRS{}".format(beam_number)

    # Get the  and crash files
    crashes = np.load(crash_file_names[beam_number-1], allow_pickle=True)

    # Call our function on each test in the trace
    results = np.zeros(args.number_of_tests)
    for index in tqdm(range(args.number_of_tests)):
            # Get the trace and crash data
            crash = crashes[index]

            # Check if this trace had a crash
            crash_detected = not (crash == None).all()

            # Append the results
            results[index] = crash_detected

    # Show the number of crashes
    total_crashes = np.sum(results)
    total_tests = len(results)
    total_non_crashes = total_tests - total_crashes
    percentage_crash = (total_crashes / total_tests) * 100
    print("Total crashes: {}/{} = {}%".format(total_crashes, total_tests, percentage_crash))



