import os
import sys
import glob
import pickle
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from general_functions import order_by_beam
from general_functions import get_beam_numbers

from general.environment_configurations import RSRConfig
from general.environment_configurations import HighwayKinematics

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

# Get the file names
original_data_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/randomly_generated/processed/' + str(args.total_samples) + "/"
generated_data_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/generated_tests/tests_single/processed/' + str(args.total_samples) + "/"
original_crash_files = glob.glob(original_data_path + "crash_*.npy")
generated_crash_files = glob.glob(generated_data_path + "crash_*.npy")
original_time_files = glob.glob(original_data_path + "time_*.npy")
generated_time_files = glob.glob(generated_data_path + "time_*.npy")
print(generated_time_files)

# Get the beam numbers
random_crash_beam_numbers = get_beam_numbers(original_crash_files)
generated_crash_beam_numbers = get_beam_numbers(generated_crash_files)
generated_time_beam_numbers = get_beam_numbers(original_time_files)
generated_time_beam_numbers = get_beam_numbers(generated_time_files)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(random_crash_beam_numbers) & set(generated_crash_beam_numbers) & set(generated_time_beam_numbers) & set(generated_time_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
original_crash_files = order_by_beam(original_crash_files, beam_numbers)
generated_crash_files = order_by_beam(generated_crash_files, beam_numbers)
original_time_files = order_by_beam(original_time_files, beam_numbers)
generated_time_files = order_by_beam(generated_time_files, beam_numbers)

print("Working with the following beam numbers: {}".format(beam_numbers))


original_crash_data = []
generated_crash_data = []
original_time_data = []
generated_time_data = []

for i in range(len(original_crash_files)):
    original_crash_data.append(np.load(original_crash_files[i]))
    generated_crash_data.append(np.load(generated_crash_files[i]))
    original_time_data.append(np.load(original_time_files[i]))
    generated_time_data.append(np.load(generated_time_files[i]))

# Sort the names
print("Loading Complete")


print("----------------------------------")
print("--------Plotting Data-------------")
print("----------------------------------")

# Create the plot
plt.figure(1)

for i in range(len(beam_numbers)):

    # Get the beam number
    beam_number = beam_numbers[i]
    print("Processing beam number: {}".format(beam_number))

    # Used to save when we switch from random to generated
    switch_point = -1

    # Get data
    o_crash_data = original_crash_data[i]
    g_crash_data = generated_crash_data[i]
    o_time_data = original_time_data[i]
    g_time_data = generated_time_data[i]

    # Find the unique crashes
    unique_crashes = set()
    crash_counter = 0
    crash_array = np.zeros(len(o_crash_data) + len(g_crash_data))
    times_array = np.zeros(len(o_time_data) + len(g_time_data))

    current_index = 0

    for j in range(len(o_crash_data)):
        # Get the current crash and time data
        c_data = o_crash_data[j]
        c_time = o_time_data[j]


        # If it is a crash
        if not np.isnan(c_data):
            # Check if it is unique
            if c_data not in unique_crashes:
                crash_counter += 1
                unique_crashes.add(c_data)

        crash_array[current_index] = crash_counter
        times_array[current_index] = times_array[current_index - 1] + c_time
        current_index += 1

    # Save the switch point
    if switch_point != -1:
        assert(switch_point == times_array[current_index - 1])
    switch_point = times_array[current_index - 1]

    for j in range(len(g_crash_data)):
        # Get the current crash and time data
        c_data = g_crash_data[j]
        c_time = g_time_data[j]
        # If it is a crash
        if not np.isnan(c_data):
            # Check if it is unique
            if c_data not in unique_crashes:
                crash_counter += 1
                unique_crashes.add(c_data)

        crash_array[current_index] = crash_counter
        times_array[current_index] = times_array[current_index - 1] + c_time
        current_index += 1

    plt.plot(times_array, crash_array, c="C" + str(i), label="RSR"+str(beam_number))

plt.axvline(x=switch_point, color='red', linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("Total crashes")
plt.legend()
plt.show()

