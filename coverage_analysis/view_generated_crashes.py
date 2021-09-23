import os
import sys
import glob
import pickle
import argparse

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
parser.add_argument('--total_samples',          type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',               type=str, default="",   help="beamng/highway")
parser.add_argument('--ordered',                action='store_true')
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
original_data_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/physical_coverage/processed/' + str(args.total_samples) + "/"
generated_data_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/generated_tests/tests_single/processed/' + str(args.total_samples) + "/"
original_crash_files = glob.glob(original_data_path + "crash_*.npy")
generated_crash_files = glob.glob(generated_data_path + "crash_*.npy")
original_time_files = glob.glob(original_data_path + "time_*.npy")
generated_time_files = glob.glob(generated_data_path + "time_*.npy")

# Make sure you have all the files you need
assert(len(original_crash_files) >= 1)
assert(len(generated_crash_files) >= 1)
assert(len(original_time_files) >= 1)
assert(len(generated_time_files) >= 1)

# Get the beam numbers
random_crash_beam_numbers = get_beam_numbers(original_crash_files)
generated_crash_beam_numbers = get_beam_numbers(generated_crash_files)
random_time_beam_numbers = get_beam_numbers(original_time_files)
generated_time_beam_numbers = get_beam_numbers(generated_time_files)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(random_crash_beam_numbers) &
                    set(generated_crash_beam_numbers) &
                    set(random_time_beam_numbers) &
                    set(generated_time_beam_numbers)) 
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

# Create the shuffler if its needed
shuffler = np.random.permutation(len(original_crash_data[0]))

# Used to save when we switch from random to generated
switch_point_x = -1
switch_point_y = -1

# Create the plot
plt.figure(1)

for i in range(len(beam_numbers)):

    # Get the beam number
    beam_number = beam_numbers[i]
    print("Processing beam number: {}".format(beam_number))

    # Get data
    o_crash_data = original_crash_data[i]
    g_crash_data = generated_crash_data[i]
    o_time_data = original_time_data[i]
    g_time_data = generated_time_data[i]

    # Shuffle the random data if not ordered (removes how they were created with increasing amounts of vehicles)
    if not args.ordered:
        o_crash_data = o_crash_data[shuffler]
        o_time_data = o_time_data[shuffler]

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
        for c in c_data:
            if ~np.isinf(c):
                # Check if it is unique
                if c not in unique_crashes:
                    crash_counter += 1
                    unique_crashes.add(c)

        # crash_array[current_index] = (crash_counter / total_possible_crashes) * 100
        crash_array[current_index] = crash_counter
        times_array[current_index] = times_array[current_index - 1] + (c_time / 86400.0)
        current_index += 1

    # Save the switch point
    switch_point_x = times_array[current_index - 1]
    switch_point_y = crash_array[current_index - 1]

    for j in range(len(g_crash_data)):
        # Get the current crash and time data
        c_data = g_crash_data[j]
        c_time = g_time_data[j]
        # If it is a crash
        for c in c_data:
            if ~np.isinf(c):
                # Check if it is unique
                if c not in unique_crashes:
                    crash_counter += 1
                    unique_crashes.add(c)

        # crash_array[current_index] = (crash_counter / total_possible_crashes) * 100
        crash_array[current_index] = crash_counter
        times_array[current_index] = times_array[current_index - 1] + (c_time / 86400.0)
        current_index += 1

    plt.plot(times_array, crash_array, c="C" + str(i), label="RSR"+str(beam_number))

plt.axvline(x=switch_point_x, color='grey', linestyle='--', linewidth=2)
plt.axvline(x=times_array[-1], color='grey', linestyle='--', linewidth=1)
plt.axhline(y=switch_point_y, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=crash_array[-1], color='grey', linestyle='--', linewidth=1)

switch_point_y_interval = 0
switch_point_x_interval = 0
times_array_interval = 0
ran_text_x = 0
ran_text_y = 0
gen_text_x = 0
gen_text_y = 0

if args.scenario == "beamng":
    switch_point_y_interval = 1
    switch_point_x_interval = 100
    times_array_interval = 150
    ran_text_x = 1
    ran_text_y = 250
    gen_text_x = 90
    gen_text_y = 1800
elif args.scenario == "highway":
    switch_point_y_interval = 10
    switch_point_x_interval = 100
    times_array_interval = 1000
    ran_text_x = 15
    ran_text_y = 1800
    gen_text_x = 90
    gen_text_y = 1800

plt.text(switch_point_y_interval, switch_point_y, str(int(switch_point_y)), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")
plt.text(switch_point_y_interval, crash_array[-1], str(int(crash_array[-1])), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")
plt.text(switch_point_x, switch_point_x_interval, str(int(np.round(switch_point_x,0))), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")
plt.text(times_array[-1], times_array_interval, str(int(np.round(times_array[-1],0))), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")

plt.text(ran_text_x, ran_text_y, "Random Tests", fontsize=12, va='center', ha='center', color="black")
plt.text(gen_text_x, gen_text_y, "Generated Tests", fontsize=12, va='center', ha='center', color="black")


crash_increase = np.round(((crash_array[-1] - switch_point_y) / switch_point_y) * 100, 2)
time_increase = np.round(((times_array[-1] - switch_point_x) / switch_point_x) * 100, 2)
print("----------------------------------")
print("Percentage increase: {}".format(crash_increase))
print("Additional time increase: {}".format(time_increase))
print("----------------------------------")

time_interval = 0
crash_interval = 0
if args.scenario == "beamng":
    time_interval = 1
    crash_interval = 25
elif args.scenario == "highway":
    time_interval = 5
    crash_interval = 250

plt.xlabel("Time (days)")
plt.ylabel("Total unique crashes")
plt.xticks(np.arange(0, times_array[-1] + 1e-6, time_interval))
plt.yticks(np.arange(0, crash_array[-1] + 1e-6, crash_interval))
plt.legend(loc=4)
plt.title("Time increase: {}% - Crash increase: {}%".format(time_increase, crash_increase))
plt.grid(alpha=0.5)
plt.axes().minorticks_on()
plt.show()

