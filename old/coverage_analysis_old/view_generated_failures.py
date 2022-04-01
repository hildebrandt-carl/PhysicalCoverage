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
parser.add_argument('--stall_rounding_factor',  type=int, default= 0,   help="How many decimal points to round to")
parser.add_argument('--ordered',                action='store_true')
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy

# Declares the rounding factor. This determines how likely two stalls are identified as unique
rounding_factor = args.stall_rounding_factor

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
original_data_path = '/media/carl/DataDrive/PhysicalCoverageData/' + str(args.scenario) +'/random_tests/physical_coverage/processed/' + str(args.total_samples) + "/"
generated_data_path = '/media/carl/DataDrive/PhysicalCoverageData/' + str(args.scenario) +'/generated_tests/tests_single/processed/' + str(args.total_samples) + "/"

original_crash_files = glob.glob(original_data_path + "crash_*.npy")
generated_crash_files = glob.glob(generated_data_path + "crash_*.npy")
original_velocity_files = glob.glob(original_data_path + "ego_velocities_*.npy")
generated_velocity_files = glob.glob(generated_data_path + "ego_velocities_*.npy")
original_stall_files = glob.glob(original_data_path + "stall_information_*.npy")
generated_stall_files = glob.glob(generated_data_path + "stall_information_*.npy")
original_time_files = glob.glob(original_data_path + "time_*.npy")
generated_time_files = glob.glob(generated_data_path + "time_*.npy")

# Make sure you have all the files you need
assert(len(original_crash_files) >= 1)
assert(len(generated_crash_files) >= 1)
assert(len(original_time_files) >= 1)
assert(len(generated_time_files) >= 1)
assert(len(original_velocity_files) >= 1)
assert(len(generated_velocity_files) >= 1)
assert(len(original_stall_files) >= 1)
assert(len(generated_stall_files) >= 1)

# Get the beam numbers
random_crash_beam_numbers       = get_beam_numbers(original_crash_files)
generated_crash_beam_numbers    = get_beam_numbers(generated_crash_files)
random_time_beam_numbers        = get_beam_numbers(original_time_files)
generated_time_beam_numbers     = get_beam_numbers(generated_time_files)
random_velocity_beam_numbers    = get_beam_numbers(original_velocity_files)
generated_velocity_beam_numbers = get_beam_numbers(generated_velocity_files)
random_stall_beam_numbers       = get_beam_numbers(original_stall_files)
generated_stall_beam_numbers    = get_beam_numbers(generated_stall_files)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(random_velocity_beam_numbers) &
                    set(generated_velocity_beam_numbers) &
                    set(random_stall_beam_numbers) &
                    set(generated_stall_beam_numbers) &
                    set(random_crash_beam_numbers) &
                    set(generated_crash_beam_numbers) &
                    set(random_time_beam_numbers) &
                    set(generated_time_beam_numbers)) 
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
original_velocity_files = order_by_beam(original_velocity_files, beam_numbers)
generated_velocity_files = order_by_beam(generated_velocity_files, beam_numbers)
original_stall_files = order_by_beam(original_stall_files, beam_numbers)
generated_stall_files = order_by_beam(generated_stall_files, beam_numbers)
original_crash_files = order_by_beam(original_crash_files, beam_numbers)
generated_crash_files = order_by_beam(generated_crash_files, beam_numbers)
original_time_files = order_by_beam(original_time_files, beam_numbers)
generated_time_files = order_by_beam(generated_time_files, beam_numbers)

print("Working with the following beam numbers: {}".format(beam_numbers))

original_crash_data = []
generated_crash_data = []
original_velocity_data = []
generated_velocity_data = []
original_stall_data = []
generated_stall_data = []
original_time_data = []
generated_time_data = []

for i in range(len(original_crash_files)):
    original_velocity_data.append(np.load(original_velocity_files[i]))
    generated_velocity_data.append(np.load(generated_velocity_files[i]))
    original_stall_data.append(np.load(original_stall_files[i]))
    generated_stall_data.append(np.load(generated_stall_files[i]))   
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
    o_vel_data      = original_velocity_data[i]
    g_vel_data      = generated_velocity_data[i]
    o_stall_data    = original_stall_data[i]
    g_stall_data    = generated_stall_data[i]
    o_crash_data    = original_crash_data[i]
    g_crash_data    = generated_crash_data[i]
    o_time_data     = original_time_data[i]
    g_time_data     = generated_time_data[i]

    # Shuffle the random data if not ordered (removes how they were created with increasing amounts of vehicles)
    if not args.ordered:
        o_vel_data = o_vel_data[shuffler]
        o_stall_data = o_stall_data[shuffler]
        o_crash_data = o_crash_data[shuffler]
        o_time_data = o_time_data[shuffler]

    # Find the unique failures
    unique_crashes = set()
    crash_counter = 0

    stall_info_set = set()
    stall_count = 0

    failures_array = np.zeros(len(o_crash_data) + len(g_crash_data))
    times_array = np.zeros(len(o_time_data) + len(g_time_data))

    current_index = 0

    for j in range(len(o_crash_data)):
        # Get the current crash and time data
        c_data          = o_crash_data[j]
        c_time          = o_time_data[j]
        c_vel           = o_vel_data[j]
        c_stall_info    = o_stall_data[j]

        if args.scenario == "beamng":
            # Used to only count a stall once
            currently_stalling = True

            # Check for a stall wait 1 second before starting the stall check
            for k in range(len(c_vel)):
                vel = c_vel[k]
                
                # Ignore any inf values's
                if np.sum(np.isinf(c_stall_info[k])) > 0:
                    continue

                # Get the angle and distance to closest obstacle
                info = tuple(np.round(c_stall_info[k][0:2], rounding_factor))

                # Get the total number of open lines
                wayout = c_stall_info[k][2]

                if np.sum(vel) < 0.01:
                    if not currently_stalling:
                        currently_stalling = True
                        if wayout > 5:
                            if info not in stall_info_set:
                                stall_info_set.add(info)
                                stall_count += 1

                # You are only out of stall once you start moving again
                if np.sum(vel) > 5:
                    currently_stalling = False
        else:
            stall_count = 0 

        # If it is a crash
        for c in c_data:
            if ~np.isinf(c):
                # Check if it is unique
                if c not in unique_crashes:
                    crash_counter += 1
                    unique_crashes.add(c)

        # crash_array[current_index] = (crash_counter / total_possible_crashes) * 100
        failures_array[current_index] = crash_counter + stall_count
        times_array[current_index] = times_array[current_index - 1] + (c_time / 86400.0)
        current_index += 1

    # Save the switch point
    switch_point_x = times_array[current_index - 1]
    switch_point_y = failures_array[current_index - 1]

    for j in range(len(g_crash_data)):
        # Get the current crash and time data
        c_data          = g_crash_data[j]
        c_time          = g_time_data[j]
        c_vel          = g_vel_data[j]
        c_stall_info    = g_stall_data[j]


        if args.scenario == "beamng":
            # Used to only count a stall once
            currently_stalling = True

            # Check for a stall
            for k in range(len(c_vel)):
                vel = c_vel[k]

                # Ignore any inf values's
                if np.sum(np.isinf(c_stall_info[k])) > 0:
                    continue

                # Get the angle and distance to closest obstacle
                info = tuple(np.round(c_stall_info[k][0:2], rounding_factor))

                # Get the total number of open lines
                wayout = c_stall_info[k][2]

                if np.sum(vel) < 0.01:
                    if not currently_stalling:
                        currently_stalling = True
                        if wayout > 5:
                            if info not in stall_info_set:
                                stall_info_set.add(info)
                                stall_count += 1
                        
                        # We can only count a stall for the generated tests once, as its stalling over the same sequence multiple times
                        # break

                        # You are only out of stall once you start moving again
                if np.sum(vel) > 5:
                    currently_stalling = False
        else:
            stall_count = 0

        # If it is a crash
        for c in c_data:
            if ~np.isinf(c):
                # Check if it is unique
                if c not in unique_crashes:
                    crash_counter += 1
                    unique_crashes.add(c)

        # crash_array[current_index] = (crash_counter / total_possible_crashes) * 100
        failures_array[current_index] = crash_counter + stall_count
        times_array[current_index] = times_array[current_index - 1] + (c_time / 86400.0)
        current_index += 1

    plt.plot(times_array, failures_array, c="C" + str(i), label="RSR"+str(beam_number))

plt.axvline(x=switch_point_x, color='grey', linestyle='--', linewidth=2)
plt.axvline(x=times_array[-1], color='grey', linestyle='--', linewidth=1)
plt.axhline(y=switch_point_y, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=failures_array[-1], color='grey', linestyle='--', linewidth=1)

switch_point_y_interval = 0
switch_point_x_interval = 0
times_array_interval = 0
ran_text_x = 0
ran_text_y = 0
gen_text_x = 0
gen_text_y = 0

if args.scenario == "beamng":
    switch_point_y_interval = 5
    switch_point_x_interval = 250
    times_array_interval = 250
    ran_text_x = 6
    ran_text_y = 100
    gen_text_x = 8.5
    gen_text_y = 100
    time_shift = 0
elif args.scenario == "highway":
    switch_point_y_interval = 40
    switch_point_x_interval = 1000
    times_array_interval = 1000
    ran_text_x = 55
    ran_text_y = 4000
    gen_text_x = 90
    gen_text_y = 4000
    time_shift = 3

plt.text(switch_point_y_interval, switch_point_y, str(int(switch_point_y)), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")
plt.text(switch_point_y_interval, failures_array[-1], str(int(failures_array[-1])), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")
plt.text(switch_point_x, switch_point_x_interval, str(np.round(switch_point_x,2)), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")
plt.text(times_array[-1] - time_shift, times_array_interval, str(np.round(times_array[-1],2)), fontsize=10, va='center', ha='center', backgroundcolor='w', color="grey")

plt.text(ran_text_x, ran_text_y, "Random\nTests", fontsize=12, va='center', ha='center', color="black")
plt.text(gen_text_x, gen_text_y, "Generated\nTests", fontsize=12, va='center', ha='center', color="black")


crash_increase = np.round(((failures_array[-1] - switch_point_y) / switch_point_y) * 100, 2)
time_increase = np.round(((times_array[-1] - switch_point_x) / switch_point_x) * 100, 2)
print("----------------------------------")
print("Percentage increase: {}".format(crash_increase))
print("Additional time increase: {}".format(time_increase))
print("----------------------------------")

time_interval = 0
crash_interval = 0
if args.scenario == "beamng":
    time_interval = 1
    crash_interval = 50
elif args.scenario == "highway":
    time_interval = 10
    crash_interval = 500

plt.xlabel("Time (days)")
plt.ylabel("Total Unique Failures")
plt.xticks(np.arange(0, times_array[-1] + 1e-6, time_interval))
plt.yticks(np.arange(0, failures_array[-1] + 1e-6, crash_interval))
plt.legend(loc=0)
plt.grid(alpha=0.5)
plt.axes().minorticks_on()


plt.figure(3)
difference = 10
grad_array = []
t_array = []
for i in range(len(failures_array)-difference):
    y1 = failures_array[i]
    y2 = failures_array[i+difference]
    x1 = times_array[i]
    x2 = times_array[i+difference]

    m = (y2-y1)/(x2-x1)
    t = (x2 + x1) / 2

    grad_array.append(m)
    t_array.append(t)

from scipy.ndimage.filters import uniform_filter1d
grad_array = uniform_filter1d(grad_array, size=1000)

plt.plot(t_array, grad_array)
plt.xlabel("Time (days)")
plt.ylabel("Rate of Failures over Time")

plt.show()