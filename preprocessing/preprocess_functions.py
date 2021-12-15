
import os
import re
import sys
import glob
import math
import random 
import argparse

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/trace_processing")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from general.crash_oracle import hash_crash

from general.environment_configurations import RSRConfig
from general.environment_configurations import HighwayKinematics

def vector_conversion_distribution(vector, steering_angle, max_distance, total_lines, original_total_lines):

    # Fix the vector to have to correct max_distance
    vector = np.clip(vector, 0, max_distance)

    # Get how many degrees between each line
    line_space = (steering_angle * 2) / float(original_total_lines - 1)

    # Get the starting lines angle
    left_index = int(len(vector) / 2)
    right_index = int(len(vector) / 2)
    current_steering_angle = 0
    if (original_total_lines % 2) == 0: 
        current_steering_angle = line_space / 2
        left_index -= 1

    # Floating point tolerance
    tolerance = 1e-6

    # This is an overapproximation
    beam_count = 0
    while current_steering_angle < (steering_angle - tolerance):
        left_index -= 1
        right_index += 1
        current_steering_angle += line_space

    # Get the corrected steering angle
    steering_angle_corrected_vector = np.array(vector[left_index:right_index+1])

    # Updates are here when selecting angles (Each beam represents a 2 degree beam)
    # Beam 0 starts at -30 degrees
    # Beam 29 ends at 30 degrees
    angle_distribution = {
        1:  [0],
        2:  [-6, 6],
        3:  [-10, 0, 10],
        4:  [-15, -6, 6, 15],
        5:  [-15, -8, 0, 8, 15],
        6:  [-20, -10, -6, 6, 10, 20],
        7:  [-20, -12, -6, 0, 6, 12, 20],
        8:  [-20, -14, -8, 4, 4, 8, 14, 20],
        9:  [-26, -18, -12, 6, 0, 6, 12, 18, 26],
        10: [-28, -22, -12, -6, -2, 2, 6, 12, 22, 28],
    }

    # angle_distribution = {
    #     1:  [0],
    #     2:  [-30, 30],
    #     3:  [-30, 0, 30],
    #     4:  [-30, -10, 10, 30],
    #     5:  [-30, -15, 0, 15, 30],
    #     6:  [-30, -18, -6, 6, 18, 30],
    #     7:  [-30, -20, -10, 0, 10, 20, 30],
    #     8:  [-30, -21.5, -13, -4.6, 4.5, 13, 21.5, 30],
    #     9:  [-30, -22.5, -15, 7.5, 0, 7.5, 15, 22.5, 30],
    #     10: [-30, -23.4, -16.8, -10.2, -3.6, 3.6, 10.2, 16.8, 23.4, 30],
    # }

    current_distribution = angle_distribution[total_lines]
    # Compute the indexes
    idx = np.round(((np.array(current_distribution) + 30) / (60/29)),0)
    idx = np.clip(idx, 0, 29).astype(int)
    final_vector = steering_angle_corrected_vector[idx]
    return final_vector

def vector_conversion_centralized(vector, steering_angle, max_distance, total_lines, original_total_lines):

    # Fix the vector to have to correct max_distance
    vector = np.clip(vector, 0, max_distance)

    # Get how many degrees between each line
    line_space = (steering_angle * 2) / float(original_total_lines - 1)

    # Get the starting lines angle
    left_index = int(len(vector) / 2)
    right_index = int(len(vector) / 2)
    current_steering_angle = 0
    if (original_total_lines % 2) == 0: 
        current_steering_angle = line_space / 2
        left_index -= 1

    # Floating point tolerance
    tolerance = 1e-6

    # This is an overapproximation
    beam_count = 0
    while current_steering_angle < (steering_angle - tolerance):
        left_index -= 1
        right_index += 1
        current_steering_angle += line_space

    # Get the corrected steering angle
    steering_angle_corrected_vector = np.array(vector[left_index:right_index+1])

    # Updates are here when selecting angles (Each beam represents a 2 degree beam)
    angle_between_vectors = 4
    idx = np.linspace(0, total_lines-1, num=total_lines).astype(int) * (angle_between_vectors/2)
    idx = idx - idx[int(len(idx)/2)]
    if len(idx) % 2 == 0:
        idx += 1
    idx = idx + int(len(steering_angle_corrected_vector)/2)
    idx = idx.astype(int)
    final_vector = steering_angle_corrected_vector[idx]

    return final_vector

def vector_conversion(vector, steering_angle, max_distance, total_lines, original_total_lines):

    # Fix the vector to have to correct max_distance
    vector = np.clip(vector, 0, max_distance)

    # Get how many degrees between each line
    line_space = (steering_angle * 2) / float(original_total_lines - 1)

    # Get the starting lines angle
    left_index = int(len(vector) / 2)
    right_index = int(len(vector) / 2)
    current_steering_angle = 0
    if (original_total_lines % 2) == 0: 
        current_steering_angle = line_space / 2
        left_index -= 1

    # Floating point tolerance
    tolerance = 1e-6

    # This is an overapproximation
    while current_steering_angle < (steering_angle - tolerance):
        left_index -= 1
        right_index += 1
        current_steering_angle += line_space

    # Get the corrected steering angle
    steering_angle_corrected_vector = vector[left_index:right_index+1]
    idx = np.round(np.linspace(0, len(steering_angle_corrected_vector) - 1, total_lines)).astype(int)
    final_vector = steering_angle_corrected_vector[idx]

    return final_vector

def centralBeamWithExponentialSampling(vector):

    # Get the vector length
    vec_len = len(vector)
    converted_vector = np.zeros(vec_len)

    # Get the predefined point distribution
    predfined_point_distribution = {
        2:  np.array([5,10]),
        3:  np.array([5, 6, 10]),
        4:  np.array([5, 6, 7, 10]),
        5:  np.array([5, 6, 7, 8, 10]),
        6:  np.array([5, 6, 7, 8, 9, 10]),
        7:  np.array([5, 6, 7, 8, 9, 10, 11]),
        8:  np.array([5, 6, 7, 8, 9, 10, 11, 12]),
        9:  np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]),
        10: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        11: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        12: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        13: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        14: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    }

    sample_distribution = {
        1:  np.array([14]),
        2:  np.array([4, 4]),
        3:  np.array([3, 3, 3]),
        4:  np.array([2, 3, 3, 2]),
        5:  np.array([2, 2, 2, 2, 2]),
        6:  np.array([2, 2, 2, 2, 2, 2]),
        7:  np.array([2, 2, 2, 2, 2, 2, 2]),
        8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
        9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
        10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
    }

    # Load the current distribution
    current_distribution = sample_distribution[vec_len]

    # Process the vector
    for i, v in enumerate(vector):
        num_samples = current_distribution[i]
        predefined_points = predfined_point_distribution[num_samples]
        distance = np.absolute(predefined_points - v)
        index = np.argmin(distance)
        converted_vector[i] = predefined_points[index]

    # Print the converted vector
    return converted_vector  

def centalBeamSampling(vector):
    # Get the vector length
    vec_len = len(vector)
    converted_vector = np.zeros(vec_len)

    # Save the distribution
    distribution = {
        1:  [6],
        2:  [6, 6],
        3:  [4, 10, 4],
        4:  [4, 8, 8, 4],
        5:  [3, 7, 10, 7, 3],
        6:  [3, 7, 8, 8, 7, 3],
        7:  [3, 5, 7, 12, 7, 5, 3],
        8:  [3, 5, 7, 9, 9, 7, 5, 3],
        9:  [3, 4, 6, 9, 10, 9, 6, 4, 3],
        10: [3, 4, 6, 8, 9, 9, 8, 6, 4, 3],
    }

    # Load the current distribution
    current_distribution = distribution[vec_len]

    # Process the vector
    for i, v in enumerate(vector):
        num_samples = current_distribution[i]
        accuracy = 30 / (num_samples)
        final_v = np.round(np.array(v, dtype=float) / accuracy) * accuracy
        final_v = max(accuracy, final_v)
        converted_vector[i] = final_v

    return converted_vector

def exponentialSampling(vector):
    # Predefine the points
    converted_vector = np.zeros(len(vector))
    predefined_points = np.array([5, 6, 7, 8, 9, 10])
    # predefined_points = np.array([5, 10, 15, 20, 25, 30])
    for i, v in enumerate(vector):
        distance = np.absolute(predefined_points - v)
        index = np.argmin(distance)
        converted_vector[i] = predefined_points[index]
    return converted_vector

def getStep(vector, accuracy):
    # Numpy rounds even numbers down and odd numbers up (make it always round down)
    converted_vector = np.round((np.array(vector, dtype=float)  - 1e-6) / accuracy) * accuracy
    # The minimum should always be > 0
    converted_vector[converted_vector <= 0] = accuracy
    return converted_vector

def processFile(file_name, total_vectors, vector_size, new_steering_angle, new_max_distance, new_total_lines, new_accuracy, max_possible_crashes, base, ignore_crashes=False):
    # Open the file
    f = open(file_name, "r")  
    
    test_vectors        = np.full((total_vectors, vector_size), np.inf, dtype='float64')
    collision_counter   = 0
    simulation_time     = ""
    vehicle_count       = -1
    current_vector      = 0
    incident_hashes     = np.full((1, max_possible_crashes), np.inf, dtype='float64')

    ego_velocities      = np.full((total_vectors, 3), np.inf, dtype='float64')
    ego_positions       = np.full((total_vectors, 3), np.inf, dtype='float64')

    crash_incident_angles   = []
    crash_ego_magnitudes    = []
    crash_veh_magnitudes    = []

    stall_information   = np.full((total_vectors, 3), np.inf, dtype='float64')

    debug_count = 0
    for line in f: 
        # Make sure we aren't writing too many lines
        assert(current_vector <= total_vectors)

        # Get the number of external vehicles
        if "External Vehicles: " in line:
            vehicle_count = int(line[line.find(": ")+2:])

        # Get each of the vectors
        if "Vector: " in line:
            vector_str = line[line.find(": ")+3:-2]
            vector = np.fromstring(vector_str, dtype=float, sep=', ')
            v = np.fromstring(vector_str, dtype=float, sep=',')
            # Compute how many values are maxed out
            tolerance = 1e-6
            total_values_at_max = (v >= new_max_distance - tolerance).sum()
            if total_values_at_max == np.max:
                print("here")
            # Get the closest obstacle for the stall data
            closest_index = np.argmin(v)
            stall_information[current_vector][0] = closest_index
            stall_information[current_vector][1] = v[closest_index]
            stall_information[current_vector][2] = total_values_at_max

            # Original method for seperating the vectors
            # vector = vector_conversion(vector, new_steering_angle, new_max_distance, new_total_lines, len(vector))
            # vector = vector_conversion_centralized(vector, new_steering_angle, new_max_distance, new_total_lines, len(vector))
            vector = vector_conversion_distribution(vector, new_steering_angle, new_max_distance, new_total_lines, len(vector))
            
            # Original method of converting to correct granularity
            # vector = getStep(vector, new_accuracy)
            # vector = exponentialSampling(vector)
            # vector = centalBeamSampling(vector)
            vector = centralBeamWithExponentialSampling(vector)
            test_vectors[current_vector] = vector
            current_vector += 1

            debug_count += 1
            if debug_count % 15 == 0:
                pass
                # plot_debugging_centralized(v, vector, new_steering_angle)
                # plot_debugging(v, vector, new_steering_angle)

        if "Ego Position: " in line:
            ego_position = np.fromstring(line[15:-1], dtype=np.float, sep=' ')
            
            # If the ego_positions are only 2 long its highway
            if len(ego_position) == 2:
                ego_position = np.pad(ego_position, (0, 1), 'constant')

            ego_positions[current_vector-1] = ego_position

        if "Ego Velocity: " in line:          
            ego_velocity = np.fromstring(line[15:-1], dtype=np.float, sep=' ')

            # If the ego_positions are only 2 long its highway
            if len(ego_velocity) == 2:
                ego_velocity = np.pad(ego_velocity, (0, 1), 'constant')

            ego_velocities[current_vector-1] = ego_velocity

        if "Crash: True" in line:
            if not ignore_crashes:
                # File the rest of the test vector up with np.nan
                while current_vector < test_vectors.shape[0]:
                    test_vectors[current_vector] = np.full(test_vectors.shape[1], np.nan, dtype='float64')
                    current_vector += 1

        # Look for collisions
        if "Collided: True" in line:
            collision_counter += 1

        if "Total Simulated Time:" in line:
            simulation_time = line[line.find(": ")+2:]

        # Get the crash details
        if "Ego velocity magnitude: " in line:
            ego_vel_data = float(line[line.find(": ")+2:])
            crash_ego_magnitudes.append(ego_vel_data)

        if "Incident vehicle velocity magnitude: " in line:
            ind_vel_data = float(line[line.find(": ")+2:])
            crash_veh_magnitudes.append(ind_vel_data)

        if "Angle of incident: " in line:
            ang_data = float(line[line.find(": ")+2:])
            crash_incident_angles.append(ang_data)

    # Print warning
    if collision_counter > max_possible_crashes:
        print("Warning: More collisions found than max possible crashes, truncating crashes")

    # If there was a crash compute the incident hash
    for i in range(min(collision_counter, max_possible_crashes)):
        # Get the incident data
        ego_magnitude = crash_ego_magnitudes[i]
        veh_magnitude = crash_veh_magnitudes[i]
        incident_angle = crash_incident_angles[i]
        current_hash = hash_crash(ego_magnitude, veh_magnitude, incident_angle, base=base)
        incident_hashes[0, i] = current_hash

    # Convert the simulated time to a float
    simulation_time = float(simulation_time)

    # Close the file
    f.close()

    return vehicle_count, collision_counter, test_vectors, simulation_time, incident_hashes, ego_positions, ego_velocities, stall_information

def processFileFeasibility(f, new_steering_angle, new_max_distance, new_total_lines, new_accuracy):
    test_vectors    = []
    crash           = False
    vehicle_count   = -1
    current_vector  = 0
    for line in f: 
        # Get the number of external vehicles
        if "External Vehicles: " in line:
            vehicle_count = int(line[line.find(": ")+2:])
        # Get each of the vectors
        if "Vector: " in line:
            vector_str = line[line.find(": ")+3:-2]
            vector = np.fromstring(vector_str, dtype=float, sep=', ')
            vector = getStep(vector, new_accuracy)
            test_vectors.append(np.array(vector))
            current_vector += 1
        # Look for crashes
        if "Crash: True" in line:
            crash = True

    test_vectors = np.array(test_vectors)

    return vehicle_count, crash, test_vectors

def countVectorsInFile(f):
    vector_count = 0
    crash = False
    for line in f: 
        if "Vector: " in line:
            vector_count += 1
        if "Crash: True" in line:
            crash = True
    return vector_count, crash

def plot_debugging_centralized(original_vectors, new_vectors, new_steering_angle):
    original_lines = []
    new_lines = []
    original_line_spacing = (new_steering_angle * 2) / float(len(original_vectors) - 1)
    new_line_spacing = 4

    print("Original line spacing: {}".format(original_line_spacing))
    print("New line spacing: {}".format(new_line_spacing))

    # Compute the start and end points for the original set of data
    current_angle = -1 * new_steering_angle
    for i in original_vectors:
        starting_point = (0, 0)
        x = i * math.cos(math.radians(current_angle))
        y = i * math.sin(math.radians(current_angle))
        end_point = (x, y)
        current_angle += original_line_spacing
        original_lines.append([starting_point, end_point])

    # Compute the start and end points for the new set of data
    current_angle = -1 * int(len(new_vectors) / 2) * 4
    if len(new_vectors) % 2 == 0:
        current_angle += 2
    for i in new_vectors:
        starting_point = (0, 0)
        x = i * math.cos(math.radians(current_angle))
        y = i * math.sin(math.radians(current_angle))
        end_point = (x, y)
        current_angle += new_line_spacing
        new_lines.append([starting_point, end_point])

    # Debugging
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16,7))
    for line in original_lines:
        x = [line[0][0], line[1][0]]
        y = [line[0][1], line[1][1]]
        ax1.plot(x,y,c="green")

    for line in new_lines:
        x = [line[0][0], line[1][0]]
        y = [line[0][1], line[1][1]]
        ax2.plot(x,y,c="green")

    ax1.set_title("Original RSR 30")
    ax1.set_xlim([-5,35])
    ax1.set_ylim([-20,20])
    ax2.set_xlim([-5,35])
    ax2.set_ylim([-20,20])
    ax2.set_title("Original RSR 5")
    plt.show()

def plot_debugging(original_vectors, new_vectors, new_steering_angle):
    original_lines = []
    new_lines = []
    original_line_spacing = (new_steering_angle * 2) / float(len(original_vectors) - 1)
    new_line_spacing = (new_steering_angle * 2) / float(len(new_vectors) - 1)

    print("Original line spacing: {}".format(original_line_spacing))
    print("New line spacing: {}".format(new_line_spacing))

    # Compute the start and end points for the original set of data
    current_angle = -1 * new_steering_angle
    for i in original_vectors:
        starting_point = (0, 0)
        x = i * math.cos(math.radians(current_angle))
        y = i * math.sin(math.radians(current_angle))
        end_point = (x, y)
        current_angle += original_line_spacing
        original_lines.append([starting_point, end_point])

    # Compute the start and end points for the new set of data
    current_angle = -1 * new_steering_angle
    for i in new_vectors:
        starting_point = (0, 0)
        x = i * math.cos(math.radians(current_angle))
        y = i * math.sin(math.radians(current_angle))
        end_point = (x, y)
        current_angle += new_line_spacing
        new_lines.append([starting_point, end_point])

    # Debugging
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16,7))
    for line in original_lines:
        x = [line[0][0], line[1][0]]
        y = [line[0][1], line[1][1]]
        ax1.plot(x,y,c="green")

    for line in new_lines:
        x = [line[0][0], line[1][0]]
        y = [line[0][1], line[1][1]]
        ax2.plot(x,y,c="green")

    ax1.set_title("Original RSR 30")
    ax1.set_xlim([-5,35])
    ax1.set_ylim([-20,20])
    ax2.set_xlim([-5,35])
    ax2.set_ylim([-20,20])
    ax2.set_title("Original RSR 5")
    plt.show()