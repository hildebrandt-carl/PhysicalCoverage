
import os
import re
import sys
import glob
import math
import random 
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from general.failure_oracle import hash_crash
from general.failure_oracle import hash_stall

from general.environment_configurations import RSRConfig
from general.environment_configurations import HighwayKinematics

def vector_conversion(vector, steering_angle, max_distance, total_lines, original_total_lines, distribution):

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
    # Beam 30 ends at 30 degrees
    angle_distribution = distribution.get_angle_distribution()

    current_distribution = angle_distribution[total_lines]
    # Compute the indexes
    l = original_total_lines - 1
    idx = np.round(((np.array(current_distribution) + 30) / (60/l)),0)
    idx = np.clip(idx, 0, l).astype(int)
    final_vector = steering_angle_corrected_vector[idx]
    return final_vector

def sample_vector(vector, distribution):

    # Get the vector length
    vec_len = len(vector)
    converted_vector = np.zeros(vec_len)

    # Get the predefined point distribution
    predefined_point_distribution = distribution.get_predefined_points()
    sample_distribution = distribution.get_sample_distribution()

    # Load the current distribution
    current_distribution = sample_distribution[vec_len]

    # Process the vector
    for i, v in enumerate(vector):
        num_samples = current_distribution[i]
        predefined_points = predefined_point_distribution[num_samples]
        distance = np.absolute(predefined_points - v)
        index = np.argmin(distance)
        converted_vector[i] = predefined_points[index]

    # Print the converted vector
    return converted_vector  

def processFile(file_name, total_vectors, vector_size, new_steering_angle, new_max_distance, new_total_lines, max_possible_crashes, max_possible_stalls, base, distribution, ignore_crashes=False):
    
    # Open the file
    f = open(file_name, "r")  
    
    test_vectors        = np.full((total_vectors, vector_size), None, dtype='float64')
    collision_counter   = 0
    simulation_time     = ""
    vehicle_count       = -1
    current_vector      = 0
    crash_hashes        = np.full((1, max_possible_crashes), None, dtype='object')
    stall_hashes        = np.full((1, max_possible_stalls), None, dtype='object')

    ego_velocities      = np.full((total_vectors, 3), None, dtype='float64')
    ego_positions       = np.full((total_vectors, 3), None, dtype='float64')

    crash_incident_angles   = []
    crash_ego_magnitudes    = []
    crash_veh_magnitudes    = []

    stall_information   = np.full((total_vectors, 3), None, dtype='float64')
    currently_stalling = False
    stall_location = None
    car_has_started_driving = False

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

            # Seperating the vectors
            vector = vector_conversion(vector, new_steering_angle, new_max_distance, new_total_lines, len(vector), distribution)
            
            # Converting to correct granularity
            vector = sample_vector(vector, distribution)
            test_vectors[current_vector] = vector

            debug_count += 1
            if debug_count % 15 == 0:
                pass
                # plot_debugging_centralized(v, vector, new_steering_angle)
                # plot_debugging(v, vector, new_steering_angle)

            # Determine the angle between each vector
            angle_between_beams = (new_steering_angle * 2) / len(v)

            # Get the closest obstacle for the stall data
            closest_index = np.argmin(v)

            # Compute the largest gap moving forward
            tolerance = 1e-6
            gap_angle = v >= new_max_distance - tolerance
            largest_gap_array = np.diff(np.where(np.concatenate(([gap_angle[0]], gap_angle[:-1] != gap_angle[1:], [True])))[0])[::2]
            if len(largest_gap_array) >= 1:
                largest_gap = np.max(largest_gap_array)
            else:
                largest_gap = 0 

            # Save this information for processing later
            stall_information[current_vector][0] = closest_index
            stall_information[current_vector][1] = v[closest_index]
            stall_information[current_vector][2] = largest_gap * angle_between_beams

            # Increment the vector
            current_vector += 1

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
        crash_hashes[0, i] = current_hash

    # Convert the simulated time to a float
    simulation_time = float(simulation_time)

    # Determine the number of stalls
    stall_counter = 0
    for i, vel in enumerate(ego_velocities):

        current_velocity = np.sum(abs(vel))

        # Check if the car has started driving or not
        if current_velocity > 5:
            car_has_started_driving = True

        # If we have stopped moving and the gap is greater than 15 degrees
        stopped_moving = current_velocity < 0.01
        can_move_forward = stall_information[i][2] > 15

        # Check if we are stalling
        if stopped_moving and can_move_forward and not currently_stalling and car_has_started_driving:
            stall_hashes[0, stall_counter] = hash_stall(stall_information[i][0], stall_information[i][1], base=base)
            stall_counter += 1
            currently_stalling = True
            stall_position = ego_positions[i]

        # If we are currently stalling check to see if we have un-stalled
        if currently_stalling:
            current_position = ego_positions[i]
            distance_between_stalls = np.linalg.norm(current_position - stall_position)
            # After having driven 5m we can stall again
            if distance_between_stalls >= 5:
                currently_stalling = False

    # Close the file
    f.close()

    return vehicle_count, collision_counter, test_vectors, simulation_time, crash_hashes, stall_hashes, ego_positions, ego_velocities, file_name

def processFileFeasibility(f, new_steering_angle, new_max_distance, new_total_lines, distribution):
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
            # Seperating the vectors
            vector = vector_conversion(vector, new_steering_angle, new_max_distance, new_total_lines, len(vector), distribution)
            # Converting to correct granularity
            vector = sample_vector(vector, distribution)
            test_vectors.append(np.array(vector))
            current_vector += 1
        # Look for crashes
        if "Crash: True" in line:
            crash = True

    test_vectors = np.array(test_vectors)

    return vehicle_count, crash, test_vectors

def getFeasibleVectors(test_vectors, new_total_lines, distribution):

    threshold = 1e-6

    # Get the predefined points and sample distribution
    predefined_point_distribution = distribution.get_predefined_points()
    sample_distribution = distribution.get_sample_distribution()

    # Get the number of samples allowed for each reading
    number_samples_allowed = sample_distribution[new_total_lines]

    # Init
    feasible_vectors_set = set()
    all_vectors_set = set()

    # Compute all vectors
    possible_cell_values = []
    for i in range(len(test_vectors[0])):
        sample_points_allowed = predefined_point_distribution[number_samples_allowed[i]]
        possible_cell_values.append(sample_points_allowed)

    # Create all possible variations of the array
    all_combinations = list(itertools.product(*possible_cell_values))

    # Save these
    for combination in all_combinations:
        all_vectors_set.add(tuple(combination))

    # For each vector
    for v in test_vectors:
        # This holds what each of the different cells could contain
        possible_cell_values = []
        # Loop through each of the individual readings (i index) )(r reading)
        for i, r in enumerate(v):
            # Determine the possible sample points
            sample_points_allowed = predefined_point_distribution[number_samples_allowed[i]]
            # Only get sample points smaller than the current reading
            possible_values = sample_points_allowed[np.where(sample_points_allowed <= (r + threshold))]
            possible_cell_values.append(possible_values)

        # Create all possible variations of the array
        all_combinations = list(itertools.product(*possible_cell_values))

        # Save these
        for combination in all_combinations:
            feasible_vectors_set.add(tuple(combination))

    # Turn to a list of lists
    all_vectors = []
    for v in all_vectors_set:
        all_vectors.append(list(v))

    feasible_vectors = []
    for v in feasible_vectors_set:
        feasible_vectors.append(list(v))

    return all_vectors, feasible_vectors

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