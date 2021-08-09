
import os
import re
import glob
import math
import random 
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics


def string_to_vector(vector_string):
    vector_str = vector_string[vector_string.find(": ")+3:-2]
    vector = np.fromstring(vector_str, dtype=float, sep=', ')
    return vector

def vector_conversion(vector, steering_angle, max_distance, total_lines, original_total_lines):

    # Get the steering angle and original max distance
    HK = HighwayKinematics()
    original_steering_angle = HK.steering_angle
    original_max_distance = HK.max_velocity

    # Fix the vector to have to correct max_distance
    vector = np.clip(vector, 0, max_distance)

    # Get how many degrees between each line
    line_space = (original_steering_angle * 2) / float(original_total_lines - 1)

    # Get the starting lines angle
    left_index = int(len(vector) / 2)
    right_index = int(len(vector) / 2)
    current_steering_angle = 0
    if (original_total_lines % 2) == 0: 
        current_steering_angle = line_space / 2
        left_index -= 1

    # This is an overapproximation
    while current_steering_angle < steering_angle:
        left_index -= 1
        right_index += 1
        current_steering_angle += line_space

    # Get the corrected steering angle
    steering_angle_corrected_vector = vector[left_index:right_index+1]

    # Select the correct number of lines
    if len(steering_angle_corrected_vector) < total_lines:
        pass
        # print("Requested more lines than we have, extrapolating")
    idx = np.round(np.linspace(0, len(steering_angle_corrected_vector) - 1, total_lines)).astype(int)
    final_vector = steering_angle_corrected_vector[idx]

    return final_vector

def getStep(vector, accuracy):
    converted_vector = np.round(np.array(vector, dtype=float) / accuracy) * accuracy
    # The minimum should always be > 0
    converted_vector[converted_vector <= 0] = accuracy
    return converted_vector

def countVectorsInFile(f):
    vector_count = 0
    crash = False
    for line in f: 
        if "Vector: " in line:
            vector_count += 1
        if "Crash: True" in line:
            crash = True
    return vector_count, crash

def processFile(f, total_vectors, vector_size, new_steering_angle, new_max_distance, new_total_lines, new_accuracy):
    test_vectors    = np.full((total_vectors, vector_size), np.inf, dtype='float64')
    crash           = False
    simulation_time = ""
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
            vector = vector_conversion(vector, new_steering_angle, new_max_distance, new_total_lines, len(vector))
            vector = getStep(vector, new_accuracy)
            test_vectors[current_vector] = vector
            current_vector += 1
        # Look for crashes
        if "Crash: True" in line:
            crash = True
            # File the rest of the test vector up with np.nan
            while current_vector < test_vectors.shape[0]:
                test_vectors[current_vector] = np.full(test_vectors.shape[1], np.nan, dtype='float64')
                current_vector += 1
        if "Total Simulated Time:" in line:
            simulation_time = line[line.find(": ")+3:]

    # Convert the simulated time to a float            
    simulation_time = float(simulation_time)

    return vehicle_count, crash, test_vectors, simulation_time

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