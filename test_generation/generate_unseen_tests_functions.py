import os
import sys
import glob
import copy
import math
import time
import pickle

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from tqdm import tqdm
from celluloid import Camera
from datetime import datetime
from general.reachset import ReachableSet
from pandas.plotting import parallel_coordinates
from shapely.geometry import Polygon, LineString, Point

def compute_coverage(load_name, return_dict, return_key, base_path, new_max_distance, new_accuracy):
    
    # Get the current time
    start=datetime.now()
    total_beams = load_name[load_name.find("_")+1:]
    total_beams = total_beams[total_beams.find("_")+1:]
    total_beams = total_beams[total_beams.find("_b")+2:]
    total_beams = total_beams[0:total_beams.find("_d")]
    total_beams = int(total_beams)

    # Compute total possible values using the above
    unique_observations_per_cell = (new_max_distance / float(new_accuracy))
    total_possible_observations = int(pow(unique_observations_per_cell, total_beams))

    print("Processing: " + load_name)
    traces = np.load(base_path + "traces" + load_name)
    vehicles = np.load(base_path + "vehicles" + load_name)

    # Sort the data based on the number of vehicles per test
    vehicles_indices = vehicles.argsort()
    traces = traces[vehicles_indices]
    vehicles = vehicles[vehicles_indices]

    total_traces = traces.shape[0]
    total_crashes = 0
    total_vectors = 0

    unique_vectors_seen                 = set()
    accumulative_graph                  = np.full(total_traces, np.nan)
    acuumulative_graph_vehicle_count    = np.full(total_traces, np.nan)

    # For each of the traces
    for i in tqdm(range(total_traces), position=int(return_key[1:]), mininterval=5):
        # Get the trace
        trace = traces[i]
        vehicle_count = vehicles[i]
        
        # See if there was a crash
        if np.isnan(trace).any():
            total_crashes += 1

        # For each vector in the trace
        for vector in trace:
            # If this vector contains nan it means it crashed (and so we can ignore it, this traces crash was already counted)
            if np.isnan(vector).any():
                continue

            # If this vector contains inf it means that the trace was extended to match sizes with the maximum sized trace
            if np.isinf(vector).any():
                continue

            # Count the traces
            total_vectors += 1
            
            # Check if it is unique
            unique_vectors_seen.add(tuple(vector))

        # Used for the accumulative graph
        unique_vector_length_count          = len(unique_vectors_seen)
        accumulative_graph[i]               = unique_vector_length_count
        acuumulative_graph_vehicle_count[i] = vehicle_count

    overall_coverage = round((unique_vector_length_count / float(total_possible_observations)) * 100, 4)
    crash_percentage = round(total_crashes / float(total_traces) * 100, 4)

    # Convert from a set to a numpy array
    unique_vectors_seen = np.array(list(unique_vectors_seen))

    print("\n\n\n\n\n\n")
    print("Filename:\t\t\t" + load_name)
    print("Total traces considered:\t" + str(total_vectors))
    print("Total crashes:\t\t\t" + str(total_crashes))
    print("Crash percentage:\t\t" + str(crash_percentage) + "%")
    print("Total vectors considered:\t" + str(total_vectors))
    print("Total unique vectors seen:\t" + str(unique_vector_length_count))
    print("Total possible vectors:\t\t" + str(total_possible_observations))
    print("Total coverage:\t\t\t" + str(overall_coverage) + "%")
    print("Total time to compute:\t\t" + str(datetime.now()-start))

    # Get all the unique number of external vehicles
    unique_vehicle_count = list(set(acuumulative_graph_vehicle_count))
    unique_vehicle_count.sort()

    # Convert to a percentage
    accumulative_graph_coverage = (accumulative_graph / total_possible_observations) * 100

    # Return both arrays
    return_dict[return_key] = [accumulative_graph_coverage, acuumulative_graph_vehicle_count, total_beams, unique_vectors_seen]
    return True

def get_trace_files(base_path):
   
    trace_file_names = glob.glob(base_path + "traces_*")

    file_names = []
    for f in trace_file_names:
        name = f.replace(base_path + "traces", "")
        file_names.append(name)

    # Sort the names
    file_names.sort()

    return file_names

def compute_unseen_vectors(return_dict, maximum_unseen_vectors, new_max_distance, unique_values_per_beam, new_accuracy):

    # Computing the unseen vectors
    final_data = {}
    for key in return_dict:
        accumulative_graph_coverage, acuumulative_graph_vehicle_count, total_beams, unique_vectors_seen = return_dict[key]
        print("Computing unseen vectors for beams: {}".format(total_beams))

        unique_vectors_seen_set = set()
        for v in unique_vectors_seen:
            unique_vectors_seen_set.add(tuple(v))

        # Check that the conversion went correctly
        if len(unique_vectors_seen_set) != unique_vectors_seen.shape[0]:
            print("Conversion from numpy array to set failed")
            exit() 

        # Use this to hold the value for iterating over all values
        current_vector = np.full(total_beams, unique_values_per_beam[0])
        complete = False

        # Compute maximum_unseen_vectors beams that have not yet been seen
        unseen_vectors = []
        while len(unseen_vectors) < maximum_unseen_vectors and not complete:
            
            # Check if we have seen this vector before
            length_before = len(unique_vectors_seen_set)
            unique_vectors_seen_set.add(tuple(current_vector))

            # If it was unique then it has not been seen before:
            if length_before != len(unique_vectors_seen_set):
                unseen_vectors.append(copy.deepcopy(current_vector))
                
            # Increment the current vector
            current_vector[0] += new_accuracy

            # Check that all the values are correct
            for i in range(total_beams - 1):
                if current_vector[i] > np.max(unique_values_per_beam):
                    current_vector[i+1] += new_accuracy
                    current_vector[i] = unique_values_per_beam[0]

            # Check if we are complete
            if current_vector[-1] > np.max(unique_values_per_beam):
                complete = True
        
        unseen_vectors = np.array(unseen_vectors)

        # Check if you have any unseen vectors
        try:
            height, width = np.shape(unseen_vectors)
            if height <= 0 or width <= 0:
                print("No unseen vectors")
                continue
        except:
            continue

        # Create the unseen dataframe
        plotting_data = {}
        for i in range(total_beams):
            plotting_data["beam" + str(i)] = unseen_vectors[:, i]
        plotting_data["Name"] = np.full(height, "unseen")

        print("Unseen data shape: {}".format(np.shape(unseen_vectors)))
        unseen = pd.DataFrame(plotting_data) 

        # Create the seen dataframe
        unique_vectors_seen = np.array(unique_vectors_seen, dtype="int")
        height, width = np.shape(unique_vectors_seen)
        seen = {}
        for i in range(total_beams):
            seen["beam" + str(i)] = unique_vectors_seen[:, i]
        seen["Name"] = np.full(height, "seen")
        seen = pd.DataFrame(seen) 

        # Create the final daya
        final_data[key] = {"seen": seen,
                           "unseen": unseen,
                           "total_beams": total_beams,
                           "max_distance": new_max_distance,
                           "steering_angle": new_max_distance}

    return final_data

def compute_reach_set_details(total_lines, max_distance, steering_angle, new_accuracy):

    # Create the ego vehicle
    ego_vehicle = Polygon([[-1, -0.5], [-1, 0.5], [0, 0.5], [0, -0.5]])

    # Create the reachable set
    reach = ReachableSet()
    r_set = reach.estimate_raw_reachset(total_lines=100, 
                                        steering_angle=max_distance,
                                        max_distance=steering_angle)

    # Create the exterior points
    exterior_points = []

    # Plot the arch of the exterior
    for i in range(len(r_set)):

        # Get the current lines boundary points
        p1, p2 = r_set[i].boundary
        new_points = [p1, p2]

        # Check if we already have these points
        for p in new_points:
            already_included = False
            for p3 in exterior_points:
                if p3 == p:
                    already_included = True

            if not already_included:
                exterior_points.append(p)

    # Create the exterior boundary
    exterior_r_set = Polygon(exterior_points)

    # Estimate the reachset
    beams = reach.estimate_raw_reachset(total_lines=total_lines, 
                                    steering_angle=max_distance,
                                    max_distance=steering_angle)

    # Break the lines up into points of significance
    segmented_lines = reach.line_to_points(beams, new_accuracy)

    return ego_vehicle, r_set, exterior_r_set, beams, segmented_lines

def create_image_representation(figure_key, data, distance_tracker=None):

    data = np.transpose(data)

    # Create the figure
    plt.figure(str(figure_key))

    # Check if we want to plot distance as well
    if distance_tracker is None:
        plt.imshow(data, cmap='gray', aspect='auto')
    else:
        plt.subplot(211)
        plt.imshow(data, cmap='gray', aspect='auto')
        plt.subplot(212)
        plt.plot(distance_tracker)
        plt.ylim(0, max(distance_tracker))
        plt.margins(x=0)

    return plt

def save_unseen_data_to_file(data, distance_tracker, segmented_lines, new_accuracy, total_samples, beams):

    # Start each test with the max value vectors
    init_position = []
    for i in range(beams):
        index = int(len(segmented_lines[i]) / 2)
        # Get the physical point
        new_point = segmented_lines[i][index]
        x, y = new_point.xy
        new_data = np.array([x[0], y[0]])
        init_position.append(new_data)

    # Init variables
    final_points = []
    final_indices = []
    counter = 0 
    vectors_per_test = 0

    # Set some config
    min_vectors_per_test = 5
    max_vectors_per_test = 10

    # Start the final points with the init data
    final_points.append(init_position)

    if not os.path.exists('../output/generated_tests/tests_merged/{}/{}_beams'.format(total_samples, beams)):
        os.makedirs('../output/generated_tests/tests_merged/{}/{}_beams'.format(total_samples, beams))

    for i in range(len(data)):
        row = data[i]

        # Save the data
        if ((distance_tracker[i] > 5) and (vectors_per_test > min_vectors_per_test)) or (vectors_per_test >= max_vectors_per_test):
            np.save("../output/generated_tests/tests_merged/{}/{}_beams/test{}_points.npy".format(total_samples, beams, counter), final_points)
            np.save("../output/generated_tests/tests_merged/{}/{}_beams/test{}_index.npy".format(total_samples, beams, counter), final_indices)
            final_points = []
            # Start the final points with the init data
            final_points.append(init_position)
            counter += 1
            vectors_per_test = 0

        physical_row = []
        # For each of the elements, convert it into a physical point
        for i in range(len(row)):
            # Get the element
            item = row[i]
            # Convert that element to the right index
            index = int(item / new_accuracy)
            # Get the physical point
            new_point = segmented_lines[i][index]
            x, y = new_point.xy
            new_data = np.array([x[0], y[0]])
            physical_row.append(new_data)

        # print(physical_row)
        final_indices.append(row)
        final_points.append(physical_row)
        vectors_per_test += 1

    # Save the final test
    np.save("../output/generated_tests/tests_merged/{}/{}_beams/test{}_points.npy".format(total_samples, beams, counter), final_points)
    np.save("../output/generated_tests/tests_merged/{}/{}_beams/test{}_index.npy".format(total_samples, beams, counter), final_indices)

    return counter + 1

def save_unseen_data_to_file_single(data, segmented_lines, new_accuracy, total_samples, beams):

    # Start each test with the max value vectors
    init_position = []
    for i in range(beams):
        index = int(len(segmented_lines[i]) / 2)
        # Get the physical point
        new_point = segmented_lines[i][index]
        x, y = new_point.xy
        new_data = np.array([x[0], y[0]])
        init_position.append(new_data)

    # Init variables
    final_points = []
    final_indices = []
    counter = 0 
    min_vectors_per_test = 0

    # Start the final points with the init data
    final_points.append(init_position)
    final_indices.append(np.full(beams, index * new_accuracy))

    if not os.path.exists('../output/generated_tests/tests_single/{}/{}_beams'.format(total_samples, beams)):
        os.makedirs('../output/generated_tests/tests_single/{}/{}_beams'.format(total_samples, beams))

    for i in range(len(data)):
        row = data[i]
        physical_row = []
        # For each of the elements, convert it into a physical point
        for i in range(len(row)):
            # Get the element
            item = row[i]
            # Convert that element to the right index
            index = int(item / new_accuracy)
            # Get the physical point
            new_point = segmented_lines[i][index]
            x, y = new_point.xy
            new_data = np.array([x[0], y[0]])
            physical_row.append(new_data)

        final_indices.append(row)
        final_points.append(physical_row)
        np.save("../output/generated_tests/tests_single/{}/{}_beams/test{}_points.npy".format(total_samples, beams, counter), final_points)
        np.save("../output/generated_tests/tests_single/{}/{}_beams/test{}_index.npy".format(total_samples, beams, counter), final_indices)
        final_points = []
        # Start the final points with the init data
        final_points.append(init_position)
        counter += 1

    return counter