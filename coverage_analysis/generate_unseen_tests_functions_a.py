import os
import glob
import copy
import math
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from tqdm import tqdm
from celluloid import Camera
from datetime import datetime
from reachset import ReachableSet
from pandas.plotting import parallel_coordinates
from shapely.geometry import Polygon, LineString, Point


def isUnique(vector, unique_vectors_seen):
    # Return false if the vector contains Nan
    if np.isnan(vector).any():
        return False
    if np.isinf(vector).any():
        return False
    # Assume True
    unique = True
    for v2 in unique_vectors_seen:
        # If we have seen this vector break out of this loop
        if np.array_equal(vector, v2):
            unique = False
            break
    return unique

def compute_coverage(load_name, return_dict, return_key, base_path, new_max_distance, new_accuracy):
    
    # Get the current time
    start=datetime.now()
    total_beams = load_name[load_name.find("_")+1:]
    total_beams = total_beams[total_beams.find("_")+1:]
    total_beams = total_beams[total_beams.find("_b")+2:]
    total_beams = total_beams[0:total_beams.find("_d")]
    total_beams = int(total_beams)

    # Compute total possible values using the above
    unique_observations_per_cell = (new_max_distance / float(new_accuracy)) + 1.0
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

    unique_vectors_seen                 = []
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
        for v in trace:
            # If this vector does not have any nan
            if not np.isnan(v).any():
                # Count it
                total_vectors += 1
                # Check if it is unique
                unique = isUnique(v, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(v)

        # Used for the accumulative graph
        unique_vector_length_count          = len(unique_vectors_seen)
        accumulative_graph[i]               = unique_vector_length_count
        acuumulative_graph_vehicle_count[i] = vehicle_count

    overall_coverage = round((unique_vector_length_count / float(total_possible_observations)) * 100, 4)
    crash_percentage = round(total_crashes / float(total_traces) * 100, 4)

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

def get_trace_files(base_path, scenario, load_name):
   
    trace_file_names = glob.glob(base_path + "traces_" + scenario + load_name)

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
        # print("Unique Vectors: {}".format(unique_vectors_seen))

        unique_vectors_seen_copy = copy.deepcopy(unique_vectors_seen)

        # Use this to hold the value for iterating over all values
        current_vector = np.full(total_beams, unique_values_per_beam[0])
        complete = False

        # Compute maximum_unseen_vectors beams that have not yet been seen
        unseen_vectors = []
        while len(unseen_vectors) < maximum_unseen_vectors and not complete:
            # Check if we have seen this vector before
            unique = isUnique(current_vector, unique_vectors_seen_copy)
            if unique:
                a = copy.deepcopy(current_vector)
                unseen_vectors.append(a)
                unique_vectors_seen_copy.append(a)
                
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

def create_parallel_plot_reach(combined_data, seen_data, unseen_data, ego_vehicle, exterior_r_set, segmented_lines, beams, new_accuracy, key):

    # Create the final figure
    plt.figure(key + "reachset")

    # Plot the ego vehicle
    x,y = ego_vehicle.exterior.xy
    # plt.plot(x,y)

    # Plot the exterior boundary
    x,y = exterior_r_set.exterior.xy
    # plt.plot(x,y, color="blue")

    # # Plot the points of significance along the line
    # for points in segmented_lines:
    #     xs = [point.x for point in points]
    #     ys = [point.y for point in points]
    #     plt.scatter(xs, ys, color="black")

    # # Plot the reachable set
    # for i in range(len(beams)):
    #     # Get the polygon
    #     p = beams[i]
    #     x,y = p.xy
    #     # Get the color
    #     c = "black"
    #     # Plot
    #     plt.plot(x, y, color=c, alpha=0.1)

    # This will be used to hold the different line strings
    unseen_line_strings = []

    # Display the seen data
    for index, row in unseen_data.iterrows():

        # Compute which point is being referred to
        point_values = np.array(row.tolist()[0:-1]) / new_accuracy
        
        # Plot the data
        xs = []
        ys = []

        points_in_line_string = []
        for i in range(len(point_values)):
            beam_number = i
            value = int(point_values[i])
            point_of_interest = segmented_lines[beam_number][value]
            points_in_line_string.append(point_of_interest)

        # Create a linestring using the 
        l = LineString(points_in_line_string)
        unseen_line_strings.append(l)

        # c = "r"
        # a = 0.2
        # x,y = l.xy
        # plt.plot(x, y, color=c, alpha=a)
    
    return plt, unseen_line_strings, exterior_r_set, ego_vehicle, segmented_lines

def order_similar_events(unseen_line_strings):

    unseen_line_strings_ordered = []

    distance_tracker = []
    counter = 0
    current_index = 0
    while len(unseen_line_strings) > 1:

        counter += 1

        # Grab the first line
        l1 = unseen_line_strings.pop(current_index)
        unseen_line_strings_ordered.append(l1)
        
        # Used to keep track of the best substitute
        min_distance = math.inf
        best_l2 = None
        best_index = None

        # Figure out which line is the most similar to it
        l2_index = 0
        for l2 in unseen_line_strings:

            # Get the coordinates for each line
            l1_coords = l1.coords
            l2_coords = l2.coords

            # For each point in the lines
            line_distance = 0
            for i in range(len(l1_coords)):
                p1 = Point(l1_coords[i])
                p2 = Point(l2_coords[i])
                distance = p1.distance(p2)
                line_distance += distance

            if line_distance < min_distance:
                min_distance = line_distance
                best_l2 = l2
                best_index = l2_index

            # increment l2 index counter
            l2_index += 1

        # Keep track of the min distance
        distance_tracker.append(min_distance)

        # Update the current index to use the new line
        current_index = best_index

    return unseen_line_strings_ordered, distance_tracker

def order_similar_events_raw(unseen_data):

    unseen_plot_data = []

    # For each heading
    for index, row in unseen_data.iterrows():
        # Get each of the columns data as a list
        unseen = row.tolist()[0:-1]
        unseen = np.array(unseen)

        # Create the plot data
        unseen_plot_data.append(unseen)

    unseen_data_ordered = []
    distance_tracker = []
    counter = 0
    current_index = 0
    while len(unseen_plot_data) > 1:

        counter += 1

        # Grab the first line
        l1 = unseen_plot_data.pop(current_index)
        unseen_data_ordered.append(l1)
        
        # Used to keep track of the best substitute
        min_distance = math.inf
        best_l2 = None
        best_index = None

        # Figure out which line is the most similar to it
        l2_index = 0
        for l2 in unseen_plot_data:

            # For each point in the lines
            line_distance = 0
            for i in range(len(l1)):
                p1 = l1[i]
                p2 = l2[i]
                distance = abs(p1 - p2)
                line_distance += distance

            if line_distance < min_distance:
                min_distance = line_distance
                best_l2 = l2
                best_index = l2_index

            # increment l2 index counter
            l2_index += 1

        # Keep track of the min distance
        distance_tracker.append(min_distance)

        # Update the current index to use the new line
        current_index = best_index

    return unseen_data_ordered, distance_tracker

def create_image_representation(unseen_line_strings_ordered, distance_tracker, key):
    # Get the data
    ordered_data = np.array(unseen_line_strings_ordered)

    ordered_data = np.transpose(ordered_data)

    plt.figure(key + " image")
    plt.subplot(211)
    plt.imshow(ordered_data, cmap='gray', aspect='auto')
    plt.subplot(212)
    plt.plot(distance_tracker)
    plt.ylim(0, len(distance_tracker))
    plt.margins(x=0)

    return plt

def save_unseen_data_to_file(unseen_data_ordered, distance_tracker, segmented_lines, new_accuracy, total_samples, beams):

    final_points = []
    counter = 0 
    min_entries_per_test = 0

    if not os.path.exists('output/{}/tests_a/{}_beams'.format(total_samples, beams)):
        os.makedirs('output/{}/tests_a/{}_beams'.format(total_samples, beams))

    for i in range(len(unseen_data_ordered)):
        row = unseen_data_ordered[i]
        min_entries_per_test += 1

        # Save the data
        if (distance_tracker[i] > 5) and (min_entries_per_test > 5):
            np.save("output/{}/tests_a/{}_beams/test{}.npy".format(total_samples, beams, counter), final_points)
            final_points = []
            counter += 1
            min_entries_per_test = 0

        physical_row = []
        # For each of the elements, convert it into a physical point
        for i in range(len(row)):
            # Get the element
            item = row[i]
            # Convert that element to the right index
            item = int(item / new_accuracy)
            # Get the physical point
            new_point = segmented_lines[i][item]
            x, y = new_point.xy
            new_data = np.array([x[0], y[0]])
            physical_row.append(new_data)

        # print(physical_row)
        final_points.append(physical_row)

    final_points = np.array(final_points)