import sys
import glob
import math
import copy
import hashlib
import argparse
import itertools

import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from collections import Counter

from utils.file_functions import get_beam_number_from_file
from utils.file_functions import order_files_by_beam_number
from utils.environment_configurations import RRSConfig
from utils.environment_configurations import WaymoKinematics
from utils.RRS_distributions import linear_distribution
from utils.RRS_distributions import center_close_distribution
from utils.RRS_distributions import center_mid_distribution

from matplotlib_venn import venn3


def show_most_common_RRS(indices, distribution, scenario, title=""):

    # Get the distribution
    if distribution   == "linear":
        distribution  = linear_distribution(scenario)
    elif distribution == "center_close":
        distribution  = center_close_distribution(scenario)
    elif distribution == "center_mid":
        distribution  = center_mid_distribution(scenario)
    else:
        print("ERROR: Unknown distribution ({})".format(distribution))
        exit()

    # Create the subplot
    fig, axs = plt.subplots(3, 5, figsize=(19, 6))
    fig.canvas.manager.set_window_title('{} - {}'.format(indices, title)) 

    # Get angle information needed for plotting
    angles              = distribution.get_angle_distribution()
    samples             = distribution.get_sample_distribution()
    predefined_points   = distribution.get_predefined_points()
    assert(len(indices) == 3)

    # Go through the indices
    for i, index in enumerate(indices):
        # Get the current trace
        current_trace = random_traces[index, :, :]

        # Create a list of tuples that we can Counter
        rrs_list = []

        # Get the current RRS
        for rrs in current_trace:
            # Ignore nans
            if not np.isnan(rrs).any():
                # Add this to a tuple
                rrs_list.append(tuple(rrs))

        # Create the rrs_counter
        rrs_counter = Counter(rrs_list)

        # Get the 5 most common RRS
        for j, rrs in enumerate(rrs_counter.most_common()[0:5]):
            rrs_count = rrs[1]
            rrs_lengths = list(rrs[0])
            actual_points = []
            possible_points = []

            rrs_resolution = len(rrs_lengths)

            for k in range(len(rrs_lengths)):
                # Compute the actual point
                l = rrs_lengths[k]
                a = angles[rrs_resolution][k]
                p = [l, 0]
                p_r = rotate(origin=[0,0], point=p, angle=a)
                actual_points.append(p_r)

                # Compute the possible points
                possible_p = predefined_points[samples[rrs_resolution][k]]
                for l in possible_p:
                    a = angles[rrs_resolution][k]
                    p = [l, 0]
                    p_r = rotate(origin=[0,0], point=p, angle=a)
                    possible_points.append(p_r)

            # Turn the points into a numpy array       
            actual_points = np.array(actual_points)
            possible_points = np.array(possible_points)

            # Create the possible points
            axs[i, j].scatter(possible_points[:, 0], possible_points[:, 1], c="C4", alpha=0.95, s=10)

            # Create the plot
            for p in actual_points:
                x = [0, p[0]]
                y = [0, p[1]]
                axs[i, j].plot(x, y, c="C0")

            axs[i, j].set_xlim([-10, 10])
            axs[i, j].set_ylim([-1, 15])
            axs[i, j].set_title(rrs_count)

def show_data_from_index(indices, title="", camera=True):

    assert(len(indices) == 3)

    # Create the subplot
    fig, axs = plt.subplots(3, 5, figsize=(19, 6))
    fig.canvas.manager.set_window_title('{} - {}'.format(indices, title)) 

    # Get the index
    for i, index in enumerate(indices):
        # Get the camera folder
        folder = "{}/waymo/random_tests/physical_coverage/additional_data/scenario{:03d}".format(args.data_path, index)

        # Get all the pictures
        if camera:
            all_pictures = glob.glob("{}/camera_data/camera*.png".format(folder))
            all_pictures = sorted(all_pictures)
        else:
            all_pictures = glob.glob("{}/point_cloud_data/RRS/point_cloud*.png".format(folder))
            all_pictures = sorted(all_pictures)

        # Select pictures from a wide range of views
        selected_pictures =  np.linspace(0, len(all_pictures)-1, 5, dtype=int)

        # Plot the pictures
        for j, picture_number in enumerate(selected_pictures):
            img = mpimg.imread(all_pictures[picture_number])

            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1) 

def find_best_worst_combination(perms, start_point, end_point):

    # Look at the results
    most_similar_indices = np.full((TRACKING, 3), -1)
    least_similar_indices = np.full((TRACKING, 3), -1)
    max_duplicates = np.full(TRACKING, -np.inf)
    min_duplicates = np.full(TRACKING, np.inf)

    # Iterate over the permutations
    for i, p in enumerate(perms):
        if start_point <= i < end_point:
            r = compare_3_way(p[0], p[1], p[2])

            num_dups = r[0]
            if num_dups > max_duplicates[0]:
                # Adding
                max_duplicates              = np.roll(max_duplicates, -1)
                most_similar_indices        = np.roll(most_similar_indices, -1, axis=0)
                max_duplicates[-1]          = num_dups
                most_similar_indices[-1]    = np.array([r[1], r[2], r[3]])
                # Sorting
                sort_indices                = np.argsort(max_duplicates)
                max_duplicates              = max_duplicates[sort_indices]
                most_similar_indices        = most_similar_indices[sort_indices]
            if num_dups < min_duplicates[-1]:
                # Adding
                min_duplicates              = np.roll(min_duplicates, 1)
                least_similar_indices       = np.roll(least_similar_indices, 1, axis=0)
                min_duplicates[0]           = num_dups
                least_similar_indices[0]    = np.array([r[1], r[2], r[3]])
                # Sorting
                sort_indices                = np.argsort(min_duplicates)
                min_duplicates              = min_duplicates[sort_indices]
                least_similar_indices       = least_similar_indices[sort_indices]

        elif i >= end_point:
            break

    return [max_duplicates, most_similar_indices, min_duplicates, least_similar_indices]

def compare_3_way(i1, i2, i3):
    # Compute the number of duplicates over the middle
    test_a = random_traces_hashes[i1]
    test_b = random_traces_hashes[i2]
    test_c = random_traces_hashes[i3]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    total_duplicates = (3 * (abc)) + (2 * (ab + ac + bc))

    return total_duplicates, i1, i2, i3

def number_of_duplicates(list_a, list_b, list_c=None):
    # Get all the keys and a count of them
    count_a = Counter(list_a)
    count_b = Counter(list_b)

    if list_c is not None:
        count_c = Counter(list_c)
    
    
    # Get all common keys between both lists
    common_keys = set(count_a.keys()).intersection(count_b.keys())
    if list_c is not None:
        common_keys = common_keys.intersection(count_c.keys())

    # The count is the min number in both.
    # i.e. if list a has "200 A's, 50 B's" and list b has "100 A's 150 B's" then together they share "100 A's and 50 B's for a total of 150"
    if list_c is not None:
        result = sum(min(count_a[key], count_b[key], count_c[key]) for key in common_keys)
    else:
        result = sum(min(count_a[key], count_b[key]) for key in common_keys)

    return result

def plot_venn(indices, fig_name, fig_title):
    
    print("Selected scenarios: {}".format(indices))

    test_a = random_traces_hashes[indices[0]]
    test_b = random_traces_hashes[indices[1]]
    test_c = random_traces_hashes[indices[2]]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    a = len(test_a) - abc - ab - ac
    b = len(test_b) - abc - ab - bc
    c = len(test_c) - abc - ac - bc

    plt.figure("{}".format(fig_name))
    v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
    plt.title("{}".format(fig_title))

    for text in v.set_labels:
        text.set_fontsize(20)
    for x in range(len(v.subset_labels)):
        if v.subset_labels[x] is not None:
            v.subset_labels[x].set_fontsize(20)

    return


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",          help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",                                               help="waymo")
parser.add_argument('--tracking',           type=int, default=1,                                                help="Declare the number of indices we are tracking for best and worst")
parser.add_argument('--cores',              type=int, default=4,                                                help="Number of cores")
args = parser.parse_args()

# Create the configuration classes
WK = WaymoKinematics()
RRS = RRSConfig()

# Turn the tracking into a global variable
global TRACKING
TRACKING = args.tracking

# Save the kinematics and RRS parameters
if args.scenario == "waymo":
    new_steering_angle  = WK.steering_angle
    new_max_distance    = WK.max_velocity
else:
    print("ERROR: Unknown scenario")
    exit()

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Max velocity:\t\t" + str(new_max_distance))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = ""
load_name += "_s" + str(new_steering_angle) 
load_name += "_b" + str('*') 
load_name += "_d" + str(new_max_distance) 
load_name += "_t" + str(args.number_of_tests)
load_name += ".npy"

# Checking the distribution
if not (args.distribution == "linear" or args.distribution == "center_close" or args.distribution == "center_mid"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '/mnt/extradrive3/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
random_trace_file_names         = glob.glob(base_path + "traces_*")
random_trace_scenario_names     = glob.glob(base_path + "processed_files_*")

if (len(random_trace_file_names) <= 0) or (len(random_trace_scenario_names) < 0):
    print("files not found")
    exit()

# Get the RRS numbers
random_trace_RRS_numbers    = get_beam_number_from_file(random_trace_file_names)
random_scenario_RRS_numbers = get_beam_number_from_file(random_trace_scenario_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(random_trace_RRS_numbers) | set(random_scenario_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
random_trace_file_names         = order_files_by_beam_number(random_trace_file_names, RRS_numbers)
random_trace_scenario_names     = order_files_by_beam_number(random_trace_scenario_names, RRS_numbers)

# For each of the different beams
for RRS_index in range(len(RRS_numbers)):

    print("\n\n")
    print("------------------------------------------")
    print("------------Processing RRS: {}------------".format(RRS_numbers[RRS_index]))
    print("------------------------------------------")

    if not RRS_numbers[RRS_index] == 10:
        print("Skipping")
        continue

    # Get the beam number and files we are currently considering
    RRS_number              = RRS_numbers[RRS_index]
    random_trace_file       = random_trace_file_names[RRS_index]
    scenario_names_file     = random_trace_scenario_names[RRS_index]

    # Skip if any of the files are blank
    if random_trace_file == "" or scenario_names_file == "":
        print(random_trace_file)
        print(scenario_names_file)
        print("\scenario_names_file: Could not find one of the files for RRS number: {}".format(RRS_number))
        continue

    # Load the random_traces
    global random_traces
    random_traces = np.load(random_trace_file)

    # Load the random_traces
    scenario_names = np.load(scenario_names_file)

    # Sort so that the scenario indices match the scenario camera data
    scenario_order = scenario_names.argsort()
    scenario_names = scenario_names[scenario_order]
    random_traces = random_traces[scenario_order]

    # Holds the trace as a set of hashes  
    global random_traces_hashes
    random_traces_hashes = []

    print("\n")
    print("-------Converting Traces to Hashes--------")

    # Convert to hash value for easier comparison
    for test_number in tqdm(range(np.shape(random_traces)[0])):
        hashes = []
        current_trace = random_traces[test_number, :, :]
        for t in current_trace:
            # Ignore nans
            if not np.isnan(t).any():
                # Convert to a hash
                trace_string = str(t)
                hash = hashlib.md5(trace_string.encode()).hexdigest()
                hashes.append(hash)

        # Save the hashes
        random_traces_hashes.append(hashes)

    print("\n")
    print("-------Same vs Distinct Scenarios---------")

    print("Comparing 3 highway scenarios highway")
    highway_scenarios = [66, 120, 129]
    plot_venn(highway_scenarios, "RRS {} - Same Scenarios Venn".format(RRS_number), "Same Scenarios")

    # 94, 97, 121, 129, 195
    print("Comparing 3 highway scenarios highway")
    random_scenarios = [203, 420, 141]
    plot_venn(random_scenarios, "RRS {} - Distinct Scenarios Venn".format(RRS_number), "Distinct Scenarios")

    print("Done")

    print("\n")
    print("----------Same vs Distinct RRS------------")

    # Create all permutations of the test indices
    total_traces = len(random_traces_hashes)
    indices = np.arange(total_traces)
    perms = itertools.combinations(indices, r=3)

    # Create copies of perms (need to do this as each is going to iterate independently)
    perms_list = []
    for i in range(args.cores):
        perms_list.append(copy.deepcopy(perms))

    # Compute the number of combination
    number_combinations = math.comb(total_traces, 3)
    print("Computing all combinations. {} choose 3: {}".format(total_traces, number_combinations))

    # Get the start and end indices
    combination_index = np.arange(number_combinations)
    combination_index = np.array_split(combination_index, args.cores)

    # Create the pool for computation
    pool =  multiprocessing.Pool(processes=args.cores)

    # Call our function total_test_suits times
    jobs = []
    for i in range(args.cores):
        # Get our function parameters
        start_index = combination_index[i][0]
        end_index = combination_index[i][-1]
        current_perm = perms_list[i]

        # Call the function
        jobs.append(pool.apply_async(find_best_worst_combination, args=([current_perm, start_index, end_index])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Its 8pm the pool is closed
    pool.close() 

    maximum = np.full(TRACKING * args.cores, -np.inf)
    minimum = np.full(TRACKING * args.cores, np.inf)
    maximum_indices = np.full((TRACKING * args.cores, 3), -1)
    minimum_indices = np.full((TRACKING * args.cores, 3), -1)

    # Get the best and worst from all results
    counter = 0
    for r in results:

        # Get the results
        max_duplicates          = r[0]
        most_similar_indices    = r[1]
        min_duplicates          = r[2]
        least_similar_indices   = r[3]
        
        # Add the data to a max and min array
        for c in range(len(max_duplicates)):
            maximum[counter] = max_duplicates[c]
            minimum[counter] = min_duplicates[c]
            maximum_indices[counter] = most_similar_indices[c]
            minimum_indices[counter] = least_similar_indices[c]
            counter += 1

    # Sort to the get the TRACKING worst and TRACKING best
    max_sort_indices = np.argsort(maximum)
    min_sort_indices = np.argsort(minimum)
    print(maximum_indices[0])
    maximum = maximum[max_sort_indices][-TRACKING:]
    maximum_indices = maximum_indices[max_sort_indices][-TRACKING:]
    minimum = minimum[min_sort_indices][:TRACKING]
    minimum_indices = minimum_indices[min_sort_indices][:TRACKING]

    # Print the results
    print("\n\n")
    print("Most similarities: {}".format(maximum))
    print("Indices:\n{}\n\n".format(maximum_indices))
    
    print("Least similarities: {}".format(minimum))
    print("Indices:\n{}\n\n".format(minimum_indices))

    print("Comparing Similar RRS")
    plot_venn(maximum_indices[-1], "RRS {} - Same RRS Venn".format(RRS_number), "Same RRS")

    print("Comparing Distinct RRS")
    plot_venn(minimum_indices[0], "RRS {} - Distinct RRS Venn".format(RRS_number), "Distinct RRS")

    # Create a plot of the best and worst
    show_data_from_index(maximum_indices[-1], title="Most similar RRS camera data", camera=True)
    show_data_from_index(minimum_indices[0], title="Least similar RRS camera data", camera=True)
    show_data_from_index(maximum_indices[-1], title="Most similar RRS reach data", camera=False)
    show_data_from_index(minimum_indices[0], title="Least similar RRS reach data", camera=False)
    show_most_common_RRS(maximum_indices[-1], args.distribution, args.scenario, title="Most similar RRS data")
    show_most_common_RRS(minimum_indices[-1], args.distribution, args.scenario, title="Least similar RRS data")

    print("Done")

plt.show()