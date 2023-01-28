import sys
import glob
import hashlib
import argparse

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
from prettytable import PrettyTable

from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number
from general.environment_configurations import RRSConfig
from general.environment_configurations import WaymoKinematics

from matplotlib_venn import venn3, venn3_circles


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


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",     help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",                                               help="waymo")
args = parser.parse_args()

# Create the configuration classes
WK = WaymoKinematics()
RRS = RRSConfig()

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
random_trace_file_names = glob.glob(base_path + "traces_*")

# Get the RRS numbers
random_trace_RRS_numbers    = get_beam_number_from_file(random_trace_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(random_trace_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
random_trace_file_names         = order_files_by_beam_number(random_trace_file_names, RRS_numbers)


# For each of the different beams
for RRS_index in range(len(RRS_numbers)):
    print("Processing RRS: {}".format(RRS_numbers[RRS_index]))

    # Get the beam number and files we are currently considering
    RRS_number              = RRS_numbers[RRS_index]

    if not ((RRS_number == 5) or (RRS_number == 6)):
        continue

    random_trace_file       = random_trace_file_names[RRS_index]

    # Skip if any of the files are blank
    if random_trace_file == "":
        print(random_trace_file)
        print("\nWarning: Could not find one of the files for RRS number: {}".format(RRS_number))
        continue

    # Load the random_traces
    global random_traces
    random_traces = np.load(random_trace_file)

    # Holds the trace as a set of hashes    
    random_traces_hashes = []

    # Convert to hash value for easier comparison
    for test_number in range(np.shape(random_traces)[0]):
        hashes = []
        current_trace = random_traces[test_number, :, :]
        for t in current_trace:
            # ignore nans
            if not np.isnan(t).any():
                # Convert to a hash
                trace_string = str(t)
                hash = hashlib.md5(trace_string.encode()).hexdigest()
                hashes.append(hash)

        # Save the hashes
        random_traces_hashes.append(hashes)

    comparison_map = np.zeros((np.shape(random_traces)[0], np.shape(random_traces)[0]))

    # For each of the tests
    for a in range(np.shape(random_traces)[0]):
        for b in range(np.shape(random_traces)[0]):
            
            # Get test A ad B
            test_a = random_traces_hashes[a]
            test_b = random_traces_hashes[b]

            # Count duplicates
            dup_count = number_of_duplicates(test_a, test_b)

            # Save to a comparison map
            comparison_map[a, b] = dup_count

    # Plot the map
    plt.figure("RRS {}".format(RRS_number))
    plt.imshow(comparison_map)

    # Create the table
    table_heading = ['Least similar', '2nd Least similar', 'current', '2nd Most similar', 'Most similar']
    table = PrettyTable(table_heading)

    # Use the comparison map to identify the most similar and least similar for each of the scenarios
    number_of_comparisons = 5
    fig, axs = plt.subplots(number_of_comparisons, 5)
    for i in range(number_of_comparisons):
        similarity = np.argsort(comparison_map[i])
        current_index = similarity[-1]
        first_most_similar = similarity[-2]
        second_most_similar = similarity[-3]
        first_least_similar = similarity[1]
        second_least_similar = similarity[0]
        table.add_row([str(first_least_similar), str(second_least_similar), str(current_index), str(second_most_similar), str(first_most_similar)])

        indices = [first_least_similar, second_least_similar, current_index, second_most_similar, first_most_similar]
        for j, loc in enumerate(indices):
            img_location = "/home/carl/Desktop/PhysicalCoverage/output/additional_data/scenario_{:03d}/camera_data/camera00001.png".format(loc)
            img = mpimg.imread(img_location)
            axs[i, j].imshow(img)
            # Turn off the axis
            axs[i, j].axis('off')
            if i == 0:
                axs[i, j].set_title(table_heading[j])


    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Plot the comparison table
    print(table)

    print("Comparing highway scenarios")
    # highway
    highway_scenarios = [9, 28, 38]

    test_a = random_traces_hashes[highway_scenarios[0]-1]
    test_b = random_traces_hashes[highway_scenarios[1]-1]
    test_c = random_traces_hashes[highway_scenarios[2]-1]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    a = len(test_a) - abc - ab - ac
    b = len(test_b) - abc - ab - bc
    c = len(test_c) - abc - ac - bc

    plt.figure("RRS {} Highway Ven".format(RRS_number))
    v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
    plt.title("Not a Venn diagram")

    print("Comparing random scenarios")
    # dense street, #steep road, #back ally
    random_scenarios = [47, 50, 21]

    test_a = random_traces_hashes[random_scenarios[0]-1]
    test_b = random_traces_hashes[random_scenarios[1]-1]
    test_c = random_traces_hashes[random_scenarios[2]-1]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    a = len(test_a) - abc - ab - ac
    b = len(test_b) - abc - ab - bc
    c = len(test_c) - abc - ac - bc

    plt.figure("RRS {} Random Ven".format(RRS_number))
    v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
    plt.title("Not a Venn diagram")


plt.show()





