import sys
import glob
import random
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

from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number
from general.environment_configurations import RRSConfig
from general.environment_configurations import WaymoKinematics

from matplotlib_venn import venn3, venn3_circles

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

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


def number_of_duplicates_with_common_list(list_a, list_b):
    # Get all the keys and a count of them
    count_a = Counter(list_a)
    count_b = Counter(list_b)
    
    # Get all common keys between both lists
    common_keys = set(count_a.keys()).intersection(count_b.keys())

    # The count is the min number in both.
    # i.e. if list a has "200 A's, 50 B's" and list b has "100 A's 150 B's" then together they share "100 A's and 50 B's for a total of 150"
    result = sum(min(count_a[key], count_b[key]) for key in common_keys)

    # Create a common list which contains all the keys that are in common and count the min number in both
    # i.e. if list a has "200 A's, 50 B's" and list b has "100 A's 150 B's" then together they share "100 A's and 50 B's for a total of 150"
    common_list = []
    num_of_duplicates = 0
    number_of_this_key = 0
    for key in common_keys:
        # Get the number of this key
        number_of_this_key = min(count_a[key], count_b[key])
        # Add this to the duplicate counter list
        num_of_duplicates += number_of_this_key
        # Add this number of keys to the common_list
        for i in range(number_of_this_key):
            common_list.append(key)

    # Make sure they are the same size
    assert(num_of_duplicates == len(common_list))
        
    return num_of_duplicates, common_list


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",          help="The location and name of the datafolder")
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

if len(random_trace_file_names) <= 0:
    print("files not found")
    exit()

# Get the RRS numbers
random_trace_RRS_numbers    = get_beam_number_from_file(random_trace_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(random_trace_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
random_trace_file_names         = order_files_by_beam_number(random_trace_file_names, RRS_numbers)


# For each of the different beams
for RRS_index in range(len(RRS_numbers)):

    print("\n\n")
    print("------------------------------------------")
    print("------------Processing RRS: {}------------".format(RRS_numbers[RRS_index]))
    print("------------------------------------------")

    if not( RRS_numbers[RRS_index] == 5 or RRS_numbers[RRS_index] == 10):
        print("Skipping")
        continue

    # Get the beam number and files we are currently considering
    RRS_number              = RRS_numbers[RRS_index]
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
    print("---------Creating Comparison Map----------")

    comparison_map = np.zeros((np.shape(random_traces)[0], np.shape(random_traces)[0]))

    # For each of the tests
    for a in tqdm(range(np.shape(random_traces)[0])):
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


    print("\n")
    print("--------Similarities in scenarios---------")

    highway_list = [0, 1, 3, 35, 69, 83, 94, 96, 110, 120, 128, 132, 157, 159, 163, 195, 197, 203, 266, 271]

    # Create the subplot
    fig, axs = plt.subplots(4, 5)

    # Plot the pictures
    for index in range(5):
        scenario1 = highway_list[index]
        scenario2 = highway_list[index + 5]
        scenario3 = highway_list[index + 10]
        scenario4 = highway_list[index + 15]

        img_location1 = "{}/waymo/random_tests/physical_coverage/additional_data/scenario{:03d}/camera_data/camera00100.png".format(args.data_path, scenario1)
        img_location2 = "{}/waymo/random_tests/physical_coverage/additional_data/scenario{:03d}/camera_data/camera00100.png".format(args.data_path, scenario2)
        img_location3 = "{}/waymo/random_tests/physical_coverage/additional_data/scenario{:03d}/camera_data/camera00100.png".format(args.data_path, scenario3)
        img_location4 = "{}/waymo/random_tests/physical_coverage/additional_data/scenario{:03d}/camera_data/camera00100.png".format(args.data_path, scenario4)

        img1 = mpimg.imread(img_location1)
        img2 = mpimg.imread(img_location2)
        img3 = mpimg.imread(img_location3)
        img4 = mpimg.imread(img_location4)

        axs[0,index].imshow(img1)
        axs[0,index].axis('off')

        axs[1,index].imshow(img2)
        axs[1,index].axis('off')

        axs[2,index].imshow(img3)
        axs[2,index].axis('off')

        axs[3,index].imshow(img4)
        axs[3,index].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)


    # Holds the similarity count
    similarity_count_highway = []

    # Initialize the common keys as the first trace
    common_keys = random_traces_hashes[highway_list[0]]

    # Compute the similarity
    for index in tqdm(highway_list):

        # Get the index
        trace = random_traces_hashes[index]

        # Get the similarities
        duplicate_count, common_keys = number_of_duplicates_with_common_list(trace, common_keys)

        # Keep track of how similar everything is
        similarity_count_highway.append(duplicate_count)


    # Create a list of random indices
    # Set the seed
    random.seed(10)
    random_list = random.sample(range(500), 20)

    # Holds the similarity count
    similarity_count_random = []

    # Initialize the common keys as the first trace
    common_keys = random_traces_hashes[random_list[0]]

    # Compute the similarity
    for index in tqdm(random_list):

        # Get the index
        trace = random_traces_hashes[index]

        # Get the similarities
        duplicate_count, common_keys = number_of_duplicates_with_common_list(trace, common_keys)

        # Keep track of how similar everything is
        similarity_count_random.append(duplicate_count)


    plt.figure("Similarity RRS {}".format(RRS_number))
    plt.plot(similarity_count_highway, c="C0")
    plt.plot(similarity_count_random, c="C1")
    plt.xlabel("Number of scenarios")
    plt.ylabel("Number of common RRS")


    print("\n")
    print("---------------Venn Diagram---------------")

    print("Comparing highway scenarios")
    # highway
    highway_scenarios = [0, 1, 3, 35, 69, 83, 94, 96, 110, 120, 128, 132, 157, 159, 163, 195, 197, 203, 266, 271]

    test_a = random_traces_hashes[highway_scenarios[0]]
    test_b = random_traces_hashes[highway_scenarios[1]]
    test_c = random_traces_hashes[highway_scenarios[2]]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    a = len(test_a) - abc - ab - ac
    b = len(test_b) - abc - ab - bc
    c = len(test_c) - abc - ac - bc

    plt.figure("RRS {} Highway Ven".format(RRS_number))
    v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
    plt.title("Highway Scenarios RRS Similarities")

    print("Comparing random scenarios")
    # Set the seed
    random.seed(10)
    random_scenarios = random.sample(range(500), 3)

    test_a = random_traces_hashes[random_scenarios[0]]
    test_b = random_traces_hashes[random_scenarios[1]]
    test_c = random_traces_hashes[random_scenarios[2]]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    a = len(test_a) - abc - ab - ac
    b = len(test_b) - abc - ab - bc
    c = len(test_c) - abc - ac - bc

    plt.figure("RRS {} Random Ven".format(RRS_number))
    v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
    plt.title("Random Scenarios RRS Similarities")

    print("\n")
    print("--------------------TSNE-------------------")

    labels = []
    filename = '/mnt/extradrive3/PhysicalCoverageData/waymo/random_tests/physical_coverage/labels.txt'
    with open(filename) as file:
        for line in file:
            labels.append(int(line.rstrip()))
    labels = np.array(labels)

    # Set the data
    x = comparison_map
    y = labels

    print("Dimenstion before tSNE-2D: {}".format(np.shape(x)))
    tsne = make_pipeline(StandardScaler(), TSNE(n_components=2, init='pca', random_state=0))
    tsne.fit(x, y)
    x_tsne_2d = tsne.fit_transform(x)
    print('Dimensions after tSNE-2D: {}'.format(x_tsne_2d.shape))



    group1 = []
    group2 = []
    group3 = []
    selected_colors = []
    for data_index, data in enumerate(zip(x_tsne_2d[:, 0], x_tsne_2d[:, 1])):
        x_data, y_data = data
        if y_data > 100:
            group1.append(data_index)
            selected_colors.append(1)
        elif y_data > 32:
            group2.append(data_index)
            selected_colors.append(2)
        elif (x_data < -35) and (y_data < -10):
            group3.append(data_index)
            selected_colors.append(3)
        else:
            selected_colors.append(0)

    print("here!!!!!")
    print("Selected color shape")
    print(np.shape(selected_colors))
    print("y shape")
    print(np.shape(y))
    print("Selected colors")
    print(selected_colors)
    print("y")
    print(y)
    print("---------")

    print("Group 1: C1")
    print(group1)
    print("Group 2: C2")
    print(group2)
    print("Group 3: C3")
    print(group3)

    plt.figure("tSNE-2D RSS {}".format(RRS_number))
    plt.title('tSNE-2D')
    plt.scatter(x_tsne_2d[:, 0], x_tsne_2d[:, 1], c=y, s=30, cmap='Set1')

    plt.figure("tSNE-2D no color RSS {}".format(RRS_number))
    plt.title('tSNE-2D no labels')
    plt.scatter(x_tsne_2d[:, 0], x_tsne_2d[:, 1], c="C0", s=30, cmap='Set1')

    plt.figure("tSNE-2D with grouping RRS {}".format(RRS_number))
    plt.title('tSNE-2D no labels')
    plt.scatter(x_tsne_2d[:, 0], x_tsne_2d[:, 1], c=np.array(selected_colors), s=30, cmap='Set1')

    print("\n")
    print("-------Finding best K for KMeans----------")
    # Finding best k
    sse = {}
    for k in tqdm(range(1, 25)):
        kmeans = KMeans(n_clusters=k, max_iter=1000, n_init='auto')
        clusters = kmeans.fit_predict(comparison_map)
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure("RRS {} best k".format(RRS_number))
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")

    print("\n")
    print("--------------Applying KMeans-------------")

    kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
    clusters = kmeans.fit_predict(comparison_map)
    # Get the closest points to each cluster
    closest_n = 5
    distances = pairwise_distances(kmeans.cluster_centers_, comparison_map, metric='euclidean')
    ind = [np.argpartition(i, closest_n)[:closest_n] for i in distances]

    for cluster_id in range(np.shape(ind)[0]):
        print("Closest indices to cluster {}: {}".format(cluster_id, ind[cluster_id].tolist()))


plt.show()
