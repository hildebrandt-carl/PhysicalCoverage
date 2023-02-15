# Thanks
# https://notebook.community/waymo-research/waymo-open-dataset/tutorial/tutorial

import os
import sys
import time
import glob
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

from tqdm import tqdm

from utils.file_functions import get_beam_number_from_file
from utils.file_functions import order_files_by_beam_number
from utils.environment_configurations import RRSConfig
from utils.environment_configurations import WaymoKinematics
from utils.RRS_distributions import center_close_distribution
from utils.RRS_distributions import center_full_distribution
from utils.common import rotate

def get_clean_lidar(frame, save_folder, frame_counter):

    # Convert the frame into a camera projection
    (range_images, camera_projections, seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    # Convert the lidar into cartesian data
    points = frame_utils.convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, 0, False)

    # Get the LIDAR from the front lidar
    cartesian_points = points[1]

    # Combine all the data from each of the different LiDARS
    points_all = np.concatenate(cartesian_points, axis=0)

    # Declare the range
    LOW_Z = 0.75
    HIGH_Z = 1.25

    # Create the array to hold the data
    points_at_z_height = []

    # For each of the points
    for i in range(len(points_all)):

        # If it is at the right height and in front of the vehicle
        if (LOW_Z < points_all[i][2] < HIGH_Z) and (points_all[i][0] >= -1):
            points_at_z_height.append(points_all[i])

    # Convert the points into a numpy array
    points_at_z_height = np.array(points_at_z_height)

    return points_at_z_height


def create_distribution_video(frame_name, scenario_name, traces, distribution, save_folder, distribution_name):

    # Create the output directory if it doesn't exists
    out_folder = '{}/{}/rrs_data/{}'.format(save_folder, scenario_name, distribution_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Load the data
    dataset = tf.data.TFRecordDataset("{}".format(frame_name), compression_type='')

    # Get angle information needed for plotting
    angles              = distribution.get_angle_distribution()
    samples             = distribution.get_sample_distribution()
    predefined_points   = distribution.get_predefined_points()

    frame_counter = -1

    # Go through each of the RRS
    for data, rrs in zip(dataset, traces):

        # Create the subplot
        plt.figure(figsize = (10, 10))
        plt.title('RRS')
        frame_counter += 1

        # Load the frame
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # If we have got to the end of the file
        if np.isnan(rrs).any():
            break

        rrs_lengths = list(rrs)
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

        # Get and clean the lidar data
        cloud_points = get_clean_lidar(frame, additional_info_save_folder, frame_counter)

        # Remove z
        cloud_points = cloud_points[:,0:2]

        # Get the x and y
        cloud_points_x = cloud_points[:,0]
        cloud_points_y = cloud_points[:,1]

        # Create the cloud points
        plt.scatter(cloud_points_x, cloud_points_y, c="C3", alpha=0.95, s=1)

        # Create the possible points
        plt.scatter(possible_points[:, 0], possible_points[:, 1], c="C4", alpha=0.95, s=10)

        # Create the plot
        for p in actual_points:
            x = [0, p[0]]
            y = [0, p[1]]
            plt.plot(x, y, c="C0")

        plt.xlim([75, -75])
        plt.ylim([-75, 75])
        plt.savefig("{}/rrs{:05d}.png".format(out_folder, frame_counter))
        plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str,  default="/mnt/extradrive3/PhysicalCoverageData",          help="The path to the data")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                                help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                                help="The distribution")
parser.add_argument('--scenario',           type=str, default="",                                                help="The scenario")
parser.add_argument('--cores',              type=int, default=4,                                                 help="number of available cores")

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

# Get the distribution
if args.distribution == "center_close":
    distribution  = center_close_distribution(args.scenario)
elif args.distribution == "center_full":
    distribution  = center_full_distribution(args.scenario)
else:
    print("ERROR: Unknown distribution ({})".format(args.distribution))
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
if not (args.distribution == "center_close" or args.distribution == "center_full"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '/mnt/extradrive3/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
random_trace_file_names         = glob.glob(base_path + "traces_*")
random_trace_scenario_names     = glob.glob(base_path + "processed_files_*")

if (len(random_trace_file_names) <= 0) or (len(random_trace_scenario_names) < 0):
    print("files not found")
    exit()

# Create the output directory if it doesn't exists
if not os.path.exists('../output/'):
    os.makedirs('../output/')

# Create the output directory if it doesn't exists
additional_info_save_folder = '../output/additional_data/'
if not os.path.exists(additional_info_save_folder):
    os.makedirs(additional_info_save_folder)

# Get the RRS numbers
random_trace_RRS_numbers    = get_beam_number_from_file(random_trace_file_names)
random_scenario_RRS_numbers = get_beam_number_from_file(random_trace_scenario_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(random_trace_RRS_numbers) | set(random_scenario_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
random_trace_file_names         = order_files_by_beam_number(random_trace_file_names, RRS_numbers)
random_trace_scenario_names     = order_files_by_beam_number(random_trace_scenario_names, RRS_numbers)

# Get all the files
frame_names = glob.glob("{}/waymo/random_tests/physical_coverage/frames/*.tfrecord".format(args.data_path))

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

    # Create a list of processors
    total_processors = int(args.cores)
    pool =  multiprocessing.Pool(processes=total_processors)

    # Call our function total_test_suites times
    jobs = []
    for scenario_name, traces in zip(scenario_names, random_traces):
        s_name = scenario_name[:scenario_name.rfind(".")]

        # Find the frame with the right name
        f_name = ""
        for f in frame_names:
            if s_name in f:
                f_name = f
                break

        if not (("scenario430" in f_name) or ("scenario351" in f_name) or ("scenario650" in f_name)):
            continue

        if len(f_name) > 0:
            jobs.append(pool.apply_async(create_distribution_video, args=([f_name, s_name, traces, distribution, additional_info_save_folder, args.distribution])))
        else:
            print("HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(0.01)

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Time for the pool to close
    pool.close()

print("Done")