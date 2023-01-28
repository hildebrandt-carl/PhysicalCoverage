import os
import sys
import glob
import math
import numpy as np
import argparse

from tqdm import tqdm

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/test_generation")])
sys.path.append(base_directory)

from general.RRS_distributions import linear_distribution
from general.RRS_distributions import center_close_distribution
from general.RRS_distributions import center_mid_distribution

from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number
from general.environment_configurations import RRSConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',              type=str, default="/mnt/extradrive3/PhysicalCoverageData",     help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',        type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',           type=str, default="",                                               help="linear/center_close/center_mid")
parser.add_argument('--scenario',               type=str, default="",                                               help="beamng/highway")
parser.add_argument('--cores',                  type=int, default=4,                                                help="number of available cores")
parser.add_argument('--maximum_tests',          type=int, default=10000,                                            help="The maximum number of unseen vectors")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
BK = BeamNGKinematics()
RRS = RRSConfig()

# Save the kinematics and RRS parameters
if args.scenario == "highway":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "beamng":
    new_steering_angle  = BK.steering_angle
    new_max_distance    = BK.max_velocity
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
trace_file_names = glob.glob(base_path + "traces_*")

# Get the feasible vectors
base_path = '/mnt/extradrive3/PhysicalCoverageData/{}/feasibility/processed/{}/'.format(args.scenario, args.distribution)
feasible_file_names = glob.glob(base_path + "*.npy")

# Get the RRS numbers
trace_RRS_numbers = get_beam_number_from_file(trace_file_names)
feasibility_RRS_numbers = get_beam_number_from_file(feasible_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(trace_RRS_numbers) | set(feasibility_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
trace_file_names = order_files_by_beam_number(trace_file_names, RRS_numbers)
feasible_file_names = order_files_by_beam_number(feasible_file_names, RRS_numbers)

for i, _ in enumerate(RRS_numbers):
    # Save the tests
    print("Processing RRS: {}".format(RRS_numbers[i]))
    trace_file = trace_file_names[i]
    feasibility_file = feasible_file_names[i]   

    trace       = np.load(trace_file)
    feasible    = np.load(feasibility_file)

    seen_traces = set()
    feasible_traces = set()

    print("Computing the feasible set of RRS:")

    for index in tqdm(range(len(feasible))):
        # Get the feasible RRS
        t = feasible[index]
        feasible_traces.add(tuple(t))

    print("Computing all RRS seen during random test generation:")
    for index in tqdm(range(len(trace))):
        # Get the test data
        test = trace[index]
        for t in test:
            if not np.isnan(t).any():
                seen_traces.add(tuple(t))

    missing_tests_set = feasible_traces - seen_traces

    print("Generated {} tests".format(len(missing_tests_set)))

    missing_tests = []
    for t in missing_tests_set:
        missing_tests.append(list(t))

    # Save the sets to files
    distribution = None
    if args.distribution == "linear":
        distribution = linear_distribution(args.scenario)
    elif args.distribution == "center_close":
        distribution = center_close_distribution(args.scenario)
    elif args.distribution == "center_mid":
        distribution = center_mid_distribution(args.scenario)

    # We need to determine what angle to place the vehicle
    angles = distribution.get_angle_distribution()
    angles = angles[RRS_numbers[i]]

    # Create the output directory if it doesn't exists
    save_path = '../output/{}/generated_tests/{}/tests/'.format(args.scenario, args.distribution)
    new_path = save_path + "{}_external_vehicles/".format(i+1)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # For each test
    for j, trace in enumerate(missing_tests):
        if j > args.maximum_tests:
            break
        # Open the file

        output_file = open(new_path + "test_{}.txt".format(j),"w+")
        output_file.write("RRS: {}\n".format(trace))
        for h, a in zip(trace, angles):
            sign = 1
            if a < 0:
                a = abs(a)
                sign = -1
            x = h * math.cos(math.radians(a))
            if a == 0:
                y = 0
            else:
                y = h * math.sin(math.radians(a)) * sign
            if math.isinf(x):
                x = 0
            if math.isinf(y):
                y = 0
            output_file.write("{}, {}, {}\n".format(round(a,4), round(x,4), round(y,4)))

        # Close the file
        output_file.close()