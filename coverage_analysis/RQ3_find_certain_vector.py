import sys
import glob
import argparse

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np

from tqdm import tqdm

from utils.file_functions import get_beam_number_from_file
from utils.file_functions import order_files_by_beam_number
from utils.environment_configurations import RRSConfig
from utils.environment_configurations import BeamNGKinematics
from utils.environment_configurations import HighwayKinematics
from utils.environment_configurations import WaymoKinematics


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',              type=str, default="/mnt/extradrive3/PhysicalCoverageData",      help="The location and name of the datafolder")
parser.add_argument('--highway_number_tests',   type=int, default=1000000,                                      help="The number of tests for highway-env")
parser.add_argument('--beamng_number_tests',    type=int, default=10000,                                        help="The number of tests for beamng")
parser.add_argument('--waymo_number_tests',     type=int, default=798,                                          help="The number of tests for waymo")
parser.add_argument('--RRS_number',             type=int, default=10,                                           help="The RRS number")
parser.add_argument('--distribution',           type=str, default="",                                           help="center_close/center_full")

args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
BK = BeamNGKinematics()
WK = WaymoKinematics()
RRS = RRSConfig()

hw_steering_angle   = HK.steering_angle
hw_max_distance     = HK.max_velocity
ng_steering_angle   = BK.steering_angle
ng_max_distance     = BK.max_velocity
wm_steering_angle   = WK.steering_angle
wm_max_distance     = WK.max_velocity

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

# Checking the distribution
if not (args.distribution == "center_close" or args.distribution == "center_full"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()


print("Checking for Waymo Data")

load_name = ""
load_name += "_s" + str(wm_steering_angle) 
load_name += "_b" + str(args.RRS_number) 
load_name += "_d" + str(wm_max_distance) 
load_name += "_t" + str(args.waymo_number_tests)
load_name += ".npy"

# Get the file names
wm_base_path                = '{}/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.data_path, "waymo", args.distribution, args.waymo_number_tests)
wm_trace_file_names         = glob.glob(wm_base_path + "traces_*" + load_name)

# Get the feasible vectors
wm_base_path = '{}/{}/feasibility/processed/{}/'.format(args.data_path, "waymo", args.distribution)
wm_feasible_file_names = glob.glob(wm_base_path + "*_b{}.npy".format(args.RRS_number))

# Get the RRS numbers
wm_trace_RRS_numbers = get_beam_number_from_file(wm_trace_file_names)
wm_feasibility_RRS_numbers = get_beam_number_from_file(wm_feasible_file_names)


print("Combining data")
# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(wm_trace_RRS_numbers) | set(wm_feasibility_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)


print("Done")


##################################################################################################################
###########################################Generate PhysCov Coverage##############################################
##################################################################################################################

print("----------------------------------")
print("----Computing PhysCov Coverage----")
print("----------------------------------")
# Used to control the colors
color_counter = 1

# For each of the different RRS_numbers
for i in range(len(RRS_numbers)):

    # Processing RRS
    print("Processing RRS: {}\n\n".format(RRS_numbers[i]))

    # Get the beam number and files we are currently considering
    RRS_number          = RRS_numbers[i]

    wm_trace_file       = wm_trace_file_names[i]
    wm_feasibility_file = wm_feasible_file_names[i]

    # Skip if any of the files are blank
    if wm_trace_file == "" or wm_feasibility_file == "":
        print("Could not find one of the files")
        continue

    # Create the list of scenarios and files        
    scenarios           = ["Waymo"]
    trace_files         = [wm_trace_file]
    feasibility_files   = [wm_feasibility_file]

    # Load all the files
    for i, scenario in enumerate(scenarios):

        print("Processing {}".format(scenario))

        # Load the trace file
        traces = np.load(trace_files[i])

        # Load the feasibility file
        feasible_traces = np.load(feasibility_files[i])

        # Compute the denominator
        print("Computing denominator")
        all_RRS = set()
        for f in tqdm(feasible_traces):
            all_RRS.add(tuple(f))

        # Compute the numerator
        print("Computing numerator")
        seen_RRS = set()
        for trace in tqdm(traces):
            for t in trace:
                if np.any(np.isnan(t)) == False:
                    seen_RRS.add(tuple(t))

    known_included_vector = tuple([15, 15, 5, 5, 35, 35, 35, 35, 15, 5])
    assert(known_included_vector in seen_RRS)

    vector_of_interest = tuple([35, 35, 5, 5, 5, 5, 5, 5, 5, 5])
    print("{} is in seen RRS: {}".format(vector_of_interest, vector_of_interest in seen_RRS))