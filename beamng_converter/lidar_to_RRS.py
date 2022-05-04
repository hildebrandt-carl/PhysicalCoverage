import os
import re
import sys
import glob
import argparse
import multiprocessing

import matplotlib.pyplot as plt

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/beamng")])
sys.path.append(base_directory)

from tqdm import tqdm
from lidar_to_RRS_functions import process_file

from general.environment_configurations import RRSConfig
from general.environment_configurations import BeamNGKinematics

parser = argparse.ArgumentParser()
parser.add_argument('--cores',            type=int, default=120,  help="number of available cores")
parser.add_argument('--scenario',         type=str, default="",   help="beamng_random/beamng_generated")
parser.add_argument('--distribution',     type=str, default="",   help="Only used when using beamng_generated (linear/center_close/center_mid)")
parser.add_argument('--plot',             action='store_true')
args = parser.parse_args()

# Create the configuration classes
NG = BeamNGKinematics()
RRS = RRSConfig(beam_count = 30)

# Save the kinematics and RRS parameters
steering_angle  = NG.steering_angle
max_distance    = NG.max_velocity
total_lines     = RRS.beam_count

if args.scenario == "beamng_random":
    raw_file_location       = "/media/carl/DataDrive/PhysicalCoverageData/beamng/random_tests/physical_coverage/lidar/"
    output_file_location    = "../output/beamng/random_tests/physical_coverage/raw/"
    file_names = glob.glob(raw_file_location + "/*/*.csv")
elif args.scenario == "beamng_generated":
    raw_file_location       = "/media/carl/DataDrive/PhysicalCoverageData/beamng/generated_tests/{}/physical_coverage/lidar/".format(args.distribution)
    output_file_location    = "../output/beamng/generated_tests/{}/physical_coverage/raw/".format(args.distribution)
    file_names = glob.glob(raw_file_location + "/*/*.csv")
else:
    print("Error: Unknown Scenario")
    exit()

print("Processing: {} files".format(len(file_names)))
assert(len(file_names) > 0)

# Create a pool with x processes
total_processors = args.cores
if args.plot:
    total_processors = 1
pool =  multiprocessing.Pool(total_processors)
jobs = []
file_number = 0

for file_name in file_names:

    # Compute the file name in the format vehiclecount-time-run#.txt
    name_only = file_name[file_name.rfind('/')+1:]
    folder = file_name[0:file_name.rfind('/')]
    folder = folder[folder.rfind('/')+1:]

    # Compute the number of external vehicles
    external_vehicle_count = folder[:folder.find("_")]

    if not os.path.exists(output_file_location + folder):
        os.makedirs(output_file_location + folder)

    save_name = ""
    save_name += str(output_file_location)
    save_name += folder + "/"
    save_name += name_only[0:-4] + ".txt"

    # Run each of the files in a separate process
    jobs.append(pool.apply_async(process_file, args=(file_name, save_name, external_vehicle_count, file_number, total_lines, steering_angle, max_distance, args.plot)))
    file_number += 1

# Wait to make sure all files are finished
results = []
for job in tqdm(jobs):
    results.append(job.get())

# pool.close()
print("All files completed")