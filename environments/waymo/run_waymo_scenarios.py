# Thanks
# https://notebook.community/waymo-research/waymo-open-dataset/tutorial/tutorial

import os
import sys
import glob
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/waymo")])
sys.path.append(base_directory)

from tqdm import tqdm
from scenario_convert_functions import convert_file_to_raw_vector

from general.environment_configurations import RRSConfig
from general.environment_configurations import WaymoKinematics

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',     type=str,  default="/mnt/extradrive3/PhysicalCoverageData",          help="The path to the data")
parser.add_argument('--cores',         type=int, default=4,                                                 help="number of available cores")
args = parser.parse_args()


# Get the different configurations
WK = WaymoKinematics()
RRS = RRSConfig(beam_count=31)

# Variables - Used for timing
total_lines     = RRS.beam_count
steering_angle  = WK.steering_angle
max_distance    = WK.max_velocity

# Get all the files
all_files = glob.glob("{}/waymo/random_tests/physical_coverage/frames/*.tfrecord".format(args.data_path))

# List which files were found
print("Found {} scenario files".format(len(all_files)))
if len(all_files) <= 0:
    exit()

# Create the output directory if it doesn't exists
if not os.path.exists('../../output/'):
    os.makedirs('../../output/')

# Create a list of processors
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
for filename in all_files:
    jobs.append(pool.apply_async(convert_file_to_raw_vector, args=([filename, total_lines, steering_angle, max_distance])))

# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

print("Done")