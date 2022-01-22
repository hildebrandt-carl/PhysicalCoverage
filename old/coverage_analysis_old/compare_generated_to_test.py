import sys
import glob
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from general_functions import order_by_beam
from general_functions import get_beam_numbers

from general.environment_configurations import RSRConfig
from general.environment_configurations import BeamNGKinematics
from general.environment_configurations import HighwayKinematics

# Get the coverage on a random test suite 
def coverage_of_test(test_index):
    global generated_traces

    # Used to compute the coverage for this trace
    unique_RSR = set()

    # Get the right test
    test = generated_traces[test_index]

    # Go through each of the indices
    for RSR in test:
    
        # Get the current scene
        s = tuple(RSR)

        # Make sure that this is a scene (not a nan or inf or -1)
        if (np.isnan(RSR).any() == False) and (np.isinf(RSR).any() == False) and (np.less(RSR, 0).any() == False):
            unique_RSR.add(tuple(s))

    return unique_RSR


# Get the coverage on a random test suite 
def locate_RSR_for_test(current_test):
    global test_names
    global generated_tests

    # Find and load the corresponding test file
    index = -1
    for k in range(len(test_names)):
        
        # Get the possible candidate
        t = test_names[k]
        t = t[t.rfind("/")+1:]

        # Check if they match
        if current_test + "_index.npy" == t:
            index = k
            break

    # Make sure we find all the correct tests
    if index == -1:
        print("Test not found")
        required_RSR = None

    # Load the test file
    required_RSR = np.load(test_names[index])
    required_RSR = tuple(required_RSR[1])

    return required_RSR



parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
NG = BeamNGKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_accuracy = RSR.accuracy
if args.scenario == "highway":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "beamng":
    new_steering_angle  = NG.steering_angle
    new_max_distance    = NG.max_velocity
else:
    print("ERROR: Unknown scenario")
    exit()

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = ""
load_name += "_s" + str(new_steering_angle) 
load_name += "_b" + str('*') 
load_name += "_d" + str(new_max_distance) 
load_name += "_a" + str(new_accuracy)
load_name += "_t" + str(args.total_samples)
load_name += ".npy"

# Get the generated trace and crash file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/generated_tests/tests_single/processed/' + str(args.total_samples) + "/"
generated_trace_file_names      = glob.glob(base_path + "traces_*.npy")
generated_processed_file_names  = glob.glob(base_path + "processed_files_*.npy")

# Make sure you have all the files you need
assert(len(generated_trace_file_names) >= 1)
assert(len(generated_processed_file_names) >= 1)

# Get the beam numbers
generated_trace_beam_numbers        = get_beam_numbers(generated_trace_file_names)
generated_processed_beam_numbers    = get_beam_numbers(generated_processed_file_names)


# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(generated_trace_beam_numbers) & set(generated_processed_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
generated_trace_file_names      = order_by_beam(generated_trace_file_names, beam_numbers)
generated_processed_file_names  = order_by_beam(generated_processed_file_names, beam_numbers)

# Count the success rate
total_tests_processed = 0
total_tests_success = 0

# Save the different success rates
success_rate_array = []

# For each of the different beams
for i in range(len(beam_numbers)):

    # Get the beam number and files we are currently considering
    beam_number = beam_numbers[i]
    generated_trace_file        = generated_trace_file_names[i]
    generated_processed_file    = generated_processed_file_names[i]

    print("\nProcessing beams: {}".format(beam_numbers[i]))

    # Load the traces and crashes
    global generated_traces
    generated_traces = np.load(generated_trace_file)
    global generated_tests
    generated_tests  = np.load(generated_processed_file)

    # Get all the test names for this beam number
    base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/generated_tests/tests_single/tests/' + str(args.total_samples) + '/' + str(beam_number) + '_beams/'
    global test_names
    test_names      = np.array(glob.glob(base_path + "*_index.npy"))






    print("Getting the RSR data seen during a test")

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=args.cores)
    jobs = []

    # Preprocess the files to compute the RSR seen vectors
    for j in range(len(generated_tests)):
        jobs.append(pool.apply_async(coverage_of_test, args=([j])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Save the coverage seen in tests so we can compare it to the test later
    coverage_seen_in_test = results

    # Close the pool
    pool.close()





    print("Locating the test and getting required RSR")

    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=args.cores)
    jobs = []


    # Find the test and get the RSR values for that test
    for j in range(len(generated_tests)):

        # Get the test name
        current_test = generated_tests[j]
        current_test = current_test[:current_test.find("_")]

        # Get the RSR values for that test
        jobs.append(pool.apply_async(locate_RSR_for_test, args=([current_test])))
        
    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Save the data
    RSR_for_given_test = results

    # Close the pool
    pool.close()





    print("Comparing the required RSR to the seen RSR")
    for j in tqdm(range(len(generated_tests))):

        # Load the seen RSR
        seen_RSR = coverage_seen_in_test[j]

        # Load the required RSR
        required_RSR = RSR_for_given_test[j]
    
        # Check if we saw the required RSR value
        test_exposed_expected_RSR = False
        if required_RSR in seen_RSR:
            test_exposed_expected_RSR = True

        # count the success rate
        total_tests_processed += 1
        if test_exposed_expected_RSR:
            total_tests_success += 1




    # Print success rate
    success_rate = np.round((total_tests_success / total_tests_processed) * 100, 4)
    success_rate_array.append(success_rate)

print("")
print("---------------------------------------")
print("---------------Results-----------------")
print("---------------------------------------")
for i in range(len(beam_numbers)):
    beam_number = beam_numbers[i]
    success_rate = success_rate_array[i]
    print("RSR {} Success rate: {}%".format(beam_number, success_rate))

