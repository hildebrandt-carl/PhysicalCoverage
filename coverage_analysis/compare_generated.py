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
def coverage_of_test(test_index, generated):
    global random_traces
    global generated_traces

    # Used to compute the coverage for this trace
    unique_RSR = set()

    # Get the right test
    if not generated:
        test = random_traces[test_index]
    elif generated:
        test = generated_traces[test_index]

    # Go through each of the indices
    for RSR in test:
    
        # Get the current scene
        s = tuple(RSR)

        # Make sure that this is a scene (not a nan or inf or -1)
        if (np.isnan(RSR).any() == False) and (np.isinf(RSR).any() == False) and (np.less(RSR, 0).any() == False):
            unique_RSR.add(tuple(s))

    return unique_RSR

# Get the crash signatures of a random test
def crash_of_test(test_index, generated):
    global random_crashes
    global generated_crashes

    # Used to compare the crashes for this test
    unique_crashes = set()
    total_crashes = 0

    # Get the crash test
    if not generated:
        crash_data = random_crashes[test_index]
    elif generated:
        crash_data = generated_crashes[test_index]

    # If it is a crash
    crash = None
    for c in crash_data:
        if ~np.isinf(c):
            total_crashes += 1
            unique_crashes.add(c)

    return unique_crashes, total_crashes



parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
parser.add_argument('--ordered',        action='store_true')
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

# Get the feasible vectors
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/feasibility/processed/'
feasible_file_names = glob.glob(base_path + "*.npy")

# Get the random trace and crash file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/physical_coverage/processed/' + str(args.total_samples) + "/"
random_trace_file_names = glob.glob(base_path + "traces_*.npy")
random_crash_file_names = glob.glob(base_path + "crash_*.npy")
random_time_file_names  = glob.glob(base_path + "time_*.npy")

# Get the generated trace and crash file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/generated_tests/tests_single/processed/' + str(args.total_samples) + "/"
generated_trace_file_names = glob.glob(base_path + "traces_*.npy")
generated_crash_file_names = glob.glob(base_path + "crash_*.npy")
generated_time_file_names  = glob.glob(base_path + "time_*.npy")

# Make sure you have all the files you need
assert(len(random_trace_file_names) > 1)
assert(len(random_crash_file_names) > 1)
assert(len(random_time_file_names) > 1)
assert(len(generated_trace_file_names) > 1)
assert(len(generated_crash_file_names) > 1)
assert(len(generated_time_file_names) > 1)
assert(len(feasible_file_names) > 1)

# Get the beam numbers
random_trace_beam_numbers       = get_beam_numbers(random_trace_file_names)
random_crash_beam_numbers       = get_beam_numbers(random_crash_file_names)
random_time_beam_numbers        = get_beam_numbers(random_time_file_names)
generated_trace_beam_numbers    = get_beam_numbers(generated_trace_file_names)
generated_crash_beam_numbers    = get_beam_numbers(generated_crash_file_names)
generated_time_beam_numbers     = get_beam_numbers(generated_time_file_names)
feasibility_beam_numbers        = get_beam_numbers(feasible_file_names)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(random_trace_beam_numbers) &
                    set(random_crash_beam_numbers) &
                    set(random_time_beam_numbers) &
                    set(generated_trace_beam_numbers) &
                    set(generated_crash_beam_numbers) &
                    set(generated_time_beam_numbers) &
                    set(feasibility_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
random_trace_file_names     = order_by_beam(random_trace_file_names, beam_numbers)
random_crash_file_names     = order_by_beam(random_crash_file_names, beam_numbers)
random_time_file_names      = order_by_beam(random_time_file_names, beam_numbers)
generated_trace_file_names  = order_by_beam(generated_trace_file_names, beam_numbers)
generated_crash_file_names  = order_by_beam(generated_crash_file_names, beam_numbers)
generated_time_file_names   = order_by_beam(generated_time_file_names, beam_numbers)
feasible_file_names         = order_by_beam(feasible_file_names, beam_numbers)

# Used to hold the shuffle indexes
random_idx = None

# For each of the different beams
for i in range(len(beam_numbers)):

    # Get the beam number and files we are currently considering
    beam_number = beam_numbers[i]
    random_trace_file       = random_trace_file_names[i]
    random_crash_file       = random_crash_file_names[i]
    random_time_file        = random_time_file_names[i]
    generated_trace_file    = generated_trace_file_names[i]
    generated_crash_file    = generated_crash_file_names[i]
    generated_time_file     = generated_time_file_names[i]
    feasibility_file        = feasible_file_names[i]

    print("\nProcessing beams: {}".format(beam_numbers[i]))

    # Load the feasible files
    feasible_RSR_set = set()
    feasible_traces = np.load(feasibility_file)
    for scene in feasible_traces:
        feasible_RSR_set.add(tuple(scene))

    # Load the traces and crashes
    global random_traces
    random_traces = np.load(random_trace_file)
    global random_crashes
    random_crashes = np.load(random_crash_file)
    global generated_traces
    generated_traces = np.load(generated_trace_file)
    global generated_crashes
    generated_crashes = np.load(generated_crash_file)

    # Load the times
    random_times = np.load(random_time_file)
    generated_times = np.load(generated_time_file)



    # Shuffle the data if it is not ordered
    if not args.ordered:
        # Perform a unified shuffle
        assert len(random_traces) == len(random_crashes) == len(random_times)
        assert len(generated_traces) == len(generated_crashes) == len(generated_times)

        # Shuffle indexes
        if random_idx is None:
            # We want to keep the random indexes the same, as the length is always the same.
            random_idx = np.random.permutation(len(random_traces))      
        generated_idx = np.random.permutation(len(generated_traces))

        # Perform unified shuffle
        random_traces   = random_traces[random_idx]
        random_crashes  = random_crashes[random_idx]
        random_times    = random_times[random_idx]
        generated_traces    = generated_traces[generated_idx]
        generated_crashes   = generated_crashes[generated_idx]
        generated_times     = generated_times[generated_idx]






    # Keep track of the final unique RSR values, and crash signatures 
    final_unique_RSR = set()
    final_unique_crashes = set()




    # Save the coverage array
    print("Processing random coverage")
    coverage_array = [0]
    time_array = [0]

    # Create the processing pool
    total_processors = int(args.cores)
    pool =  multiprocessing.Pool(processes=total_processors)

    # Go through each of the traces
    jobs = []
    for j in range(len(random_traces)):
        # Process each trace (Generated == False)
        jobs.append(pool.apply_async(coverage_of_test, args=([j, False])))

        # Keep track of the timing
        time_array.append(time_array[-1] + random_times[j])

    # Get the results:
    for job in tqdm(jobs):

        # Get the results from each job
        test_RSR_values = job.get()

        # Add them to the current RSR values
        final_unique_RSR = final_unique_RSR | test_RSR_values

        # Check the coverage
        coverage = len(final_unique_RSR) / len(feasible_RSR_set)
        coverage_array.append(coverage)
        

    # Close the pool
    pool.close()


    # Get the coverage switch point
    switch_point = time_array[-1]



    # Save the coverage array
    print("Processing generated coverage")

    # Create the processing pool
    total_processors = int(args.cores)
    pool =  multiprocessing.Pool(processes=total_processors)

    # Go through each of the traces
    jobs = []
    for j in range(len(generated_traces)):
        # Process each trace (Generated == True)
        jobs.append(pool.apply_async(coverage_of_test, args=([j, True])))

        # Keep track of the timing
        time_array.append(time_array[-1] + generated_times[j])

    # Get the results:
    for job in tqdm(jobs):

        # Get the results from each job
        test_RSR_values = job.get()

        # Add them to the current RSR values
        final_unique_RSR = final_unique_RSR | test_RSR_values

        # Check the coverage
        coverage = len(final_unique_RSR) / len(feasible_RSR_set)
        coverage_array.append(coverage)

    # Close the pool
    pool.close()






    # Save the coverage array
    print("Processing random crashes")
    unique_crash_array = [0]
    total_crash_array = [0]

    # Create the processing pool
    total_processors = int(args.cores)
    pool =  multiprocessing.Pool(processes=total_processors)

    # Go through each of the traces
    jobs = []
    for j in range(len(random_crashes)):
        # Process each trace (Generated == False)
        jobs.append(pool.apply_async(crash_of_test, args=([j, False])))

    # Get the results:
    for job in tqdm(jobs):

        # Get the results from each job
        unique_crashes, total_crashes = job.get()

        # Add them to the current crash signatures
        final_unique_crashes = final_unique_crashes | unique_crashes
        unique_crash_array.append(len(final_unique_crashes))

        # Add them to the current crash total count
        total_crash_array.append(total_crash_array[-1] + total_crashes)

    # Close the pool
    pool.close()






    # Save the coverage array
    print("Processing generated crashes")

    # Create the processing pool
    total_processors = int(args.cores)
    pool =  multiprocessing.Pool(processes=total_processors)

    # Go through each of the traces
    jobs = []
    for j in range(len(generated_crashes)):
        # Process each trace (Generated == True)
        jobs.append(pool.apply_async(crash_of_test, args=([j, True])))

    # Get the results:
    for job in tqdm(jobs):

        # Get the results from each job
        unique_crashes, total_crashes = job.get()

        # Add them to the current crash signatures
        final_unique_crashes = final_unique_crashes | unique_crashes
        unique_crash_array.append(len(final_unique_crashes))

        # Add them to the current crash total count
        total_crash_array.append(total_crash_array[-1] + total_crashes)

    # Close the pool
    pool.close()



    # Convert the time array to days rather than seconds
    time_array = np.array(time_array)
    time_array = time_array / 86400.0
    switch_point = switch_point / 86400.0
    


    # Plot the data
    print("Plotting the data")
    plt.figure(1)
    plt.plot(time_array, coverage_array, linestyle="-", color='C{}'.format(i), label="RSR{}".format(beam_number))
    plt.axvline(x=switch_point, color='grey', linestyle='--', linewidth=1)
    plt.figure(2)
    plt.plot(time_array, total_crash_array, linestyle="--", color='C{}'.format(i), label="Total Crashes{}".format(beam_number))
    plt.plot(time_array, unique_crash_array, linestyle="-", color='C{}'.format(i), label="Unique Crashes{}".format(beam_number))
    plt.axvline(x=switch_point, color='grey', linestyle='--', linewidth=1)


# Draw the plot
plt.figure(1)
plt.legend()
plt.xlabel("Time (Days)")
plt.ylabel("Coverage")
plt.ticklabel_format(style='plain')
plt.grid(alpha=0.5)

plt.figure(2)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Coverage")
plt.ticklabel_format(style='plain')
plt.grid(alpha=0.5)
plt.show()
