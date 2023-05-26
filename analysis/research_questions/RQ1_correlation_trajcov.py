import sys
import glob
import hashlib
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/analysis/research_questions")])
sys.path.append(base_directory)

from tqdm import tqdm
from scipy import stats
from matplotlib_venn import venn2

import numpy as np
import matplotlib.pyplot as plt

from utils.file_functions import get_beam_number_from_file
from utils.file_functions import order_files_by_beam_number

from utils.environment_configurations import RRSConfig
from utils.environment_configurations import BeamNGKinematics
from utils.environment_configurations import HighwayKinematics

from utils.trajectory_coverage import load_driving_area
from utils.trajectory_coverage import create_coverage_array
from utils.trajectory_coverage import compute_trajectory_coverage
from utils.trajectory_coverage import load_improved_bounded_driving_area

# multiple core
def random_selection(cores, test_suite_size, number_of_test_suites, scenario):
    # Create the pool for parallel processing
    pool =  multiprocessing.Pool(processes=cores)
    # Call our function total_test_suits times
    jobs = []
    for i in range(number_of_test_suites):
        jobs.append(pool.apply_async(random_select, args=([test_suite_size, scenario])))
    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())
    # Its 8pm the pool is closed
    pool.close() 

    # Get the results
    results = np.array(results)
    results = np.transpose(results)
    naive_coverage_percentages      = results[0, :]
    improved_coverage_percentages   = results[1, :]       
    unique_crash_count              = results[2, :]  




    return naive_coverage_percentages, improved_coverage_percentages, unique_crash_count

# Used to generated a random selection of tests
def random_select(number_of_tests, scenario):
    global index_x_array
    global index_y_array
    global index_lower_bound_x
    global index_lower_bound_y
    global index_upper_bound_x
    global index_upper_bound_y
    global crashes
    global stalls
    global unique_failure_set

    # Generate the indices for the random tests cases
    local_state = np.random.RandomState()
    indices = local_state.choice(vehicle_positions.shape[0], size=number_of_tests, replace=False)

    # Update index_x_array and index_y_array based on the random selection
    subset_index_x_array = index_x_array[indices]
    subset_index_y_array = index_y_array[indices]

    # Create the coverage array
    # (1  == Covered) (0  == Not Covered) (-1 == Invalid) (-2 == Boarder)
    coverage_array, subset_index_x_array, subset_index_y_array = create_coverage_array(scenario, drivable_x_size, drivable_y_size, index_lower_bound_x, index_lower_bound_y, index_upper_bound_x, index_upper_bound_y, subset_index_x_array, subset_index_y_array, debug=False)

    # # Compute the coverage
    naive_coverage_percentage, improved_coverage_percentage, coverage_array = compute_trajectory_coverage(coverage_array, 1, number_of_tests, subset_index_x_array, subset_index_y_array, drivable_x_size, drivable_y_size, debug=False)

    # We dont want the final coverage
    naive_coverage_percentage = naive_coverage_percentage[0][-1]
    improved_coverage_percentage = improved_coverage_percentage[0][-1]

    # Get the coverage and failure set
    seen_failure_set = set()

    # Compute the number of crashes
    for i in indices:
        # Get the failures
        crash = crashes[i]
        stall = stalls[i]

        # Check if there was a crash and if there was count it
        for c in crash:
            if c is not None:
                seen_failure_set.add(c)

        # Check if there was a stall and if there was count it
        for s in stall:
            if s is not None:
                seen_failure_set.add(s)

    # Compute the coverage and the crash percentage
    failures_found              = len(seen_failure_set)
    all_failures                = len(unique_failure_set)
    failure_percentage          = float(failures_found / all_failures) * 100

    return [naive_coverage_percentage, improved_coverage_percentage, failure_percentage]

# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',             type=str, default="/mnt/extradrive3/PhysicalCoverageData",       help="The location and name of the datafolder")
parser.add_argument('--number_of_test_suites', type=int, default=10,                                            help="The number of random test suites created")
parser.add_argument('--number_of_tests',       type=int, default=-1,                                            help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',          type=str, default="",                                            help="center_close/center_full")
parser.add_argument('--scenario',              type=str, default="",                                            help="beamng/highway")
parser.add_argument('--cores',                 type=int, default=4,                                             help="number of available cores")
parser.add_argument('--RRS',                   type=int, default=10,                                            help="Which RRS number you want to compute a correlation for")
args = parser.parse_args()

# Declare drivable area
drivable_x = [-10, 10]
drivable_y = [-10, 10]

# Create the configuration classes
HK = HighwayKinematics()
BK = BeamNGKinematics()

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
if not (args.distribution == "center_close" or args.distribution == "center_full"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '{}/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.data_path, args.scenario, args.distribution, args.number_of_tests)
position_file_names     = glob.glob(base_path + "ego_positions_*")
crash_file_names        = glob.glob(base_path + "crash_*")
stall_file_names        = glob.glob(base_path + "stall_*")


# Get the RRS numbers
position_RRS_numbers    = get_beam_number_from_file(position_file_names)
crash_RRS_numbers       = get_beam_number_from_file(crash_file_names)
stall_RRS_numbers       = get_beam_number_from_file(stall_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(position_RRS_numbers) | set(crash_RRS_numbers) | set(stall_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
position_file_names     = order_files_by_beam_number(position_file_names, RRS_numbers)
crash_file_names        = order_files_by_beam_number(crash_file_names, RRS_numbers)
stall_file_names        = order_files_by_beam_number(stall_file_names, RRS_numbers)

# Select a specific RRS
i = args.RRS - 1

# Get the beam number and files we are currently considering
RRS_number          = RRS_numbers[i]
position_file       = position_file_names[i]
crash_file          = crash_file_names[i]
stall_file          = stall_file_names[i]

# Skip if any of the files are blank
if position_file == "" or crash_file == "" or stall_file == "":
    print(position_file)
    print(crash_file)
    print(stall_file)
    print("\nWarning: Could not find one of the files for RRS number: {}".format(RRS_number))
    exit()

# Load the stall and crash file
global stalls
global crashes
stalls = np.load(stall_file, allow_pickle=True)
crashes = np.load(crash_file, allow_pickle=True)

# Create the failure unique set
global unique_failure_set
unique_failure_set = set()
for crash in crashes:
    for c in crash:
        if c is not None:
            unique_failure_set.add(c)
for stall in stalls:
    for s in stall:
        if s is not None:
            unique_failure_set.add(s)

# Get the drivable area
drivable_x, drivable_y = load_driving_area(args.scenario)

# Compute the size of the drivable area
global drivable_x_size
drivable_x_size = drivable_x[1] - drivable_x[0]
global drivable_y_size
drivable_y_size = drivable_y[1] - drivable_y[0]
print("Loaded driving area")

# Load the vehicle positions
global vehicle_positions
vehicle_positions = np.load(position_file)
print("Loaded data")

# Get all X Y and Z positions
x_positions = vehicle_positions[:,:,0]
y_positions = vehicle_positions[:,:,1]
z_positions = vehicle_positions[:,:,2]

# Remove all nans
x_positions[np.isnan(x_positions)] = sys.maxsize
y_positions[np.isnan(y_positions)] = sys.maxsize

# Convert each position into an index
global index_x_array
index_x_array = np.round(x_positions, 0).astype(int) - drivable_x[0] -1
global index_y_array
index_y_array = np.round(y_positions, 0).astype(int) - drivable_y[0] -1

global index_lower_bound_x
global index_lower_bound_y
global index_upper_bound_x
global index_upper_bound_y

# Load the upper and lower bound of the driving area
if args.scenario == "beamng":
    print("Loading improved driving area")
    bounds = load_improved_bounded_driving_area(args.scenario)
    lower_bound_x, lower_bound_y, upper_bound_x, upper_bound_y = bounds

    # Convert the bounds to indices
    index_lower_bound_x = np.round(lower_bound_x, 0).astype(int) - drivable_x[0] -1
    index_lower_bound_y = np.round(lower_bound_y, 0).astype(int) - drivable_y[0] -1
    index_upper_bound_x = np.round(upper_bound_x, 0).astype(int) - drivable_x[0] -1
    index_upper_bound_y = np.round(upper_bound_y, 0).astype(int) - drivable_y[0] -1

    # Make sure these bounds are confined by available space
    index_lower_bound_x = np.clip(index_lower_bound_x, 0, drivable_x_size-1)
    index_lower_bound_y = np.clip(index_lower_bound_y, 0, drivable_y_size-1)
    index_upper_bound_x = np.clip(index_upper_bound_x, 0, drivable_x_size-1)
    index_upper_bound_y = np.clip(index_upper_bound_y, 0, drivable_y_size-1)
elif args.scenario == "highway":
    index_lower_bound_x = None
    index_lower_bound_y = None
    index_upper_bound_x = None
    index_upper_bound_y = None
else:
    print("Scenario not known")
    exit()


test_suit_sizes = [10, 50, 100, 500, 1000, 5000]

f_i = open("Traj_imporved-{}.txt".format(args.scenario), "w")
f_n = open("Traj_naive-{}.txt".format(args.scenario), "w")

for j, test_suite_size in enumerate(test_suit_sizes):
    print("Computing {} test suites of size {}".format(args.number_of_test_suites, test_suite_size))
    # Create random test suites
    results = random_selection(cores=args.cores,
                               test_suite_size=test_suite_size,
                               number_of_test_suites=args.number_of_test_suites,
                               scenario=args.scenario)

    # Get the results
    naive_coverage_percentage       = results[0]
    improved_coverage_percentage    = results[1]
    unique_crash_count              = results[2]

    # Compute the correlation
    n_r = stats.pearsonr(naive_coverage_percentage, unique_crash_count)
    n_r_value = round(n_r[0], 4)
    n_p_value = round(n_r[1], 4)

    i_r = stats.pearsonr(improved_coverage_percentage, unique_crash_count)
    i_r_value = round(i_r[0], 4)
    i_p_value = round(i_r[1], 4)

    f_n.write("Test suite size: {}\n".format(test_suite_size))
    f_n.write("Naive Trajectory Coverage             R value: {} - P value: {}\n".format(n_r_value, n_p_value))
    f_n.write("Naive Average coverage: {}\n".format(np.average(naive_coverage_percentage)))
    f_n.write("---------------------------------------------\n")

    f_i.write("Test suite size: {}\n".format(test_suite_size))
    f_i.write("Improved Trajectory Coverage coverage R value: {} - P value: {}\n".format(i_r_value, i_p_value))
    f_i.write("Improved Average coverage: {}\n".format(np.average(improved_coverage_percentage)))
    f_i.write("---------------------------------------------\n")

    print("Naive Trajectory Coverage             R value: {} - P value: {}".format(n_r_value, n_p_value))
    print("Improved Trajectory Coverage coverage R value: {} - P value: {}".format(i_r_value, i_p_value))
    print("Naive Average coverage: {}".format(np.average(naive_coverage_percentage)))
    print("Improved Average coverage: {}".format(np.average(improved_coverage_percentage)))

    # Plot the results
    plt.figure("Naive")
    plt.scatter(naive_coverage_percentage, unique_crash_count, color="C{}".format(j), label="Size: {} - Correlation: {}".format(test_suite_size, n_r_value))

    plt.figure("Improved")
    plt.scatter(improved_coverage_percentage, unique_crash_count, color="C{}".format(j), label="Size: {} - Correlation: {}".format(test_suite_size, i_r_value))

    print("---------------------------------------------")
f_n.close()
f_i.close()


plt.figure("Naive")
plt.xlabel("Naive Trajectory Coverage (%)")
plt.ylabel("Unique Failures (%)")
plt.title("Naive Trajectory Coverage: {}".format(args.scenario))
plt.xlim([-5,100])
plt.ylim([-5,100])
plt.legend()

plt.figure("Improved")
plt.xlabel("Improved Trajectory Coverage (%)")
plt.ylabel("Unique Failures (%)")
plt.title("Improved Trajectory Coverage: {}".format(args.scenario))
plt.xlim([-5,100])
plt.ylim([-5,100])
plt.legend()

plt.show()