import sys
import math
import glob
import hashlib
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/analysis/research_questions")])
sys.path.append(base_directory)

import numpy as np

from tqdm import tqdm
from collections import Counter
from prettytable import PrettyTable

from utils.file_functions import get_beam_number_from_file
from utils.file_functions import order_files_by_beam_number
from utils.trajectory_coverage import unison_shuffled_copies
from utils.line_coverage_configuration import clean_branch_data
from utils.line_coverage_configuration import get_code_coverage
from utils.line_coverage_configuration import get_ignored_lines
from utils.line_coverage_configuration import get_ignored_branches

def compute_RRS_details():
    global traces
    global crashes

    # Get the total number of tests
    total_number_of_tests = traces.shape[0]

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                =  multiprocessing.Pool(processes=total_processors)

    # Call our function on each test in the trace
    jobs = []
    for random_test_index in range(total_number_of_tests):
        jobs.append(pool.apply_async(compute_trace_signature_and_crash, args=([random_test_index])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.full(total_number_of_tests, None, dtype="object")
    all_crash_detections = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        signature, crash_detected = r

        # Collect all the signatures
        all_signatures[i] = signature
        all_crash_detections[i] = crash_detected

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consistent
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Final signatures holds the list of all signatures
    # Count of signatures holds the list integers representing how many times each signature was seen

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count    = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count     = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    if np.size(consistent_class) <= 0:
        percentage_of_inconsistency = 0
    else:
        percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

    # Compute the number of tests in muticlass
    number_tests_in_multiclass = total_number_of_tests - single_test_signatures_count

    # Compute the average number of tests per class with more than 1 signatures
    if multi_test_signatures_count <= 0:
        average_number_multi_class_tests = np.inf
    else:
        average_number_multi_class_tests = np.round(number_tests_in_multiclass / multi_test_signatures_count, 1)

    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_number_multi_class_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]

def compute_line_coverage_details():
    global code_coverage_file_names

    # Get the total number of tests
    total_number_of_tests = len(code_coverage_file_names)

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                = multiprocessing.Pool(processes=total_processors)

    # Call our function on each file
    jobs = []
    for i in range(total_number_of_tests):
        jobs.append(pool.apply_async(compute_line_coverage_hash, args=([i])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.full(total_number_of_tests, None, dtype="object")
    all_crash_detections = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        signature, crash_detected = r

        # Collect all the signatures
        all_signatures[i] = signature
        all_crash_detections[i] = crash_detected

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consisten
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Final signatures holds the list of all signatures
    # Count of signatures holds the list intergers representing how many times each signature was seen

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)
    
    # Compute the number of tests in muticlass
    number_tests_in_multiclass = total_number_of_tests - single_test_signatures_count

    # Compute the average number of tests per class with more than 1 signatures
    if multi_test_signatures_count <= 0:
        average_number_multi_class_tests = np.inf
    else:
        average_number_multi_class_tests = np.round(number_tests_in_multiclass / multi_test_signatures_count, 1)

    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_number_multi_class_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]

def compute_branch_coverage_details(scenario):

    global code_coverage_file_names

    # Get the total number of tests
    total_number_of_tests = len(code_coverage_file_names)

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                = multiprocessing.Pool(processes=total_processors)

    # Call our function on each file
    jobs = []
    for i in range(total_number_of_tests):
        jobs.append(pool.apply_async(compute_branch_coverage_hash, args=([i, scenario])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.full(total_number_of_tests, None, dtype="object")
    all_crash_detections = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        signature, crash_detected = r

        # Collect all the signatures
        all_signatures[i] = signature
        all_crash_detections[i] = crash_detected

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consistent
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)
    
    # Compute the number of tests in muticlass
    number_tests_in_multiclass = total_number_of_tests - single_test_signatures_count

    # Compute the average number of tests per class with more than 1 signatures
    if multi_test_signatures_count <= 0:
        average_number_multi_class_tests = np.inf
    else:
        average_number_multi_class_tests = np.round(number_tests_in_multiclass / multi_test_signatures_count, 1)

    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_number_multi_class_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]

def compute_path_coverage_details(scenario, absolute=True, prime=True):

    if scenario == "highway":
        return [0, 0, 0, 0, 0, 0, 0]

    global code_coverage_file_names

    # Get the total number of tests
    total_number_of_tests = len(code_coverage_file_names)

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                = multiprocessing.Pool(processes=total_processors)

    # Call our function on each file
    jobs = []
    for i in range(total_number_of_tests):
        jobs.append(pool.apply_async(get_path_coverage_hash, args=([i, scenario])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.full(total_number_of_tests, None, dtype="object")
    all_crash_detections = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        intra_prime_path_signature, intra_path_signature, absolute_path_signature, crash_detected = r

        # Collect all the signatures
        if (absolute == True) and (prime == False):
            all_signatures[i] = absolute_path_signature
        elif (absolute == False) and (prime == False):
            all_signatures[i] = intra_path_signature
        elif (absolute == False) and (prime == True):
            all_signatures[i] = intra_prime_path_signature
        else:
            print("Error: Unknown combination of prime and absolute")
            exit()
        
        # Get the crash signatures
        all_crash_detections[i] = crash_detected

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consistent
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    if np.size(consistent_class) == 0:
        percentage_of_inconsistency = np.inf
    else:
        percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

    # Compute the number of tests in muticlass
    number_tests_in_multiclass = total_number_of_tests - single_test_signatures_count

    # Compute the average number of tests per class with more than 1 signatures
    if multi_test_signatures_count <= 0:
        average_number_multi_class_tests = np.inf
    else:
        average_number_multi_class_tests = np.round(number_tests_in_multiclass / multi_test_signatures_count, 1)
    
    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_number_multi_class_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]

def compute_trace_signature_and_crash(index):
    global traces
    global crashes

    # Get the trace and crash data
    trace = traces[index]
    crash = crashes[index]

    # Init the trace signature and crash detected variable 
    trace_signature  = set()
    crash_detected   = False

    # The signature for the trace is the set of all RRS signatures
    for sig in trace:
        trace_signature.add(tuple(sig))

    # Create the hash of the signature for each comparison
    trace_string = str(tuple(sorted(trace_signature)))
    trace_hash = hashlib.md5(trace_string.encode()).hexdigest()

    # Check if this trace had a crash
    crash_detected = not (crash == None).all()

    return [trace_hash, crash_detected]

def compute_line_coverage_hash(index):
    global code_coverage_file_names
    global ignored_lines

    coverage_hash = None
    number_of_crashes = None

    # Get the code coverage file
    code_coverage_file = code_coverage_file_names[index]

    # Get the coverage
    coverage_data = get_code_coverage(code_coverage_file)
    lines_covered = coverage_data[0]
    number_of_crashes = coverage_data[7]
    print

    # Make sure converting to a set was done correctly
    lines_covered_set = set(lines_covered)
    assert(len(lines_covered) == len(lines_covered_set))

    # Remove the ignored lines
    lines_covered_set -= ignored_lines

    # Sort the lines
    all_lines_coverage = sorted(list(lines_covered_set))

    # Get the coverage hash
    all_lines_string = str(tuple(all_lines_coverage))
    coverage_hash = hashlib.md5(all_lines_string.encode()).hexdigest()

    return [coverage_hash, number_of_crashes]

def compute_branch_coverage_hash(index, scenario):
    global code_coverage_file_names
    global ignored_branches

    coverage_hash = None
    number_of_crashes = None

    # Get the code coverage file
    code_coverage_file = code_coverage_file_names[index]
    
    # Get the coverage
    coverage_data = get_code_coverage(code_coverage_file)

    # Break the coverage up into its components
    branches_covered    = coverage_data[2]
    all_branches        = coverage_data[3]
    number_of_crashes   = coverage_data[7]

    # Make sure converting to a set was done correctly
    all_branches_set = set(all_branches)
    assert(len(all_branches) == len(all_branches_set))
    branches_covered_set = set(branches_covered)
    assert(len(branches_covered) == len(branches_covered_set))

    if scenario == "highway":
        # Clean the branch data
        all_branches_set_clean, branches_covered_set_clean = clean_branch_data(all_branches_set, branches_covered_set)
    else:
        all_branches_set_clean = all_branches_set
        branches_covered_set_clean = branches_covered_set

    # Remove the ignored lines
    branches_covered_set_clean -= ignored_branches

    # Sort the lines
    branches_covered_set_clean = sorted(list(branches_covered_set_clean))

    # Get the coverage hash
    branches_covered_string = str(tuple(branches_covered_set_clean))
    coverage_hash = hashlib.md5(branches_covered_string.encode()).hexdigest()

    return [coverage_hash, number_of_crashes]

def get_path_coverage_hash(index, scenario):
    global code_coverage_file_names

    coverage_hash = None
    number_of_crashes = None

    # Get the code coverage file
    code_coverage_file = code_coverage_file_names[index]

    # Get the coverage
    coverage_data = get_code_coverage(code_coverage_file)

    # Break the coverage up into its components
    intra_prime_path_signature  = coverage_data[4]
    intra_path_signature        = coverage_data[5]
    absolute_path_signature     = coverage_data[6]
    number_of_crashes           = coverage_data[7]

    return [intra_prime_path_signature, intra_path_signature, absolute_path_signature, number_of_crashes]

def compute_trajectory_coverage(vehicle_positions, crashes, total_number_of_tests):

    # Get all X Y and Z positions
    x_positions = vehicle_positions[:,:,0]
    y_positions = vehicle_positions[:,:,1]
    z_positions = vehicle_positions[:,:,2]

    # Get the min and max of the X and Y
    min_x = np.nanmin(x_positions)
    max_x = np.nanmax(x_positions)
    min_y = np.nanmin(y_positions)
    max_y = np.nanmax(y_positions)

    # For now use these as the drivable area
    drivable_x = [int(math.floor(min_x)), int(math.ceil(max_x))]
    drivable_y = [int(math.floor(min_y)), int(math.ceil(max_y))]

    # Compute the size of the drivable area
    x_size = drivable_x[1] - drivable_x[0]
    y_size = drivable_y[1] - drivable_y[0]

    # Create the coverage boolean array
    coverage_array = np.zeros((x_size, y_size), dtype=bool)
    coverage_denominator = float(x_size * y_size)

    # Remove all nans
    x_positions[np.isnan(x_positions)] = sys.maxsize
    y_positions[np.isnan(y_positions)] = sys.maxsize

    # Convert each position into an index
    index_x_array = np.round(x_positions, 0).astype(int) - drivable_x[0] -1
    index_y_array = np.round(y_positions, 0).astype(int) - drivable_y[0] -1

    # Create the final coverage array
    final_coverage = np.zeros(total_number_of_tests)

    # Declare the upper bound used to detect nan
    upper_bound = sys.maxsize - 1e5

    # Shuffle both arrays
    index_x_array, index_y_array = unison_shuffled_copies(index_x_array, index_y_array)

    # Turn all the signatures into a list
    all_signatures = np.full(total_number_of_tests, None, dtype="object")
    all_crash_detections = np.zeros(total_number_of_tests)

    # Loop through the data and mark off all that we have seen
    for i, coverage_index in tqdm(enumerate(zip(index_x_array, index_y_array)), total=len(index_x_array)):

        # Get the x and y index
        x_index = coverage_index[0]
        y_index = coverage_index[1]

        while ((abs(x_index[-1]) >= upper_bound) or (abs(y_index[-1]) >= upper_bound)):
            x_index = x_index[:-1]
            y_index = y_index[:-1]

        # The signature for the trace is the set of all RRS signatures
        position_list = []
        for j in range(len(x_index)):
            x = x_index[j]
            y = y_index[j]
            position_list.append(tuple((x, y)))

        # Make sure we remove any duplicates
        position_set = set(position_list)

        # Create the hash of the signature for each comparison
        position_string = str(tuple(sorted(position_list)))

        # Compute the hash
        position_hash = hashlib.md5(position_string.encode()).hexdigest()

        # Check if a crash was detected
        crash = crashes[i]
        crash_detected = not (crash == None).all()

        # Save the data
        all_signatures[i] = position_hash
        all_crash_detections[i] = crash_detected  


    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consistent
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Final signatures holds the list of all signatures
    # Count of signatures holds the list integers representing how many times each signature was seen

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count    = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count     = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    print("Here: {}".format(np.size(consistent_class)))
    if np.size(consistent_class) <= 0:
        percentage_of_inconsistency = np.inf
    else:
        percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

    # Compute the number of tests in muticlass
    number_tests_in_multiclass = total_number_of_tests - single_test_signatures_count

    # Compute the average number of tests per class with more than 1 signatures
    if multi_test_signatures_count <= 0:
        average_number_multi_class_tests = np.inf
    else:
        average_number_multi_class_tests = np.round(number_tests_in_multiclass / multi_test_signatures_count, 1)
    

    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_number_multi_class_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',       type=str, default="/mnt/extradrive3/PhysicalCoverageData",    help="The location and name of the datafolder")
parser.add_argument('--number_of_tests', type=int, default=-1,                                              help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',    type=str, default="",                                              help="center_close/center_full")
parser.add_argument('--scenario',        type=str, default="",                                              help="beamng/highway")
parser.add_argument('--cores',           type=int, default=4,                                               help="number of available cores")
args = parser.parse_args()

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = "*.npy"

# Checking the distribution
if not (args.distribution == "center_close" or args.distribution == "center_full"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '{}/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.data_path, args.scenario, args.distribution, args.number_of_tests)
trace_file_names        = glob.glob(base_path + "traces_*.npy")
crash_file_names        = glob.glob(base_path + "crash_*.npy")
position_file_names     = glob.glob(base_path + "ego_positions_*")

# Get the code coverage
base_path = '{}/{}/random_tests/code_coverage/processed/{}/'.format(args.data_path, args.scenario, args.number_of_tests)
global code_coverage_file_names
code_coverage_file_names = glob.glob(base_path + "*/*.txt")

# Make sure we have enough samples
assert(len(trace_file_names) >= 1)
assert(len(crash_file_names) >= 1)
assert(len(position_file_names) >= 1)
assert(len(code_coverage_file_names) >= 1)

# Load the vehicle positions and crashes - Select one of the files (they are all the same)
for file_name in position_file_names:
    if "_b1_" in file_name:
        break

# Load the vehicle positions
vehicle_positions = np.load(file_name)

# Load the vehicle positions and crashes - Select one of the files (they are all the same)
for file_name in crash_file_names:
    if "_b1_" in file_name:
        break

# Load the crashes
crashes_traj = np.load(file_name, allow_pickle=True)

# Select args.number_of_tests total code coverage files
assert(len(code_coverage_file_names) == args.number_of_tests)

global ignored_lines
global ignored_branches
ignored_lines       = set(get_ignored_lines(args.scenario))
ignored_branches    = set(get_ignored_branches(args.scenario))

# Get the beam numbers
trace_beam_numbers = get_beam_number_from_file(trace_file_names)
crash_beam_numbers = get_beam_number_from_file(crash_file_names)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(trace_beam_numbers) | set(crash_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
trace_file_names = order_files_by_beam_number(trace_file_names, beam_numbers)
crash_file_names = order_files_by_beam_number(crash_file_names, beam_numbers)

# Create the output table
t = PrettyTable()
t.field_names = ["Coverage Type", "Total classes" , "Single Test classes", "Multitest classes", "Avg # tests per mutitest class", "Consistent multitest classes", "Inconsistent multitest classes", "Percentage inconsistent"]

# Create a list to hold the latex
latex_list = []

# Compute the line coverage details
print("\nProcessing Line Coverage")
results                         = compute_line_coverage_details()
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
average_num_multiclass_tests    = results[3]
consistent_class_count          = results[4]
inconsistent_class_count        = results[5]
percentage_of_inconsistency     = results[6]
print("Total classes: {}".format(total_signatures_count))
print("Total single test classes: {}".format(single_test_signatures_count))
print("Total multi test classes: {}".format(multi_test_signatures_count))
print("Avg # tests per mutitest class: {}".format(average_num_multiclass_tests))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Line Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])
latex_list.append([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])

# Compute the branch coverage details
print("\nProcessing Branch Coverage")
results                         = compute_branch_coverage_details(args.scenario)
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
average_num_multiclass_tests    = results[3]
consistent_class_count          = results[4]
inconsistent_class_count        = results[5]
percentage_of_inconsistency     = results[6]
print("Total classes: {}".format(total_signatures_count))
print("Total single test classes: {}".format(single_test_signatures_count))
print("Total multi test classes: {}".format(multi_test_signatures_count))
print("Avg # tests per mutitest class: {}".format(average_num_multiclass_tests))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Branch Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])
latex_list.append([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])


# Compute the intraprocedural Path coverage details
print("\nProcessing Introprocedural Prime Path Coverage")
results                         = compute_path_coverage_details(args.scenario, absolute=False, prime=True)
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
average_num_multiclass_tests    = results[3]
consistent_class_count          = results[4]
inconsistent_class_count        = results[5]
percentage_of_inconsistency     = results[6]
print("Total classes: {}".format(total_signatures_count))
print("Total single test classes: {}".format(single_test_signatures_count))
print("Total multi test classes: {}".format(multi_test_signatures_count))
print("Avg # tests per mutitest class: {}".format(average_num_multiclass_tests))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Intraprocedural Prime Path Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])
latex_list.append([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])

print("\nProcessing Introprocedural Path Coverage")
results                         = compute_path_coverage_details(args.scenario, absolute=False, prime=False)
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
average_num_multiclass_tests    = results[3]
consistent_class_count          = results[4]
inconsistent_class_count        = results[5]
percentage_of_inconsistency     = results[6]
print("Total classes: {}".format(total_signatures_count))
print("Total single test classes: {}".format(single_test_signatures_count))
print("Total multi test classes: {}".format(multi_test_signatures_count))
print("Avg # tests per mutitest class: {}".format(average_num_multiclass_tests))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Intraprocedural Path Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])
latex_list.append([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])

# Compute the Absolute Path coverage details
print("\nProcessing Absolute Path Coverage")
results                         = compute_path_coverage_details(args.scenario, absolute=True, prime=False)
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
average_num_multiclass_tests    = results[3]
consistent_class_count          = results[4]
inconsistent_class_count        = results[5]
percentage_of_inconsistency     = results[6]
print("Total classes: {}".format(total_signatures_count))
print("Total single test classes: {}".format(single_test_signatures_count))
print("Total multi test classes: {}".format(multi_test_signatures_count))
print("Avg # tests per mutitest class: {}".format(average_num_multiclass_tests))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Absolute Path Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])
latex_list.append([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])


# Compute the Trajectory Path coverage details
print("\nProcessing Trajectory Coverage")
results                         = compute_trajectory_coverage(vehicle_positions, crashes_traj, args.number_of_tests)
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
average_num_multiclass_tests    = results[3]
consistent_class_count          = results[4]
inconsistent_class_count        = results[5]
percentage_of_inconsistency     = results[6]
print("Total classes: {}".format(total_signatures_count))
print("Total single test classes: {}".format(single_test_signatures_count))
print("Total multi test classes: {}".format(multi_test_signatures_count))
print("Avg # tests per mutitest class: {}".format(average_num_multiclass_tests))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Trajectory Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])
latex_list.append([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])





# Loop through each of the files and compute both an RRS signature as well as determine if there was a crash
for beam_number in beam_numbers:
    print("\nProcessing RRS{}".format(beam_number))
    key = "RRS{}".format(beam_number)

    # Get the trace and crash files
    global traces
    traces  = np.load(trace_file_names[beam_number-1])
    global crashes
    crashes = np.load(crash_file_names[beam_number-1], allow_pickle=True)

    # Compute the different metrics
    results                         = compute_RRS_details()
    total_signatures_count          = results[0]
    single_test_signatures_count    = results[1]
    multi_test_signatures_count     = results[2]
    average_num_multiclass_tests    = results[3]
    consistent_class_count          = results[4]
    inconsistent_class_count        = results[5]
    percentage_of_inconsistency     = results[6]
    print("Total signatures: {}".format(total_signatures_count))
    print("Total single test signatures: {}".format(single_test_signatures_count))
    print("Total multi test signatures: {}".format(multi_test_signatures_count))
    print("Total consistent classes: {}".format(consistent_class_count))
    print("Total inconsistent classes: {}".format(inconsistent_class_count))
    print("Avg # tests per mutitest class: {}".format(average_num_multiclass_tests))
    print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
    t.add_row([key, total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])
    latex_list.append([total_signatures_count, single_test_signatures_count, multi_test_signatures_count, average_num_multiclass_tests, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency])

# Display the table
print("")
print("Scenario: {}".format(args.scenario))
print("Number of tests: {}".format(args.number_of_tests))
print("Distribution: {}".format(args.distribution))
print("")
print(t)



print("\n\nLatex\n")
for line in latex_list:
    print_line = ""
    for item in line[:-1]:
        print_line += "& {} ".format(item)
    print_line += "& {}\\%".format(line[-1])
    print(print_line)
