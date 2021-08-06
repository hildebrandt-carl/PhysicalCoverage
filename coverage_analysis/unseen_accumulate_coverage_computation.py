import os
import glob
import pickle
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from datetime import datetime

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics


def unique_vector_config(scenario, number_of_seconds):
    if scenario == "highway":
        hash_size = 4 * number_of_seconds
    elif scenario == "beamng":
        hash_size = 4 * number_of_seconds
    else:
        exit()
    return hash_size

def crash_hasher(trace, hash_size):
    # Used to hold the last vectors before a crash
    last_seen_vectors = np.zeros((hash_size, trace[0].shape[0]))

    # Create the hash
    hash_value = np.nan

    if not np.isnan(trace).any():
        return [np.nan]

    # For each vector in the trace
    for i in range(trace.shape[0]):
        # Get the vector
        v = trace[i]

        # Check if there was a crash
        if np.isnan(v).any():
            hash_value = hash(tuple(last_seen_vectors.reshape(-1)))
            break
        # There wasn't a crash
        else:
            # Roll the data in the last_seen_vectors
            last_seen_vectors = np.roll(last_seen_vectors, v.shape[0])
            # Save the data to the last_seen_vectors
            last_seen_vectors[0] = v

    return [hash_value]

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def compute_accumulative_coverage(traces, vehicles, print_position, scenario, total_beams, feasible_vectors):

    assert(len(traces) == len(vehicles))

    # Time how long this operation takes
    start = datetime.now()

    # Count the number of unique vectors seen for the combined data
    traces_count        = traces.shape[0]
    crash_count         = 0
    unique_crash_count  = 0
    total_vectors       = 0

    # Turn the feasible vectors into a set
    feasible_vectors_set = set()
    previous_size = 0
    for vector in feasible_vectors:
        feasible_vectors_set.add(tuple(vector))
        if previous_size >= len(feasible_vectors_set):
            print("Something is wrong with your feasible vector set, as it contains duplicates")
            exit()
        previous_size = len(feasible_vectors_set)

    # Count the number of unique vectors seen for each of the different combinations of data
    unique_vectors_seen_set                 = set()
    unique_feasible_vector_seen_set         = set()
    unique_crashes_seen_set                 = set()
    coverage_accumulative_graph             = np.full(traces_count, np.nan)
    feasible_coverage_accumulative_graph    = np.full(traces_count, np.nan) 
    coverage_vehicle_count                  = np.full(traces_count, np.nan)
    coverage_crash_count                    = np.full(traces_count, np.nan)
    coverage_crash_count_unique             = np.full(traces_count, np.nan)

    # For each of the traces
    for i in tqdm(range(traces_count), position=int(print_position), mininterval=5):
        # Get the trace
        trace = traces[i]
        vehicle_count = vehicles[i]
        
        # See if there was a crash
        if np.isnan(trace).any():
            crash_count += 1

            # Check if the crash is unique
            hash_size = unique_vector_config(scenario, number_of_seconds=0.5)
            crash_hash = crash_hasher(trace, hash_size)
            l = len(unique_crashes_seen_set)
            assert(len(crash_hash) == 1)
            unique_crashes_seen_set.add(crash_hash[0])
            if l != len(unique_crashes_seen_set):
                unique_crash_count += 1

        # For each vector in the trace
        for vector in trace:
            # If this vector contains nan it means it crashed (and so we can ignore it, this traces crash was already counted)
            if np.isnan(vector).any():
                continue

            # If this vector contains inf it means that the trace was extended to match sizes with the maximum sized trace
            if np.isinf(vector).any():
                continue

            # If this vector contains any -1 it means it needed to be expanded to fit
            if np.all(vector==-1):
                continue

            # Count the traces
            total_vectors += 1
            
            # Check if it is unique
            unique_vectors_seen_set.add(tuple(vector))

            # Check if any of the vectors are infeasible
            if tuple(vector) not in feasible_vectors_set:
                continue

            unique_feasible_vector_seen_set.add(tuple(vector))
                
        # Used for the accumulative graph
        unique_vector_count                          = len(unique_vectors_seen_set)
        unique_feasible_vector_count                 = len(unique_feasible_vector_seen_set)
        coverage_accumulative_graph[i]               = unique_vector_count
        feasible_coverage_accumulative_graph[i]      = unique_feasible_vector_count
        coverage_vehicle_count[i]                    = vehicle_count
        coverage_crash_count[i]                      = crash_count
        coverage_crash_count_unique[i]               = unique_crash_count


    # Convert the set back to a list
    unique_vectors_seen = []
    for vec in unique_vectors_seen_set:
        unique_vectors_seen.append(np.array(vec))

    unique_feasible_vector_seen = []
    for vec in unique_feasible_vector_seen_set:
        unique_feasible_vector_seen.append(np.array(vec))

    # End the timer
    end = datetime.now()
    time_taken = end - start

    # Return the data
    return time_taken, traces_count, crash_count, total_vectors, unique_vectors_seen, unique_vector_count, unique_feasible_vector_seen, unique_feasible_vector_count, coverage_accumulative_graph, feasible_coverage_accumulative_graph, coverage_vehicle_count, coverage_crash_count, coverage_crash_count_unique

def compute_coverage(save_name, return_dict, return_key):

    # Load the data
    data = load_obj(save_name)

    # Unpack the data
    combined_traces     = data[0]
    original_traces     = data[1]
    unseen_traces       = data[2]
    combined_vehicles   = data[3]
    original_vehicles   = data[4]
    unseen_vehicles     = data[5]
    total_beams         = int(data[6])
    load_name           = data[7]
    scenario            = data[8]

    assert(len(combined_vehicles) == len(combined_traces))
    assert(len(unseen_vehicles) == len(unseen_traces))
    assert(len(original_vehicles) == len(original_traces))

    print("Processing beam count{}: ".format(total_beams))

    # Compute total possible values using the above
    unique_observations_per_cell = (new_max_distance / float(new_accuracy))
    total_possible_observations = int(pow(unique_observations_per_cell, total_beams))
    
    # Sort the data based on the number of vehicles per test
    combined_indices = combined_vehicles.argsort()
    original_indices = original_vehicles.argsort()
    unseen_indices = unseen_vehicles.argsort()

    # Sort the data using the new indices
    combined_traces     = combined_traces[combined_indices]
    combined_vehicles   = combined_vehicles[combined_indices]
    original_traces     = original_traces[original_indices]
    original_vehicles   = original_vehicles[original_indices]
    unseen_traces       = unseen_traces[unseen_indices]
    unseen_vehicles     = unseen_vehicles[unseen_indices]

    # Load the feasible trajectories
    fname = '../../PhysicalCoverageData/' + str(scenario) +'/feasibility/processed/FeasibleVectors_b' + str(total_beams) + ".npy"
    feasible_vectors = list(np.load(fname))
    total_feasible_observations = np.shape(feasible_vectors)[0]

    # Compute on all the different combinations of data
    c_time, c_trace_count, c_crash_count, c_total_vectors, c_unique_vectors, c_unique_vectors_count, c_unique_feasible_vectors_seen, c_unique_feasible_vector_count, c_coverage_accumulative_graph, c_feasible_coverage_accumulative_graph, c_coverage_vehicle_count, c_crash_array, c_unique_crash_array = compute_accumulative_coverage(combined_traces, combined_vehicles, return_key[1:], scenario, total_beams, feasible_vectors)
    o_time, o_trace_count, o_crash_count, o_total_vectors, o_unique_vectors, o_unique_vectors_count, o_unique_feasible_vectors_seen, o_unique_feasible_vector_count, o_coverage_accumulative_graph, o_feasible_coverage_accumulative_graph, o_coverage_vehicle_count, o_crash_array, o_unique_crash_array = compute_accumulative_coverage(original_traces, original_vehicles, return_key[1:], scenario, total_beams, feasible_vectors)
    u_time, u_trace_count, u_crash_count, u_total_vectors, u_unique_vectors, u_unique_vectors_count, u_unique_feasible_vectors_seen, u_unique_feasible_vector_count, u_coverage_accumulative_graph, u_feasible_coverage_accumulative_graph, u_coverage_vehicle_count, u_crash_array, u_unique_crash_array = compute_accumulative_coverage(unseen_traces, unseen_vehicles, return_key[1:], scenario, total_beams, feasible_vectors)

    # Compute crash percentages
    c_crash_percentage = round(c_crash_count / float(c_total_vectors) * 100, 4)
    o_crash_percentage = round(o_crash_count / float(o_total_vectors) * 100, 4)
    u_crash_percentage = round(u_crash_count / float(u_total_vectors) * 100, 4)
    
    # Compute coverage
    c_feasible_coverage = round((c_unique_feasible_vector_count / float(total_feasible_observations)) * 100, 4)
    c_possible_coverage = round((c_unique_vectors_count / float(total_possible_observations)) * 100, 4)
    o_feasible_coverage = round((o_unique_feasible_vector_count / float(total_feasible_observations)) * 100, 4)
    o_possible_coverage = round((o_unique_vectors_count / float(total_possible_observations)) * 100, 4)
    u_feasible_coverage = round((u_unique_feasible_vector_count / float(total_feasible_observations)) * 100, 4)
    u_possible_coverage = round((u_unique_vectors_count / float(total_possible_observations)) * 100, 4)

    print("\n\n\n\n\n\n")
    print("Filename:\t\t\t" + str(load_name))
    print("-------------------------------------------------------")
    print("Combined data")
    print("-------------------------------------------------------")
    print("Total traces considered:\t"      + str(c_trace_count))
    print("Total crashes:\t\t\t"            + str(c_crash_count))
    print("Crash percentage:\t\t"           + str(c_crash_percentage) + "%")
    print("Total vectors considered:\t"     + str(c_total_vectors))
    print("Total unique vectors seen:\t"    + str(c_unique_vectors_count))
    print("Total feasible vectors:\t\t"     + str(total_feasible_observations))
    print("Total feasible coverage:\t"      + str(c_feasible_coverage) + "%")
    print("Total possible vectors:\t\t"     + str(total_possible_observations))
    print("Total possible coverage:\t"      + str(c_possible_coverage) + "%")
    print("Total time to compute:\t\t"      + str(c_time))
    print("-------------------------------------------------------")
    print("Original data")
    print("-------------------------------------------------------")
    print("Unseen data")
    print("Total traces considered:\t"      + str(o_trace_count))
    print("Total crashes:\t\t\t"            + str(o_crash_count))
    print("Crash percentage:\t\t"           + str(o_crash_percentage) + "%")
    print("Total vectors considered:\t"     + str(o_total_vectors))
    print("Total unique vectors seen:\t"    + str(o_unique_vectors_count))
    print("Total feasible vectors:\t\t"     + str(total_feasible_observations))
    print("Total feasible coverage:\t"      + str(o_feasible_coverage) + "%")
    print("Total possible vectors:\t\t"     + str(total_possible_observations))
    print("Total possible coverage:\t"      + str(o_possible_coverage) + "%")
    print("Total time to compute:\t\t"      + str(o_time))
    print("-------------------------------------------------------")
    print("Unseen New data")
    print("-------------------------------------------------------")
    print("Total traces considered:\t"      + str(u_trace_count))
    print("Total crashes:\t\t\t"            + str(u_crash_count))
    print("Crash percentage:\t\t"           + str(u_crash_percentage) + "%")
    print("Total vectors considered:\t"     + str(u_total_vectors))
    print("Total unique vectors seen:\t"    + str(u_unique_vectors_count))
    print("Total feasible vectors:\t\t"     + str(total_feasible_observations))
    print("Total feasible coverage:\t"      + str(u_feasible_coverage) + "%")
    print("Total possible vectors:\t\t"     + str(total_possible_observations))
    print("Total possible coverage:\t"      + str(u_possible_coverage) + "%")
    print("Total time to compute:\t\t"      + str(u_time))
    print("Original data")

    # Get all the unique number of external vehicles
    unique_vehicle_count = list(set(c_coverage_vehicle_count))
    unique_vehicle_count.sort()

    # Convert the data to percentages
    c_accumulative_graph_possible_coverage = (c_coverage_accumulative_graph / total_possible_observations) * 100
    c_accumulative_graph_feasible_coverage = (c_feasible_coverage_accumulative_graph / total_feasible_observations) * 100
    o_accumulative_graph_possible_coverage = (o_coverage_accumulative_graph / total_possible_observations) * 100
    o_accumulative_graph_feasible_coverage = (o_feasible_coverage_accumulative_graph / total_feasible_observations) * 100
    u_accumulative_graph_possible_coverage = (u_coverage_accumulative_graph / total_possible_observations) * 100
    u_accumulative_graph_feasible_coverage = (u_feasible_coverage_accumulative_graph / total_feasible_observations) * 100

    # Get the data ready for return
    combined_data   = [c_accumulative_graph_possible_coverage, c_accumulative_graph_feasible_coverage, c_coverage_vehicle_count, c_unique_vectors, c_crash_array, c_unique_crash_array]
    original_data   = [o_accumulative_graph_possible_coverage, o_accumulative_graph_feasible_coverage, o_coverage_vehicle_count, o_unique_vectors, o_crash_array, o_unique_crash_array]
    unseen_data     = [u_accumulative_graph_possible_coverage, u_accumulative_graph_feasible_coverage, u_coverage_vehicle_count, u_unique_vectors, u_crash_array, u_unique_crash_array]
    misc_data       = [total_possible_observations, feasible_vectors]

    # Return both arrays
    return_dict[return_key] = [combined_data, original_data, unseen_data, misc_data, total_beams]
    return True

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
RSR = RSRConfig()

# Save the kinematics and RSR parameters
new_steering_angle  = HK.steering_angle
new_max_distance    = HK.max_velocity
new_accuracy        = RSR.accuracy

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
load_name += "_t" + str("*")
load_name += ".npy"

# Get the file names
original_data_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/processed/' + str(args.total_samples) + "/"
unseen_data_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/unseen/' + str(args.total_samples) + "/processed/"

original_trace_files = glob.glob(original_data_path + "traces_" + args.scenario + load_name)
unseen_trace_files = glob.glob(unseen_data_path + "traces_" + args.scenario + "*.npy")

original_file_names = []
for f in original_trace_files:
    name = f.replace(original_data_path + "traces", "")
    original_file_names.append(name)

unseen_file_names = []
for f in unseen_trace_files:
    name = f.replace(unseen_data_path + "traces", "")
    unseen_file_names.append(name)

# Sort the names
original_file_names.sort()
unseen_file_names.sort()
print("Original Files: " + str(original_file_names))
print("Unseen Files: " + str(unseen_file_names))
print("Loading Complete")

print("----------------------------------")
print("---------Merging Data-------------")
print("----------------------------------")

# Unseen file names
new_data = []
for i in range(len(unseen_file_names)):
    # Get the file name
    file_name = unseen_file_names[i]

    # Get the beam number
    l = file_name.find("_b") + 2
    beam_number = file_name[l:]
    r = beam_number.find("_")
    beam_number = beam_number[:r]

    # Find the corresponding original file
    original_file = ""
    for j in range(len(original_file_names)):
        ori_file_name = original_file_names[j]
        if "_b{}_".format(beam_number) in ori_file_name:
            original_file = ori_file_name
            break

    load_name = "Merging: {} and {}".format(file_name, original_file)

    # Load the traces
    o_traces = np.load(original_data_path + "traces" + original_file)
    u_traces = np.load(unseen_data_path + "traces" + file_name)
    
    print("Original trace shape: {}".format(np.shape(o_traces)))
    print("Unseen trace shape: {}".format(np.shape(u_traces)))

    # Make sure that the b trace fits into trace a
    size_difference = o_traces.shape[1] - u_traces.shape[1]
    if size_difference > 0:
        filler = np.full((u_traces.shape[0], size_difference, u_traces.shape[2]), -1, dtype='float64')
        print("Correcting unseen dimensions using shape: {}".format(np.shape(filler)))
        u_traces = np.hstack([u_traces, filler])
    elif size_difference < 0:
        filler = np.full((o_traces.shape[0], -1 * size_difference, o_traces.shape[2]), -1, dtype='float64')
        print("Correcting unseen dimensions using shape: {}".format(np.shape(filler)))
        o_traces = np.hstack([o_traces, filler])
        
    # Load the traces
    o_veh = np.load(original_data_path + "vehicles" + original_file)
    u_veh = np.load(unseen_data_path + "vehicles" + file_name)

    # DELETE
    if not os.path.exists('combined_data/'):
        os.makedirs('combined_data/')
    combined_trace_data = np.concatenate([o_traces, u_traces], axis=0)
    combined_vehicle_data = np.concatenate([o_veh, u_veh], axis=0)
    save_name = args.scenario
    save_name += "_s" + str(new_steering_angle) 
    save_name += "_b" + str(beam_number) 
    save_name += "_d" + str(new_max_distance) 
    save_name += "_a" + str(new_accuracy)
    save_name += "_t" + str(np.shape(combined_trace_data)[0])
    save_name += ".npy"
    np.save("combined_data/traces_{}".format(save_name), combined_trace_data)
    np.save("combined_data/vehicles_{}".format(save_name), combined_vehicle_data)

    # Add 1000 to the number of vehicles so its easy to identify which are the new tests.
    # Thus the vehicle count will now be 1001, 1002, etc for the unseen tests, making them get appended to the end of the set
    u_veh = u_veh + 1000

    # Create the new data
    combined_trace_data = np.concatenate([o_traces, u_traces], axis=0)
    combined_vehicle_data = np.concatenate([o_veh, u_veh], axis=0)
    
    print("Combined trace shape: {}".format(np.shape(combined_trace_data)))
    print("Combined veh shape: {}".format(np.shape(combined_vehicle_data)))
    print("")

    # Save the new data
    new_data.append([combined_trace_data, o_traces, u_traces, combined_vehicle_data, o_veh, u_veh, beam_number, load_name, args.scenario])

print("----------------------------------")
print("-------Computing Coverage---------")
print("----------------------------------")

manager = multiprocessing.Manager()
unseen_return_dict = manager.dict()

# Create a pool with x processes
total_processors = int(args.cores)
pool =  multiprocessing.Pool(total_processors)

# Call our function total_test_suites times
unseen_result_object = []

for index in range(len(new_data)):

    # Get the data
    data = new_data[index]

    # Save the data so that it can be loaded by the async function
    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')

    save_name = 'tmp/p' + str(index) + "_" + str(args.total_samples)
    save_obj(data, save_name )

    # Get the file name and the return key
    return_key = 'p' + str(index)

    unseen_result_object.append(pool.apply_async(compute_coverage, args=(save_name, unseen_return_dict, return_key)))

# Get the results (results are actually stored in return_dict)
unseen_results = [r.get() for r in unseen_result_object]
unseen_results = np.array(unseen_results)

# Convert the return_dict results to a normal dictionary
unseen_final_results = {}
for key in unseen_return_dict.keys():
    unseen_final_results[key] = list(unseen_return_dict[key])

# Close the pool
pool.close()

# Save the results
save_name = "../results/rq5_" + args.scenario
save_obj(unseen_final_results, save_name)
return_dict = load_obj(save_name)

# Run the plotting code
exec(compile(open("unseen_accumulate_coverage_plot.py", "rb").read(), "unseen_accumulate_coverage_plot.py", 'exec'))