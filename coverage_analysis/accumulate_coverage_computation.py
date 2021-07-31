import glob
import pickle
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from datetime import datetime

from environment_configurations import RSRConfig
from environment_configurations import HighwayKinematics


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def compute_coverage(load_name, return_dict, return_key, base_path):
    
    # Get the current time
    start=datetime.now()
    total_beams = load_name[load_name.find("_")+1:]
    total_beams = total_beams[total_beams.find("_")+1:]
    total_beams = total_beams[total_beams.find("_b")+2:]
    total_beams = total_beams[0:total_beams.find("_d")]
    total_beams = int(total_beams)

    scenario = load_name[1:]
    scenario = scenario[:scenario.find("_")]

    # Compute total possible values using the above
    unique_observations_per_cell = (new_max_distance / float(new_accuracy))
    total_possible_observations = int(pow(unique_observations_per_cell, total_beams))

    print("Processing: " + load_name)
    traces = np.load(base_path + "traces" + load_name)
    vehicles = np.load(base_path + "vehicles" + load_name)

    # Sort the data based on the number of vehicles per test
    vehicles_indices = vehicles.argsort()
    traces = traces[vehicles_indices]
    vehicles = vehicles[vehicles_indices]

    total_traces = traces.shape[0]
    total_crashes = 0
    total_vectors = 0

    # Load the feasible trajectories
    fname = '../../PhysicalCoverageData/' + str(scenario) +'/feasibility/processed/FeasibleVectors_b' + str(total_beams) + ".npy"
    feasible_vectors = list(np.load(fname))

    # Turn the feasible_vectors into a set
    feasible_vectors_set = set()
    previous_size = 0
    for vector in feasible_vectors:
        feasible_vectors_set.add(tuple(vector))
        if previous_size >= len(feasible_vectors_set):
            print("Something is wrong with your feasible vector set, as it contains duplicates")
            exit()
        previous_size = len(feasible_vectors_set)

    total_feasible_observations = len(feasible_vectors_set)

    unique_vectors_seen                 = set()
    accumulative_graph                  = np.full(total_traces, np.nan)
    accumulative_graph_vehicle_count    = np.full(total_traces, np.nan)

    # For each of the traces
    for i in tqdm(range(total_traces), position=int(return_key[1:]), mininterval=5):
        # Get the trace
        trace = traces[i]
        vehicle_count = vehicles[i]
        
        # See if there was a crash
        if np.isnan(trace).any():
            total_crashes += 1

        # For each vector in the trace
        for vector in trace:
            # If this vector contains nan it means it crashed (and so we can ignore it, this traces crash was already counted)
            if np.isnan(vector).any():
                continue

            # If this vector contains inf it means that the trace was extended to match sizes with the maximum sized trace
            if np.isinf(vector).any():
                continue

            # Count the traces
            total_vectors += 1
            
            # Check if it is unique
            unique_vectors_seen.add(tuple(vector))

            # Check if any of the vectors are infeasible
            l = len(feasible_vectors_set)
            feasible_vectors_set.add(tuple(vector))
            # Added therefor it must be infeasible
            if l != len(feasible_vectors_set):
                print("Error - Infeasible vector found: {}".format(vector))

        # Used for the accumulative graph
        unique_vector_length_count          = len(unique_vectors_seen)
        accumulative_graph[i]               = unique_vector_length_count
        accumulative_graph_vehicle_count[i] = vehicle_count

    print("Total observations: {}".format(len(unique_vectors_seen)))
    print("Total feasible observations: {}".format(len(feasible_vectors_set)))

    feasible_coverage = round((unique_vector_length_count / float(total_feasible_observations)) * 100, 4)
    possible_coverage = round((unique_vector_length_count / float(total_possible_observations)) * 100, 4)
    crash_percentage = round(total_crashes / float(total_traces) * 100, 4)

    print("\n\n\n\n\n\n")
    print("Filename:\t\t\t" + load_name)
    print("Total traces considered:\t" + str(total_traces))
    print("Total crashes:\t\t\t" + str(total_crashes))
    print("Crash percentage:\t\t" + str(crash_percentage) + "%")
    print("Total vectors considered:\t" + str(total_vectors))
    print("Total unique vectors seen:\t" + str(unique_vector_length_count))
    print("Total feasible vectors:\t\t" + str(total_feasible_observations))
    print("Total feasible coverage:\t" + str(feasible_coverage) + "%")
    print("Total possible vectors:\t\t" + str(total_possible_observations))
    print("Total possible coverage:\t" + str(possible_coverage) + "%")
    print("Total time to compute:\t\t" + str(datetime.now()-start))

    # Get all the unique number of external vehicles
    unique_vehicle_count = list(set(accumulative_graph_vehicle_count))
    unique_vehicle_count.sort()

    # Convert to a percentage
    accumulative_graph_possible_coverage = (accumulative_graph / total_possible_observations) * 100
    accumulative_graph_feasible_coverage = (accumulative_graph / total_feasible_observations) * 100

    # Return both arrays
    return_dict[return_key] = [accumulative_graph_possible_coverage, accumulative_graph_feasible_coverage, accumulative_graph_vehicle_count, total_beams]
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
load_name += "_t" + str(args.total_samples)
load_name += ".npy"

# Get the file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/processed/' + str(args.total_samples) + "/"
trace_file_names = glob.glob(base_path + "traces_" + args.scenario + load_name)
file_names = []
for f in trace_file_names:
    name = f.replace(base_path + "traces", "")
    file_names.append(name)

# Sort the names
file_names.sort()
print("Files: " + str(file_names))
print("Loading Complete")

print("----------------------------------")
print("-------Computing Coverage---------")
print("----------------------------------")

manager = multiprocessing.Manager()
return_dict = manager.dict()

# Create a pool with x processes
total_processors = int(args.cores)
pool =  multiprocessing.Pool(total_processors)

# Call our function total_test_suites times
result_object = []
for file_index in range(len(file_names)):

    # Get the file name and the return key
    file_name = file_names[file_index]
    return_key = 'p' + str(file_index)

    result_object.append(pool.apply_async(compute_coverage, args=(file_name, return_dict, return_key, base_path)))

# Get the results (results are actually stored in return_dict)
results = [r.get() for r in result_object]
results = np.array(results)

# Convert the return_dict results to a normal dictionary
final_results = {}
for key in return_dict.keys():
    final_results[key] = list(return_dict[key])

# Close the pool
pool.close()

# Save the results
save_name = "../results/rq1_" + args.scenario
save_obj(final_results, save_name)
return_dict = load_obj(save_name)

# Run the plotting code
exec(compile(open("accumulate_coverage_plot.py", "rb").read(), "accumulate_coverage_plot.py", 'exec'))