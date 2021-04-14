import glob
import pickle
import argparse
import multiprocessing

import numpy as np

from tqdm import tqdm
from datetime import datetime


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def isUnique(vector, unique_vectors_seen):
    # Return false if the vector contains Nan
    if np.isnan(vector).any():
        return False
    # Assume True
    unique = True
    for v2 in unique_vectors_seen:
        # If we have seen this vector break out of this loop
        if np.array_equal(vector, v2):
            unique = False
            break
    return unique

def compute_coverage(load_name, return_dict, return_key, base_path):
    
    # Get the current time
    start=datetime.now()
    total_beams = load_name[load_name.find("_")+1:]
    total_beams = total_beams[total_beams.find("_")+1:]
    total_beams = total_beams[total_beams.find("_b")+2:]
    total_beams = total_beams[0:total_beams.find("_d")]
    total_beams = int(total_beams)

    # Compute total possible values using the above
    unique_observations_per_cell = (new_max_distance / float(new_accuracy)) + 1.0
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

    unique_vectors_seen                 = []
    accumulative_graph                  = np.full(total_traces, np.nan)
    acuumulative_graph_vehicle_count    = np.full(total_traces, np.nan)

    # For each of the traces
    for i in tqdm(range(total_traces), position=int(return_key[1:]), mininterval=5):
        # Get the trace
        trace = traces[i]
        vehicle_count = vehicles[i]
        
        # See if there was a crash
        if np.isnan(trace).any():
            total_crashes += 1

        # For each vector in the trace
        for v in trace:
            # If this vector does not have any nan
            if not np.isnan(v).any():
                # Count it
                total_vectors += 1
                # Check if it is unique
                unique = isUnique(v, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(v)

        # Used for the accumulative graph
        unique_vector_length_count          = len(unique_vectors_seen)
        accumulative_graph[i]               = unique_vector_length_count
        acuumulative_graph_vehicle_count[i] = vehicle_count

    overall_coverage = round((unique_vector_length_count / float(total_possible_observations)) * 100, 4)
    crash_percentage = round(total_crashes / float(total_traces) * 100, 4)

    print("\n\n\n\n\n\n")
    print("Filename:\t\t\t" + load_name)
    print("Total traces considered:\t" + str(total_vectors))
    print("Total crashes:\t\t\t" + str(total_crashes))
    print("Crash percentage:\t\t" + str(crash_percentage) + "%")
    print("Total vectors considered:\t" + str(total_vectors))
    print("Total unique vectors seen:\t" + str(unique_vector_length_count))
    print("Total possible vectors:\t\t" + str(total_possible_observations))
    print("Total coverage:\t\t\t" + str(overall_coverage) + "%")
    print("Total time to compute:\t\t" + str(datetime.now()-start))

    # Get all the unique number of external vehicles
    unique_vehicle_count = list(set(acuumulative_graph_vehicle_count))
    unique_vehicle_count.sort()

    # Convert to a percentage
    accumulative_graph_coverage = (accumulative_graph / total_possible_observations) * 100

    # Return both arrays
    return_dict[return_key] = [accumulative_graph_coverage, acuumulative_graph_vehicle_count, total_beams]
    return True

parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,   help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=4,    help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',   type=int, default=20,   help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,    help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

new_steering_angle  = args.steering_angle
new_total_lines     = args.beam_count
new_max_distance    = args.max_distance
new_accuracy        = args.accuracy

if new_total_lines == -1:
    new_total_lines = "*"

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Total beams:\t\t" + str(new_total_lines))
print("Max velocity:\t\t" + str(new_max_distance))
print("Vector accuracy:\t" + str(new_accuracy))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = ""
load_name += "_s" + str(new_steering_angle) 
# load_name += "_b" + str(new_total_lines) 
load_name += "_b" + str(new_total_lines) 
load_name += "_d" + str(new_max_distance) 
load_name += "_a" + str(new_accuracy)
load_name += "_t" + str(args.total_samples)
load_name += ".npy"

# Get the file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/numpy_data/' + str(args.total_samples) + "/"
    
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
exec(compile(open("rq1_plot.py", "rb").read(), "rq1_plot.py", 'exec'))