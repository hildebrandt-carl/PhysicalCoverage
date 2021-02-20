import glob
import random 
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import argparse

def isUnique(vector, unique_vectors_seen):
    unique = True
    for v2 in unique_vectors_seen:
        # If we have seen this vector break out of this loop
        if np.array_equal(vector, v2):
            unique = False
            break
    return unique

parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,   help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=4,    help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',   type=int, default=20,   help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,    help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
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
trace_file_names = glob.glob("traces" + load_name)
file_names = []
for f in trace_file_names:
    name = f.replace("traces", "")
    file_names.append(name)

# Sort the names
file_names.sort()

print("----------------------------------")
print("-------Computing Coverage---------")
print("----------------------------------")

for file_index in range(len(file_names)):

    # Get the name we are processing
    load_name = file_names[file_index]

    total_beams = load_name[load_name.find("_b")+2:]
    total_beams = total_beams[0:total_beams.find("_d")]
    total_beams = int(total_beams)

    # Compute total possible values using the above
    unique_observations_per_cell = (new_max_distance / float(new_accuracy)) + 1.0
    total_possible_observations = pow(unique_observations_per_cell, total_beams)

    print("Processing: " + load_name)
    traces = np.load("traces" + load_name)
    vehicles = np.load("vehicles" + load_name)

    # Sort the data based on the number of vehicles per test
    vehicles_indices = vehicles.argsort()
    traces = traces[vehicles_indices]
    vehicles = vehicles[vehicles_indices]

    unique_vectors_seen = []
    accumulative_graph = []
    acuumulative_graph_vehicle_count = []
    total_traces = traces.shape[0]
    total_crashes = 0
    total_vectors = 0

    # For each file
    for i in tqdm(range(total_traces)):

        vectors = traces[i]
        vehicle_count = vehicles[i]

        if np.isnan(vectors).any():
            total_crashes += 1
        # Update total vectors observed
        for j in range(vectors.shape[0]):
            if not np.isnan(vectors[j]).any():
                total_vectors += 1
        # Check to see if any of the vectors are new
        for v in vectors:
            if not np.isnan(v).any():
                unique = isUnique(v, unique_vectors_seen)
                if unique:
                    unique_vectors_seen.append(v)
        # Used for the accumulative graph
        accumulative_graph.append(len(unique_vectors_seen))
        acuumulative_graph_vehicle_count.append(vehicle_count)

    overall_coverage = round((len(unique_vectors_seen) / float(total_possible_observations)) * 100, 4)
    crash_percentage = round(total_crashes / float(total_traces) * 100, 4)

    print("Total traces considered:\t" + str(total_vectors))
    print("Total crashes:\t\t\t" + str(total_crashes))
    print("Crash percentage:\t\t" + str(crash_percentage) + "%")
    print("Total vectors considered:\t" + str(total_vectors))
    print("Total unique vectors seen:\t" + str(len(unique_vectors_seen)))
    print("Total possible vectors:\t\t" + str(total_possible_observations))
    print("Total coverage:\t\t\t" + str(overall_coverage) + "%")

    # Get all the unique number of external vehicles
    unique_vehicle_count = list(set(acuumulative_graph_vehicle_count))
    unique_vehicle_count.sort()

    # Convert to numpy arrays
    accumulative_graph_coverage = (np.array(accumulative_graph) / total_possible_observations) * 100
    acuumulative_graph_vehicle_count = np.array(acuumulative_graph_vehicle_count)

    # Plotting the coverage per scenario            
    plt.figure(1)
    plt.scatter(np.arange(len(accumulative_graph_coverage)), accumulative_graph_coverage, color='C'+str(file_index), marker='o', label=str(total_beams) + " beams", s=1)

plt.legend(loc='upper left', markerscale=7)
plt.title("Reachable Set Coverage")
plt.xlabel("Scenario")
plt.ylabel("Reachable Set Coverage (%)")

plt.show()