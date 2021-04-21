import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle', type=int, default=30,   help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',     type=int, default=4,    help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',   type=int, default=20,   help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',       type=int, default=5,    help="What each vector is rounded to")
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Save the results
save_name = "../results/rq1_" + args.scenario
results = load_obj(save_name)

# Create the figure
fig = plt.figure(1)

# Proxy for additional label
plt.plot([], [], ' ', label="Total Vectors")

# For all the data plot it
color_index = 0
for key in results:
    # Expand the data
    accumulative_graph_coverage, accumulative_graph_vehicle_count, total_beams = results[key]

    # Plotting the coverage per scenario            
    plt.scatter(np.arange(len(accumulative_graph_coverage)), accumulative_graph_coverage, color='C'+str(color_index), marker='o', label=str(total_beams), s=1)
    color_index += 1

# Determine the indices where the vehicle count changes
previous_count = 0 
vehicle_count_index_change = []
for i in range(len(accumulative_graph_vehicle_count)):
    vc = accumulative_graph_vehicle_count[i]
    if previous_count != vc:
        vehicle_count_index_change.append(i)
        previous_count = vc
        plt.axvline(x=i, linewidth=1, color='0.7', linestyle="--")
        # Highway counts all cars and so we need to subtract 1 for Highway
        if args.scenario == "highway":
            vc -= 1
        plt.text(i, 105, str(int(vc)), rotation=0, fontsize=14, color='0.5')

print("Index where #vehicles increased - " + str(vehicle_count_index_change))

# Plot the legend
plt.xlabel("Tests")
plt.ylabel("Physical Coverage (%)")
plt.ylim([-5,110])
plt.legend(markerscale=7, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=7, handletextpad=0.1)
plt.tight_layout()

plt.show()