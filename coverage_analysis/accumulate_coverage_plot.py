import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Used to output the in a table
tab_rows = []

# Save the results
save_name = "../results/rq1_" + args.scenario
results = load_obj(save_name)

# Create the figures
fig = plt.figure(1)
plt.plot([], [], ' ', label="Total Vectors")

fig = plt.figure(2)
plt.plot([], [], ' ', label="Total Vectors")

 

# For all the data plot it
color_index = 1
for key in results:
    # Expand the data
    accumulative_graph_possible_coverage, accumulative_graph_feasible_coverage, accumulative_graph_vehicle_count, total_beams = results[key]

    # Plotting the coverage per scenario
    fig = plt.figure(1)          
    plt.scatter(np.arange(len(accumulative_graph_possible_coverage)), accumulative_graph_possible_coverage, color='C'+str(color_index), marker='o', label=str(total_beams), s=1)
    fig = plt.figure(2)          
    plt.scatter(np.arange(len(accumulative_graph_feasible_coverage)), accumulative_graph_feasible_coverage, color='C'+str(color_index), marker='o', label=str(total_beams), s=1)
    color_index += 1

    # Save the data into a table
    pc = np.round(accumulative_graph_possible_coverage[-1], 4)
    fc = np.round(accumulative_graph_feasible_coverage[-1], 4)
    tab_rows.append([total_beams, pc, fc])

# Determine the indices where the vehicle count changes
previous_count = 0 
vehicle_count_index_change = []
for i in range(len(accumulative_graph_vehicle_count)):
    vc = accumulative_graph_vehicle_count[i]
    if previous_count != vc:
        vehicle_count_index_change.append(i)
        previous_count = vc
        fig = plt.figure(1) 
        plt.axvline(x=i, linewidth=1, color='C0', linestyle="--")
        fig = plt.figure(2) 
        plt.axvline(x=i, linewidth=1, color='C0', linestyle="--")
        # Highway counts all cars and so we need to subtract 1 for Highway
        if args.scenario == "highway":
            vc -= 1
        fig = plt.figure(1) 
        plt.text(i, 115, str(int(vc)), rotation=0, fontsize=14, color='C0')
        fig = plt.figure(2) 
        plt.text(i, 115, str(int(vc)), rotation=0, fontsize=14, color='C0')

print("\n\n")
print("Index where #vehicles increased - " + str(vehicle_count_index_change))

# Plot the legend
fig = plt.figure(1) 
plt.xlabel("Tests")
plt.ylabel("Physical Coverage (%) - Considering all Vectors")
plt.ylim([-5,125])
plt.legend(markerscale=7, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=7, handletextpad=0.1)
plt.tight_layout()
plt.grid(True, linestyle=':', linewidth=1)


fig = plt.figure(2) 
plt.xlabel("Tests")
plt.ylabel("Physical Coverage (%) - Considering Feasible Vectors")
plt.ylim([-5,125])
plt.legend(markerscale=7, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=7, handletextpad=0.1)
plt.tight_layout()
plt.grid(True, linestyle=':', linewidth=1)



# Display the data
headings = ['Beam Count', 'Percentage Coverage - All possible vectors', 'Percentage Coverage - All feasible vectors']   
print(tabulate(tab_rows, headers=headings))

plt.show()