import pickle
import argparse

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
return_dict = load_obj(save_name)

# For all the data plot it
color_index = 0
for key in return_dict.keys():
    print(key)
    # Expand the data
    accumulative_graph_coverage, acuumulative_graph_vehicle_count, total_beams = return_dict[key]

    # Plotting the coverage per scenario            
    plt.figure(1)
    plt.scatter(np.arange(len(accumulative_graph_coverage)), accumulative_graph_coverage, color='C'+str(color_index), marker='o', label=str(total_beams) + " beams", s=1)
    color_index += 1

# Plot the legend
plt.legend(loc='upper left', markerscale=7)
plt.title("Reachable Set Coverage")
plt.xlabel("Scenario")
plt.ylabel("Reachable Set Coverage (%)")

plt.show()