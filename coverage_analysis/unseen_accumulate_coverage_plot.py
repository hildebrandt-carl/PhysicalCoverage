import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
from matplotlib_venn import venn3

def create_venn(original_vectors, new_vectors, feasible_vectors, total_possible_observations, beam_number):

    # Create the sets that will hold the unique values
    original_data_set       = set()
    unseen_data_set         = set()
    feasible_vectors_set    = set()

    # Convert the original data to a set
    for vector in original_vectors:
        # Add it to the set
        original_data_set.add(tuple(vector))

    # Convert the new data to a set
    for vector in new_vectors:
        # Add it to the set
        unseen_data_set.add(tuple(vector))

    # Convert the new data to a set
    for vector in feasible_vectors:
        # Add it to the set
        feasible_vectors_set.add(tuple(vector))

    print("Total unique vectors in original data: {}".format(len(original_data_set)))
    print("Total unique vectors in new data: {}".format(len(unseen_data_set)))
    print("Total unique vectors in feasible data: {}".format(len(unseen_data_set)))

    # Quick check to see if nothing messed up with the feasible vectors (each should be unique)
    if len(feasible_vectors) != len(feasible_vectors_set):
        print("Error: Something is wrong with your feasible vectors list. It should be the same size as the set but it is not")
        exit()

    # Compute the number of vectors outside the vendiagram
    vectors_not_seen = total_possible_observations - len(feasible_vectors_set) - len(original_data_set - feasible_vectors_set) - len(unseen_data_set - feasible_vectors_set)

    plt.figure("Venn - Beams: {}".format(beam_number))
    plt.rcParams["figure.autolayout"] = True
    venn3([original_data_set, unseen_data_set, feasible_vectors_set], ('Randomly Found Vectors', 'Generated Vectors', 'Feasible Vectors'))
    plt.text(-1, -0.8 , "Total possible vectors: " + str(total_possible_observations))
    plt.text(0.5, -0.5 , str(vectors_not_seen))
    plt.ylim([-0.8, 0.8])
    plt.xlim([-0.8, 0.8])
    return plt


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Save the results
save_name = "../results/rq5_" + args.scenario
results = load_obj(save_name)

color_index = 0

for key in results:

    # Get the data
    combined_data, original_data, unseen_data, misc_data, total_beams = results[key]

    print("Processing beams: {}".format(total_beams))

    # Expand the data
    c_possible_coverage, c_feasible_coverage, c_veh_count, c_unique_vectors, c_crash_array, c_unique_crash_array = combined_data
    o_possible_coverage, o_feasible_coverage, o_veh_count, o_unique_vectors, o_crash_array, o_unique_crash_array = original_data
    u_possible_coverage, u_feasible_coverage, u_veh_count, u_unique_vectors, u_crash_array, u_unique_crash_array = unseen_data
    all_possible_vector_count, feasible_vectors                                            = misc_data

    # Create a Ven diagram showing the additional coverage added by the new tests
    print("Generating ven diagrams")
    plt = create_venn(o_unique_vectors, u_unique_vectors, feasible_vectors, all_possible_vector_count, total_beams)

    # Compute the time taken for each
    original_test_duration = 20
    new_test_duration = 100
    original_time_taken = np.arange(len(o_possible_coverage)) * original_test_duration
    new_time_taken = (np.arange(len(u_possible_coverage)) * new_test_duration) + original_time_taken[-1]
    time_taken = np.concatenate([original_time_taken, new_time_taken])

    # Define where the newly generated tests start
    newly_generated_tests = original_time_taken[-1]

    # Create a line graph
    fig = plt.figure("Possible_Coverage") 
    plt.scatter(time_taken, c_possible_coverage, color='C'+str(color_index), marker='o', label=str(total_beams), s=1)
    plt.axvline(x=newly_generated_tests, color='red', linestyle='--')

    fig = plt.figure("Feasible_Coverage") 
    plt.scatter(time_taken, c_feasible_coverage, color='C'+str(color_index), marker='o', label=str(total_beams), s=1)
    plt.axvline(x=newly_generated_tests, color='red', linestyle='--')

    fig = plt.figure("Crashes") 
    plt.scatter(time_taken, c_crash_array, color='C'+str(color_index), marker='o', label=str(total_beams), s=1)
    plt.axvline(x=newly_generated_tests, color='red', linestyle='--')

    fig = plt.figure("Unique_Crashes") 
    plt.scatter(time_taken, c_unique_crash_array, color='C'+str(color_index), marker='o', label=str(total_beams), s=1)
    plt.axvline(x=newly_generated_tests, color='red', linestyle='--')

    color_index += 1

# Update the plots data
fig = plt.figure("Possible_Coverage") 
plt.xlabel("Time Taken")
plt.ylabel("Physical Coverage (%) - Considering All Vectors")
plt.ylim([-5,105])
plt.legend(markerscale=7, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=7, handletextpad=0.1)
plt.tight_layout()
plt.grid(True, linestyle=':', linewidth=1)

fig = plt.figure("Feasible_Coverage") 
plt.xlabel("Time Taken")
plt.ylabel("Physical Coverage (%) - Considering Feasible Vectors")
plt.ylim([-5,105])
plt.legend(markerscale=7, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=7, handletextpad=0.1)
plt.tight_layout()
plt.grid(True, linestyle=':', linewidth=1)

fig = plt.figure("Crashes") 
plt.xlabel("Time Taken")
plt.ylabel("Number of crashes")
plt.legend(markerscale=7, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=7, handletextpad=0.1)
plt.tight_layout()
plt.grid(True, linestyle=':', linewidth=1)

fig = plt.figure("Unique_Crashes") 
plt.xlabel("Time Taken")
plt.ylabel("Number of unique crashes")
plt.legend(markerscale=7, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=7, handletextpad=0.1)
plt.tight_layout()
plt.grid(True, linestyle=':', linewidth=1)

# Show the plots
plt.show()
