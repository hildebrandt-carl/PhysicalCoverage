import numpy as np

# Use the tests_per_test_suite
def plot_config(scenario):
    if scenario == "beamng":
        total_random_test_suites = 1000
        test_suite_size = [20, 50, 100]
        total_greedy_test_suites = 100
        greedy_sample_sizes = [2, 3, 4, 5, 10]
    elif scenario == "highway":
        total_random_test_suites = 1000
        test_suite_size_percentage = [0.01, 0.025, 0.05]
        total_greedy_test_suites = 100
        greedy_sample_sizes = [2, 3, 4, 5, 10]
        # total_random_test_suites = 1000
        # test_suite_size = [10000, 25000, 50000]
        # total_greedy_test_suites = 100
        # greedy_sample_sizes = [2, 3, 4, 5, 10]
    else:
        exit()

    return total_random_test_suites, test_suite_size_percentage, total_greedy_test_suites, greedy_sample_sizes


def unique_vector_config(scenario, number_of_seconds):
    if scenario == "highway":
        hash_size = 5 * number_of_seconds
    elif scenario == "beamng":
        hash_size = 4 * number_of_seconds
    else:
        exit()
    return hash_size


def compute_crash_hash(trace, hash_size):
    # Used to hold the last vectors before a crash
    last_seen_vectors = np.zeros((hash_size, trace[0].shape[0]))

    # Create the hash
    hash_value = np.nan

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

    return hash_value
