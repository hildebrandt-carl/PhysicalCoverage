import sys
import copy

import numpy as np

from tqdm import tqdm
from scipy.spatial.distance import cdist


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_driving_area(scenario):
    drivable_x = [-1, -1]
    drivable_y = [-1, -1]
    if scenario == "highway":
        drivable_x = [305, 908]
        drivable_y = [-1, 12]
    elif scenario == "beamng":
        drivable_x = [-853, 431]
        drivable_y = [-525, 667]

    return drivable_x, drivable_y

def load_improved_bounded_driving_area(scenario):

    # Get the file
    lower_file_name = "../beamng/trajectory_driving_area/{}/right.txt".format(scenario)
    upper_file_name = "../beamng/trajectory_driving_area/{}/left.txt".format(scenario)

    # Load the lower data
    lower_file = open(lower_file_name, "r")
    upper_file = open(upper_file_name, "r")
    files = [lower_file, upper_file]

    # Holds the bounds
    bounds = []

    # For first the lower and then the upper
    for f in files:

        # Holds the lower data
        x_data = []
        y_data = []

        # Read data from the lower file
        line = " "
        while True:
    
            # Get next line from file
            line = f.readline()

            # If we reached the end of the file break
            if len(line) <= 0:
                break

            # Clean the line 
            line = line.strip()
            line = line[1:-1]

            # Get the x,y,z data
            x,y,z = line.split(", ")

            # Save the data
            x_data.append(float(x))
            y_data.append(float(y))

        # Save the bounds
        bounds.append(x_data)
        bounds.append(y_data)

    # Close the files
    lower_file.close()
    upper_file.close()

    # Return the data
    return bounds

def crossed_line(line, incoming_point):

    # Turn the line into a numpy array for easier processing
    line = np.array(line).transpose()
    incoming_point = np.array(incoming_point)
    incoming_point = np.reshape(incoming_point, (1, 2))

    # Define the return variable
    crossed = False

    # Find the closest point on the line to incoming point
    d = cdist(line, incoming_point)

    return d

def create_coverage_array(scenario, drivable_x_size, drivable_y_size, index_lower_bound_x, index_lower_bound_y, index_upper_bound_x, index_upper_bound_y, index_x_array, index_y_array):

    # Create the coverage array
    coverage_array = np.full((drivable_x_size, drivable_y_size), 0,  dtype=int)

    if scenario == "beamng":

        # Loop through the coverage array
        for x in tqdm(range(0, drivable_x_size, 1)):

            # Set the current state (start with invalid)
            state = -1

            # Start saying we have not found lower
            lower_found = False

            for y in range(0, drivable_y_size, 1):
                # Subtract one to account that we start at 0
                y = y - 1

                # If the current index goes over the bounds change it
                if (state == -1) and (lower_found == False):
                    # print("here1")

                    # Check if we have crossed the line yet
                    crossed = crossed_line([index_lower_bound_x, index_lower_bound_y], (x, y))

                    # Find the closest point on the line to the current point
                    closest_index = np.argmin(crossed)

                    # Compare the two points
                    comparison_point = (index_lower_bound_x[closest_index], index_lower_bound_y[closest_index])

                    if comparison_point[1] < y:
                        state = 0
                        lower_found = True

                # If the current index goes over the bounds change it
                if (state == 0) and (lower_found == True):
                    # print("here1")

                    # Check if we have crossed the line yet
                    crossed = crossed_line([index_upper_bound_x, index_upper_bound_y], (x, y))

                    # Find the closest point on the line to the current point
                    closest_index = np.argmin(crossed)

                    # Compare the two points
                    comparison_point = (index_upper_bound_x[closest_index], index_upper_bound_y[closest_index])

                    if comparison_point[1] < y:
                        state = -1

                # Set all invalid areas properly
                coverage_array[x, y] = state

        for p in zip(index_lower_bound_x, index_lower_bound_y):
            coverage_array[p[0], p[1]] = -2

        for p in zip(index_upper_bound_x, index_upper_bound_y):
            coverage_array[p[0], p[1]] = -2

    elif scenario == "highway":
        index_x_array = np.clip(index_x_array, 0, drivable_x_size-1)
        index_y_array = np.clip(index_y_array, 0, drivable_y_size-1)

    # Return the coverage array
    return coverage_array, index_x_array, index_y_array

def compute_trajectory_coverage(coverage_array, number_test_suites, number_tests, index_x_array, index_y_array, drivable_x_size, drivable_y_size):

    # Declare the upper bound used to detect nan
    upper_bound = sys.maxsize - 1e5

    # Get the naive coverage 
    naive_coverage_denominator = float(drivable_x_size * drivable_y_size)
    improved_coverage_denominator = float(np.count_nonzero(coverage_array==0))

    print("Naive denominator: {}".format(naive_coverage_denominator))
    print("Improved denominator: {}".format(improved_coverage_denominator))

    # Create the final coverage array
    naive_coverage_percentage       = np.zeros((number_test_suites, number_tests))
    improved_coverage_percentage    = np.zeros((number_test_suites, number_tests))

    # Run this 10 times
    for test_suite_number in range(number_test_suites):
        print("Test Suite {}".format(test_suite_number))

        # Shuffle both arrays
        index_x_array, index_y_array = unison_shuffled_copies(index_x_array, index_y_array)

        # Create a temporary coverage array
        tmp_coverage_array = copy.deepcopy(coverage_array)

        # Loop through the data and mark off all that we have seen
        for i, coverage_index in tqdm(enumerate(zip(index_x_array, index_y_array)), total=len(index_x_array)):

            # Get the x and y index
            x_index = coverage_index[0]
            y_index = coverage_index[1]

            while ((abs(x_index[-1]) >= upper_bound) or (abs(y_index[-1]) >= upper_bound)):
                x_index = x_index[:-1]
                y_index = y_index[:-1]

            # Update the coverage array
            tmp_coverage_array[x_index, y_index] = 1

            # Compute the coverage
            naive_coverage_percentage[test_suite_number, i]    = (np.count_nonzero(tmp_coverage_array==1) / naive_coverage_denominator) * 100
            improved_coverage_percentage[test_suite_number, i] = (np.count_nonzero(tmp_coverage_array==1) / improved_coverage_denominator) * 100

    return naive_coverage_percentage, improved_coverage_percentage, tmp_coverage_array