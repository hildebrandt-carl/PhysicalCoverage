import numpy as np
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
    lower_file_name = "../trajectory_driving_area/{}/right.txt".format(scenario)
    upper_file_name = "../trajectory_driving_area/{}/left.txt".format(scenario)

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
