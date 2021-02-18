from shapely.geometry import LineString
import matplotlib.pyplot as plt
import glob

file_names = glob.glob("./data/point*.csv")

for file_number in range(len(file_names)):
    # Get the filename
    file_name = file_names[file_number]

    # Open the file and count vectors
    f = open(file_name, "r")    
    
    # Print the file
    fig_num = 0
    for line in f:
        # Add to the time step 
        fig_num += 1

        # Remove unnecessary characters
        data = line.replace('[', '') 
        data = data.replace(']', '') 
        data = data.split(", ")

        # Create the data
        x = []
        y = []
        z = []

        x_flat = []
        y_flat = []
        z_flat = []

        # Parse the points
        data_length = int(len(data) / 3.0)
        data_counter = 0
        for i in range(data_length):
            
            # Only save x data from a small section
            if 120 <= float(data[data_counter + 2]) <= 121:
                x_flat.append(float(data[data_counter]))
                y_flat.append(float(data[data_counter + 1]))
                z_flat.append(float(data[data_counter + 2]))

            elif i % 100 == 0:
                x.append(float(data[data_counter]))
                y.append(float(data[data_counter + 1]))
                z.append(float(data[data_counter + 2]))             

            data_counter = data_counter + 3

        # Create the plot
        print("Creating figure")
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111)
        ax.scatter(x_flat, y_flat, s=1)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        fig_num += 1

        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_flat, y_flat, z_flat, s=1, c='C0')
        ax.scatter(x, y, z, s=1, c='C1')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_ylabel('Z Label')
        

    f.close()

    plt.show()


