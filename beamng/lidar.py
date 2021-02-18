import glob
import math
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import Polygon, LineString, Point

def create_frame_plot(data, origin, orientation, title, fig_num):
    fig = plt.figure(fig_num)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=1)
    ax.quiver(origin[0], origin[1], orientation[0], orientation[1])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title(title)
    plt.xlim([-175, 175])
    plt.ylim([-175, 175])
    return plt

def create_lidar_plot(data, title, x_range, y_range, fig_num):
    plt.figure(fig_num)
    plt.clf()
    plt.title(title)
    # Display the environment
    for i in range(len(data["polygons"])):
        # Get the polygon
        p = data["polygons"][i]
        x,y = p.exterior.xy
        # Get the color
        c = "g" if i == 0 else "r"
        # Plot
        plt.plot(x, y, color=c)
    # Display the reachset
    for i in range(len(data["r_set"])):
        # Get the polygon
        p = data["r_set"][i]
        x,y = p.xy
        # Get the color
        c = "r"
        # Plot
        plt.plot(x, y, color=c, alpha=0.5)
    # Display the reachset
    for i in range(len(data["final_r_set"])):
        # Get the polygon
        p = data["final_r_set"][i]
        x,y = p.xy
        # Get the color
        c = "g"
        # Plot
        plt.plot(x, y, color=c)
    # Set the size of the graph
    plt.xlim(x_range)
    plt.ylim(y_range)
    # Invert the y axis as negative is up and show ticks
    ax = plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    return plt

def getStep(a, MinClip):
    return round(float(a) / MinClip) * MinClip

def vectorize_reachset(lines, accuracy=0.25):
    vector = []
    # For each line:
    for l in lines:
        l_len = l.length
        l_len = getStep(l_len, accuracy)
        l_len = round(l_len, 6)
        vector.append(l_len)
    return vector

def rotate(p, origin=(0, 0), angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

file_names = glob.glob("./data/*.csv")

steering_angle  = 60
total_lines     = 30
max_distance    = 30

for file_number in tqdm(range(len(file_names))):
    # Get the filename
    file_name = file_names[file_number]
    name_only = file_name[file_name.rfind('/')+1:]
    save_name = './data/output/' + name_only[0:-4] + ".txt"

    # Open the file and count vectors
    input_file = open(file_name, "r")
    
    # Open a text file to save output
    # print("Analysing: " + str(file_name))
    # print("Saving to: " + str(save_name))
    output_file = open(save_name, "w")
    output_file.write("Name: %s\n" % file_name)
    e = datetime.datetime.now()
    output_file.write("Date: %s/%s/%s\n" % (e.day, e.month, e.year))
    output_file.write("Time: %s:%s:%s\n" % (e.hour, e.minute, e.second))
    output_file.write("External Vehicles: %d\n" % args.environment_vehicles)
    output_file.write("Reach set total lines: %d\n" % total_lines)
    output_file.write("Reach set steering angle: %d\n" % steering_angle)
    output_file.write("Reach set max distance: %d\n" % max_distance)
    output_file.write("------------------------------\n")

    # Print the file
    first_line = True
    for line in input_file:
        # Add to the time step 
        
        if first_line:
            first_line = False
            continue
    
        # Remove unnecessary characters
        data = line.split("],")
        time_step_position = data[0].split(",[")
        data = data[1:]
        data.insert(0, time_step_position[0])
        data.insert(1, time_step_position[1])

        # Get the data
        current_data = {}
        current_data["position"]        = data[1]
        current_data["orientation"]     = data[2]
        current_data["velocity"]        = data[3]
        current_data["lidar"]           = data[4]
        current_data["crash"]           = data[5]
        current_data['origin']          = "[0, 0, 0]"
        current_data["ego_orientation"] = "[1, 0, 0]"

        # Clean the data
        for key in current_data:
            current_data[key] = current_data[key].replace('[', '') 
            current_data[key] = current_data[key].replace(']', '') 

        # Convert to numpy
        for key in current_data:
            current_data[key] = current_data[key].split(", ")
            current_data[key] = np.array(current_data[key], dtype=float)

        # Get the lidar data into the right shape
        unique_entries = int(current_data["lidar"].shape[0] / 3)
        current_data["lidar"] = current_data["lidar"].reshape(unique_entries, -1)

        # Subtract the cars position from the data
        current_data["lidar"] = current_data["lidar"] - current_data["position"]

        # Remove every 50th element
        # current_data["lidar"] = current_data["lidar"][0::50]
        # print("Analyzing " + str(current_data["lidar"].shape[0]) + " points")

        # Plot the data
        # plt = create_frame_plot(current_data["lidar"], current_data["origin"], current_data["orientation"], "World frame", 1)

        # Compute how much the world is rotated by
        deltaX = current_data["orientation"][0] - current_data['origin'][0]
        deltaY = current_data["orientation"][1] - current_data['origin'][1]
        rotation = -1 * math.atan2(deltaY, deltaX)

        # Rotate all points 
        points_xy   = current_data["lidar"][:,0:2]
        origin      = current_data["origin"][0:2]
        current_data["rotated_lidar"] = rotate(points_xy, origin, rotation)

        # Plot rotated points
        # plt = create_frame_plot(current_data["rotated_lidar"], current_data["origin"], current_data["ego_orientation"] , "Vehicle frame", 2)

        # Create the car as an object
        ego_position = [0, 0]
        s = 1
        ego_vehicle = Polygon([(ego_position[0]-(2*s), ego_position[1]-s),
                               (ego_position[0]+(2*s), ego_position[1]-s),
                               (ego_position[0]+(2*s), ego_position[1]+s),
                               (ego_position[0]-(2*s), ego_position[1]+s)])

        # Estimate the reach set
        ego_heading     = 0
        r_set = []
        # Convert steering angle to radians
        steering_angle_rad = math.radians(steering_angle)
        # Compute the intervals 
        intervals = (steering_angle_rad * 2) / float(total_lines - 1)
        # Create each line
        for i in range(total_lines):
            # Compute the angle of the beam
            theta = (-1 * steering_angle_rad) + (i * intervals) +  ego_heading
            # Compute the new point
            p2 = (ego_position[0] + (max_distance * math.cos(theta)), ego_position[1] + (max_distance * math.sin(theta)))
            # Save the linestring
            l = LineString([ego_position, p2])
            r_set.append(l)

        # Create a list of all polygons for plotting
        polygons = [ego_vehicle]

        # Turn all readings into small polygons
        for p in current_data["rotated_lidar"]:
            s = 0.2
            new_point = Polygon([(p[0]-s, p[1]-s),
                                 (p[0]+s, p[1]-s),
                                 (p[0]+s, p[1]+s),
                                 (p[0]-s, p[1]+s)])
            polygons.append(new_point)

        # Check if any of the reach set intersects with the points
        final_r_set = []
        # For each line
        for l in r_set:
            # Get the origin
            origin = l.coords[0]
            end_position = l.coords[1]
            min_distance = Point(origin).distance(Point(end_position))
            min_point = end_position
            # For each polygon (except the ego vehicle)
            for p in polygons[1:]:
                # Check if they intersect and if so where
                intersect = l.intersection(p)
                if not intersect.is_empty:
                    for i in intersect.coords:
                        # Check which distance is the closest
                        dis = Point(origin).distance(Point(i))
                        if dis < min_distance:
                            min_distance = dis
                            min_point = i
                                
            # Update the line
            true_l = LineString([origin, min_point])
            final_r_set.append(true_l)

        environment_data = {}
        environment_data["polygons"]    = polygons
        environment_data["r_set"]       = r_set
        environment_data["final_r_set"] = final_r_set

        # Plot the environment figures
        # plt = create_lidar_plot(environment_data, "Environment Zoomed", [-15, 45], [-30, 30], 3)
        # plt = create_lidar_plot(environment_data, "Environment", [-100, 100], [-100, 100], 4)

        # Compute the vectorized reach set
        r_vector = vectorize_reachset(environment_data["final_r_set"], accuracy=0.001)
        output_file.write("Vector: " + str(r_vector) + "\n")
        output_file.write("Crash: " + str(current_data["crash"]) + "\n")
        output_file.write("\n")
        if current_data["crash"] > 0:
            print("Crash occured: " + str(file_name))

        # Draw all figures
        plt.pause(0.1)

    # Close both files
    output_file.close()
    input_file.close()