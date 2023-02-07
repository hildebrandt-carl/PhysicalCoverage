import ast
import math
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString

np.set_printoptions(suppress=True)

def create_frame_plot(data, origin, orientation, title, fig_num):
    fig = plt.figure(fig_num)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=5)
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

def combine_environment_and_reachset(r_set, polygons):
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
    return final_r_set

def estimate_reachset(ego_position, steering_angle, total_lines, max_distance):
    # Estimate the reach set
    ego_heading = 0
    r_set = []
    # Convert steering angle to radians
    steering_angle_rad = math.radians(steering_angle)
    # Compute the intervals 
    intervals = 0
    if total_lines > 1:
        intervals = (steering_angle_rad * 2) / float(total_lines - 1)
    # Create each line
    for i in range(total_lines):
        # Compute the angle of the beam
        if total_lines > 1:
            theta = (-1 * steering_angle_rad) + (i * intervals) +  ego_heading
        else:
            theta = ego_heading
        # Compute the new point
        p2 = (ego_position[0] + (max_distance * math.cos(theta)), ego_position[1] + (max_distance * math.sin(theta)))
        # Save the linestring
        l = LineString([ego_position, p2])
        r_set.append(l)
    return r_set

def estimate_obstacles(ego_vehicle, current_lidar_data):
    # Create a list of all polygons for plotting
    polygons = [ego_vehicle]

    # Turn all readings into small polygons
    for pi in range(len(current_lidar_data)):
        # Get the point
        p = current_lidar_data[pi]
        s = 0.2
        new_point = Polygon([(p[0]-s, p[1]-s),
                            (p[0]+s, p[1]-s),
                            (p[0]+s, p[1]+s),
                            (p[0]-s, p[1]+s)])
        polygons.append(new_point)
    return polygons

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

def process_file(file_name, save_name, external_vehicle_count, file_number, total_lines, steering_angle, max_distance, plot):

    output_success = True

    # Check if this file has data
    try:
        with open(file_name, "r") as fp:
            len_of_file = len(fp.readlines())
    except Exception as e:
        logging.exception(e)
        print("Error in file: {}".format(file_name))
        output_success = False  
        
    # If the file is empty
    if len_of_file <= 1:
        output_success = False  
        return output_success

    try:
        # Open the file and count vectors
        input_file = open(file_name, "r")       
        
        # Get the time
        start_time = datetime.now()

        # Save the output to a string which will be saved in the final file
        output_string = "Name: %s\n" % file_name
        output_string += "Date: %s/%s/%s\n" % (start_time.day, start_time.month, start_time.year)
        output_string += "Time: %s:%s:%s\n" % (start_time.hour, start_time.minute, start_time.second)
        output_string += "External Vehicles: %s\n" % external_vehicle_count
        output_string += "Reach set total lines: %d\n" % total_lines
        output_string += "Reach set steering angle: %d\n" % steering_angle
        output_string += "Reach set max distance: %d\n" % max_distance
        output_string += "------------------------------\n"

        frame_counter                   = 0
        previous_time_stamp             = 0
        previous_ego_velocity           = np.array([0, 0, 0])
        previous_traffic_velocity       = np.array([0, 0, 0])
        previous_total_crashes          = 0

        # Print the file
        for line in input_file:
            # Count what frame we are busy computing
            frame_counter += 1

            if frame_counter <= 1:
                continue

            # Read in the data
            # TODO my generated tests for now print it as a numpy array we need to not have it save like that for example
            # TODO 9921875, 10.679703712463379, 0.8139291405677795],0,0,8,7.9708,[2.67828e+01 1.09170e+00 1.00000e-04
            line = line[:line.rfind(",[")]
            read_data = ast.literal_eval(line)

            # Get the data
            data = {}
            data["timestamp"]            = read_data[0]
            data["position"]             = read_data[1]
            data["orientation"]          = read_data[2]
            data["velocity"]             = read_data[3]
            data["lidar"]                = read_data[4]
            data["damgage"]              = read_data[5]
            data["total_accidents"]      = read_data[6]
            data["veh_count"]            = read_data[7]
            data["traffic_vehicle_dist"] = 100
            data["traffic_vehicle_vel"]  = [99, 99, 99]
            # data["traffic_vehicle_dist"] = read_data[8]
            # data["traffic_vehicle_vel"]  = read_data[9]
            data['origin']               = [0, 0, 0]
            data["ego_orientation"]      = [1, 0, 0]

            # Check if there is a collision happening
            collision = False
            if data["total_accidents"] > previous_total_crashes:
                data["collided"] = True
                previous_total_crashes += 1
            else:
                data["collided"] = False

            # If multiple crashes aren't allowed set crash to true
            multiple_crashes_allowed = True
            if multiple_crashes_allowed:
                data["crash"] = False
            else:
                data["crash"] = data["collided"]
            
            # Convert to numpy
            for key in data:
                data[key] = np.array(data[key], dtype=float)

            # Make sure the vehicle counts match
            assert(data["veh_count"] == int(external_vehicle_count))

            # Lidar is given as (x,y,z), (x,y,z), ... etc 
            # Only handle if points are given
            if data["lidar"].shape[0] > 0:
                unique_entries = int(data["lidar"].shape[0] / 3)
                data["lidar"] = data["lidar"].reshape(unique_entries, -1)

                # Change the lidar readings into the car frame
                data["lidar"] = data["lidar"] - data["position"]

                # Compute how much the world is rotated by
                deltaX = data["orientation"][0] - data['origin'][0]
                deltaY = data["orientation"][1] - data['origin'][1]
                rotation = -1 * math.atan2(deltaY, deltaX)

                # Rotate all points 
                points_xy   = data["lidar"][:,0:2]
                origin      = data["origin"][0:2]
                data["rotated_lidar"] = rotate(points_xy, origin, rotation)
                
                # Handle the corner case of only 1 point
                if unique_entries == 1:
                    data["rotated_lidar"] = data["rotated_lidar"].reshape(1,-1)

            else:
                data["rotated_lidar"] = data["lidar"]


            # Plot the points and rotated points
            if plot:
                plt = create_frame_plot(data["lidar"], data["origin"], data["orientation"], "World frame", 1)
                plt = create_frame_plot(data["rotated_lidar"], data["origin"], data["ego_orientation"] , "Vehicle frame", 2)

            # Create the car as an object
            ego_position = [0, 0]
            s = 1
            ego_vehicle = Polygon([(ego_position[0]-(2*s), ego_position[1]-s),
                                (ego_position[0]+(2*s), ego_position[1]-s),
                                (ego_position[0]+(2*s), ego_position[1]+s),
                                (ego_position[0]-(2*s), ego_position[1]+s)])

            # Time our technique
            technique_start_time = datetime.now()

            # Estimate the reachset
            r_set = estimate_reachset(ego_position, steering_angle, total_lines, max_distance)
            # Create a list of all polygons for plotting
            polygons = estimate_obstacles(ego_vehicle, data["rotated_lidar"])
            # Get the final reach set
            final_r_set = combine_environment_and_reachset(r_set, polygons)
            # Vectorize the reach set and round it
            r_vector = vectorize_reachset(final_r_set, accuracy=0.001)

            # Time our technique
            technique_end_time = datetime.now()
            technique_time = (technique_end_time - technique_start_time).total_seconds()
            
            environment_data = {}
            environment_data["polygons"]    = polygons
            environment_data["r_set"]       = r_set
            environment_data["final_r_set"] = final_r_set

            if plot:
                plt.figure(1)
                plt.clf()
                plt.title('Environment')

                # Invert the y axis for easier viewing
                plt.gca().invert_yaxis()

                # Display the environment
                for i in range(len(environment_data["polygons"])):
                    # Get the polygon
                    p = environment_data["polygons"][i]
                    x,y = p.exterior.xy
                    # Get the color
                    c = "g" if i == 0 else "r"
                    # Plot
                    plt.plot(x, y, color=c)

                # Display the reachset
                for i in range(len(environment_data["r_set"])):
                    # Get the polygon
                    p = environment_data["r_set"][i]
                    x,y = p.xy
                    # Get the color
                    c = "r"
                    # Plot
                    plt.plot(x, y, color=c, alpha=0.5)

                # Display the reachset
                for i in range(len(environment_data["final_r_set"])):
                    # Get the polygon
                    p = environment_data["final_r_set"][i]
                    x,y = p.xy
                    # Get the color
                    c = "g"
                    # Plot
                    plt.plot(x, y, color=c)

                # Set the size of the graph
                plt.xlim([-30, 100])
                plt.ylim([-40, 40])

                # Invert the y axis as negative is up and show ticks
                ax = plt.gca()
                ax.set_ylim(ax.get_ylim()[::-1])
                ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

                # plot the graph
                plt.pause(0.1)
                plt.savefig('../../output/file' + str(file_number) + 'frame' + str(frame_counter) + '.png')

                # Plot the environment figures
                plt = create_lidar_plot(environment_data, "Environment Zoomed", [-15, 45], [-30, 30], 3)
                plt = create_lidar_plot(environment_data, "Environment", [-100, 100], [-100, 100], 4)

            # Get the wall time
            wall_time = data["timestamp"] - previous_time_stamp

            # Compute the vectorized reach set
            output_string += "Vector: {}\n".format(r_vector)
            output_string += "Ego Position: {}\n".format(np.round(data["position"],4))
            output_string += "Ego Velocity: {}\n".format(np.round(data["velocity"],4))
            output_string += "Crash: {}\n".format(bool(data["crash"]))
            output_string += "Collided: {}\n".format(bool(data["collided"]))
            output_string += "Operation Time: {}\n".format(technique_time)
            output_string += "Total Wall Time: {}\n".format(wall_time)
            output_string += "Total Simulated Time: {}\n".format(data["timestamp"])
            output_string += "\n"

            # If we have a collision we need to compute the crash details
            if data["collided"]:
                try:
                    # Get the velocity of the two vehicles (we want the velocities just before we crashed)
                    ego_vx, ego_vy = previous_ego_velocity[0:2]
                    veh_vx, veh_vy = previous_traffic_velocity[0:2]

                    # Get magnitude of both velocity vectors
                    ego_mag = np.linalg.norm([ego_vx, ego_vy])
                    veh_mag = np.linalg.norm([veh_vx, veh_vy])

                    # Get the angle of incidence
                    angle_of_incidence = math.degrees(math.atan2(veh_vy, veh_vx) - math.atan2(ego_vy, ego_vx))

                    # Round all values to 4 decimal places
                    ego_mag = np.round(ego_mag, 4)
                    veh_mag = np.round(veh_mag, 4)
                    angle_of_incidence = np.round(angle_of_incidence, 4)
                except ValueError:
                    ego_mag = 0
                    veh_mag = 0 
                    angle_of_incidence = 0

                output_string += "Ego velocity magnitude: {}\n".format(ego_mag)
                output_string += "Incident vehicle velocity magnitude: {}\n".format(veh_mag)
                output_string += "Angle of incident: {}\n\n".format(angle_of_incidence)

            # Save the pervious information for the next loop
            previous_time_stamp = data["timestamp"]
            previous_ego_velocity = data["velocity"]
            previous_traffic_velocity = data["traffic_vehicle_vel"]

            # If we crashed end the trace
            if bool(data["crash"]):
                break

        # Close the input files
        input_file.close()
    except Exception as e:
        logging.exception(e)
        print("Error in file: {}".format(file_name))
        output_success = False        

    # Save the data to a file
    if output_success:
        output_file = open(save_name, "w")
        output_file.write(output_string)
        output_file.close()
        
    return output_success
