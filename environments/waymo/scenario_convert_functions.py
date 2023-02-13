import os
# Stop tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from shapely.geometry import Polygon
from lidar_to_RRS_functions import estimate_reachset
from lidar_to_RRS_functions import estimate_obstacles
from lidar_to_RRS_functions import vectorize_reachset
from lidar_to_RRS_functions import combine_environment_and_reachset

def convert_file_to_raw_vector(filename, total_lines, steering_angle, max_distance):

    # Enable eager execution
    tf.enable_eager_execution()

    # Get the scenario name
    scenario_name = filename[filename.rfind("/")+1:filename.rfind(".")]

    # Create the additional info save folder
    additional_info_save_folder = '../../output/additional_data/{}'.format(scenario_name)
    if not os.path.exists("{}".format(additional_info_save_folder)):
        os.makedirs("{}".format(additional_info_save_folder))

    if not os.path.exists("{}/camera_data".format(additional_info_save_folder)):
        os.makedirs("{}/camera_data".format(additional_info_save_folder))

    if not os.path.exists("{}/point_cloud_data/raw".format(additional_info_save_folder)):
        os.makedirs("{}/point_cloud_data/raw".format(additional_info_save_folder))

    if not os.path.exists("{}/point_cloud_data/RRS".format(additional_info_save_folder)):
        os.makedirs("{}/point_cloud_data/RRS".format(additional_info_save_folder))

    # Create the output data
    save_folder = "../../output/waymo/random_tests/physical_coverage/raw/"
    if not os.path.exists("{}".format(save_folder)):
        os.makedirs("{}".format(save_folder))

    # Load the data
    dataset = tf.data.TFRecordDataset("{}".format(filename), compression_type='')

    # Get the time
    start_time = datetime.now()

    # Save the output to a string which will be saved in the final file
    output_string = "Name: %s\n" % scenario_name
    output_string += "Date: %s/%s/%s\n" % (start_time.day, start_time.month, start_time.year)
    output_string += "Time: %s:%s:%s\n" % (start_time.hour, start_time.minute, start_time.second)
    output_string += "External Vehicles: %s\n" % "na"
    output_string += "Reach set total lines: %d\n" % total_lines
    output_string += "Reach set steering angle: %d\n" % steering_angle
    output_string += "Reach set max distance: %d\n" % max_distance
    output_string += "------------------------------\n"

    # Initialize the variables
    frame_counter = 0

    # Loop through the dataset
    for data in dataset:

        # Load the frame
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Used to merge cameras
        CAMERA_WIDTH=1920
        PIXEL_OVERLAP=150

        # Track the time for this operation
        op_start_time = datetime.now()

        # Save the camera data
        merge_and_save_camera(frame, additional_info_save_folder, frame_counter, camera_width=CAMERA_WIDTH, pixel_overlap=PIXEL_OVERLAP)

        # Get and clean the lidar data
        cloud_points = get_clean_lidar(frame, additional_info_save_folder, frame_counter)

        # Create the environment
        environment_data, r_vector = create_reachset(cloud_points)

        # Plot the lidar datat
        plot_lidar(environment_data, additional_info_save_folder, frame_counter)
        

        # Track the time for this operation
        current_time = datetime.now()
        operation_time = (current_time - op_start_time).total_seconds()
        elapsed_time = (current_time - start_time).total_seconds()
        simulated_time = 0

        # Get the position of the vehicle
        ego_position = np.zeros(3)         # pose = np.reshape(np.array(frame.pose.transform, np.float32), (4, 4))
        ego_velocity = np.array([frame.images[0].velocity.v_x, frame.images[0].velocity.v_y, frame.images[0].velocity.v_z])

        # Compute the vectorized reach set
        output_string += "Vector: {}\n".format(r_vector)
        output_string += "Ego Position: {}\n".format(np.round(ego_position,4))
        output_string += "Ego Velocity: {}\n".format(np.round(ego_velocity,4))
        output_string += "Crash: {}\n".format(bool(False))
        output_string += "Collided: {}\n".format(bool(False))
        output_string += "Operation Time: {}\n".format(operation_time)
        output_string += "Total Wall Time: {}\n".format(elapsed_time)
        output_string += "Total Simulated Time: {}\n".format(simulated_time)
        output_string += "\n"

        # Increment the frame counter
        frame_counter += 1

    # Create the output file
    output_file = open("{}/{}.txt".format(save_folder, scenario_name), "w")
    output_file.write(output_string)
    output_file.close()

    # Return
    return True


def merge_and_save_camera(frame, save_folder, frame_counter, camera_width=1920, pixel_overlap=170):

    # Get the front, left and right camera
    cam_front, cam_left, cam_right = None, None, None
    for img in frame.images:
        if (img.name == 1):
            cam_front = img
        elif (img.name == 2):
            cam_left = img
        elif (img.name == 3):
            cam_right = img

    # Get the 3 different images
    front_img = tf.image.decode_jpeg(cam_front.image)[:,:,:]
    left_img = tf.image.decode_jpeg(cam_left.image)[:,:camera_width-pixel_overlap,:]
    right_img = tf.image.decode_jpeg(cam_right.image)[:,pixel_overlap:,:]

    # Plot the camera data as a panorama
    f, ax_array = plt.subplots(1,3, figsize=(10, 5))
    ax_array[0].imshow(left_img, aspect="auto")
    ax_array[1].imshow(front_img, aspect="auto")
    ax_array[2].imshow(right_img, aspect="auto")

    # Turn off the axis
    ax_array[0].axis('off')
    ax_array[1].axis('off')
    ax_array[2].axis('off')

    # Remove the boarders and save image
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig("{}/camera_data/camera{:05d}.png".format(save_folder, frame_counter),bbox_inches='tight')
    plt.close()


def get_clean_lidar(frame, save_folder, frame_counter):

    # Convert the frame into a camera projection
    (range_images, camera_projections, seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    # Convert the lidar into cartesian data
    points = frame_utils.convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, 0, False)

    # Get the LIDAR from the front lidar
    cartesian_points = points[1]

    # Combine all the data from each of the different LiDARS
    points_all = np.concatenate(cartesian_points, axis=0)

    # Declare the range
    LOW_Z = 0.5
    HIGH_Z = 1

    # Create the array to hold the data
    points_at_z_height = []

    # For each of the points
    for i in range(len(points_all)):

        # If it is at the right height and in front of the vehicle
        if (LOW_Z < points_all[i][2] < HIGH_Z) and (points_all[i][0] >= -1):
            points_at_z_height.append(points_all[i])

    # Convert the points into a numpy array
    points_at_z_height = np.array(points_at_z_height)

    # Plot the scatter plot
    fig = plt.figure(figsize = (10, 10))
    plt.scatter(points_at_z_height[:, 1], points_at_z_height[:, 0], s=1)

    # Set the size
    plt.xlim([75, -75])
    plt.ylim([-75, 75])

    # Create a title
    plt.title("{} > z > {}".format(LOW_Z, HIGH_Z))
    plt.savefig("{}/point_cloud_data/raw/point_cloud{:05d}.png".format(save_folder, frame_counter))
    plt.close()

    return points_at_z_height


def create_reachset(point_cloud):

    # Create the car as an object
    ego_position = [0, 0]
    s = 1
    ego_vehicle = Polygon([(ego_position[0]-(2*s), ego_position[1]-s),
                        (ego_position[0]+(2*s), ego_position[1]-s),
                        (ego_position[0]+(2*s), ego_position[1]+s),
                        (ego_position[0]-(2*s), ego_position[1]+s)])

    # Compute the RRS values
    r_set       = estimate_reachset([0,0], 45, 30, 35)
    polygons    = estimate_obstacles(ego_vehicle, point_cloud)
    final_r_set = combine_environment_and_reachset(r_set, polygons)
    r_vector    = vectorize_reachset(final_r_set, accuracy=0.001)

    environment_data = {}
    environment_data["polygons"]    = polygons
    environment_data["r_set"]       = r_set
    environment_data["final_r_set"] = final_r_set

    return environment_data, r_vector


def plot_lidar(environment_data, save_folder, frame_counter):
    plt.figure(figsize = (10, 10))
    plt.title('Environment')

    # Display the environment
    for i in range(len(environment_data["polygons"])):
        # Get the polygon
        p = environment_data["polygons"][i]
        x,y = p.exterior.xy
        # Get the color
        c = "g" if i == 0 else "b"
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

    # # Set the size of the graph
    plt.xlim([75, -75])
    plt.ylim([-75, 75])

    # plot the graph
    plt.savefig("{}/point_cloud_data/RRS/point_cloud{:05d}.png".format(save_folder, frame_counter))
    plt.close()
