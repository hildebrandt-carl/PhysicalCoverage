import copy
import time
import math
import numpy as np
import networkx as nx
from tracked_obj import TrackedObject
from shapely.geometry import LineString
from shapely.geometry import Polygon


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class PhysicalAnalysis:
    def __init__(self, distance_threshold, time_threshold, debug=False):
        self.dist_thresh = distance_threshold
        self.time_thresh = time_threshold
        self.debug = debug
        self.tracked_objects = []
        self.current_id = 0
        

    def track(self, current_obs):
        
        # For each observation
        for obs in current_obs:

            # Get the observations position and color
            pos = obs[1:3]
            vel = obs[3:5]
            color = obs[-2]

            # Match the current observation to all known other observations
            distances = []
            distances_objects = []
            for obj in self.tracked_objects:
                # Check if the vehicle colors match
                if obj.color_id == color:
                    # Compute the distance between the two obects
                    d = obj.distance_to(pos)
                    distances.append(d)
                    distances_objects.append(obj)

            update_occured = True
            # If we found possible matches
            if len(distances) > 0:
                # Find the object closest to the current position
                min_index = np.argmin(distances)
                min_d  = distances[min_index]
                min_obj = distances_objects[min_index]
                # If the min distance is less than our threshold we can update the vehicle
                if min_d < self.dist_thresh:
                    min_obj.update_state(pos, vel)
                # We did not update the vehicle
                else:
                    print("Min distance thresh missed: " + str(min_d))
                    update_occured = False
            else:
                # We did not update the vehicle
                update_occured = False

            # We need to create a new object
            if not update_occured:
                self.current_id += 1
                self.tracked_objects.append(TrackedObject(pos, vel, self.current_id, color))

        # Remove all objects which havent been updated in a while
        renewed_objects = []
        for obj in self.tracked_objects:
            if time.time() - obj.last_update < self.time_thresh:
                renewed_objects.append(obj)
        self.tracked_objects = renewed_objects


        return self.tracked_objects

    def compute_trajectories(self, history_size, look_ahead_distance, steering_angle):
        
        # For each object
        for obj in self.tracked_objects:
            # Get the average velocity
            velocties = obj.get_vel_history(history_size)
            avg_vel = np.sum(velocties, axis=0) / len(velocties)
            # Compute the expected next position           
            pos = obj.position[-1]
            obj.update_trajectory(p1=pos, p2=pos + avg_vel)

            # Compute the objects reachable set
            origin, p1 = obj.get_trajectory_points(look_ahead_distance)
            r1 = rotate(origin, p1, math.radians(steering_angle))
            r2 = rotate(origin, p1, math.radians(-1 * steering_angle))
            obj.reachable_set = Polygon([origin, r1, r2])

    def compute_analysis_graph(self):

        # Physical stack
        G = nx.Graph()
        G.add_node("ego")
        current_observation = self.tracked_objects[0]
        pos = {'ego'  :(current_observation.position[-1][0], current_observation.position[-1][1])}
        node_colors = ['g']
    
        # Add a node for each vehicle
        for i in np.arange(1, len(self.tracked_objects)): 
            current_observation = self.tracked_objects[i]
            pos[str(current_observation.obj_id)] = (current_observation.position[-1][0], current_observation.position[-1][1])
            node_colors.append('#A0CBE2')
            G.add_node(str(current_observation.obj_id))
            # G.add_edge("ego", str(current_observation.obj_id))

        return G, pos, node_colors

    def compute_intersection(self, distance_considered):
        # Compute a list of intersecting ids
        checked_ids = []
        intersection_data = []

        # For each object, check if it intersects with every other object
        for x in self.tracked_objects:
            for y in self.tracked_objects:
                # Dont check lines that are the same:
                if x.obj_id != y.obj_id:
                    pair_id = np.sort([x.obj_id, y.obj_id]).tolist()
                    data = {}
                    data["id"] = pair_id
                    # Dont check if the pair has already been checked
                    if pair_id not in checked_ids:
                        # Add them to the checked list
                        checked_ids.append(pair_id)
                        #Get the reach set for both
                        reach1 = x.reachable_set
                        reach2 = y.reachable_set
                        # Compute the percentage area they overlap
                        intersection = (reach1.intersection(reach2).area/reach1.area)*100
                        # Save the intersections
                        data["intersection_percentage"] = int(np.round(intersection,0))
                        intersection_data.append(data)

        return intersection_data



