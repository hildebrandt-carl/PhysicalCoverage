import time
import numpy as np
from tracked_obj import TrackedObject


class Tracker:
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

    def get_observations(self):
        return self.tracked_objects