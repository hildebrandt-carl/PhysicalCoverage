import time
import math
import numpy as np


class TrackedObject:
    def __init__(self, coordinates, velocity, obj_id, color_id):
        # Init variables
        self.position = [coordinates]
        self.obj_id = obj_id
        self.color_id = color_id
        self.last_update = time.time()
        self.velocity = [velocity]
        self.current_heading = 0

        self.traj_p1 = None
        self.traj_p2 = None

        self.reachable_set = None

    def update_state(self, pos, vel):
        self.position.append(pos)
        self.velocity.append(vel)
        self.last_update = time.time()
        if abs(vel[0]) < 1e-6:
            self.current_heading = 0
        else:
            self.current_heading = math.atan(vel[1]/vel[0])

    def distance_to(self, pos_in):
        # Get the latest position
        current_pos = self.position[-1]
        # Compute distance
        euclidian_distance = np.sqrt(np.sum(np.square(current_pos - pos_in)))
        return euclidian_distance

    def get_vel_history(self, history_size):
        # If we have enough history to give
        if np.shape(self.velocity)[0] > history_size:
            return self.velocity[-3:]
        else:
            return self.velocity
