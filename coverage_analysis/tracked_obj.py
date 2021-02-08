import time
import numpy as np


def line_equation(l1, l2):
    """Line encoded as l=(x,y)."""
    m = (l2[1] - l1[1]) / (l2[0] - l1[0])
    b = (l2[1] - (m * l2[0]))
    return m, b


class TrackedObject:
    def __init__(self, coordinates, velocity, obj_id, color_id):
        # Init variables
        self.position = [coordinates]
        self.obj_id = obj_id
        self.color_id = color_id
        self.last_update = time.time()
        self.velocity = [velocity]

        self.traj_p1 = None
        self.traj_p2 = None

        self.reachable_set = None

    def update_state(self, pos, vel):
        self.position.append(pos)
        self.velocity.append(vel)
        self.last_update = time.time()

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

    def get_trajectory_velocity(self):
        return self.traj_p1, self.traj_p2

    def get_trajectory_line(self):
        if (self.traj_p1 is not None) and (self.traj_p2 is not None):
            m, b = line_equation(self.traj_p1, self.traj_p2)
            return m, b
        else:
            return None, None

    def get_trajectory_points(self, lookahead=50):
        if (self.traj_p1 is not None) and (self.traj_p2 is not None):
            m, b = line_equation(self.traj_p1, self.traj_p2)
            x = lookahead + self.position[-1][0]
            y = (m * x) + b 
            return self.traj_p1, np.array([x, y]).tolist()
        else:
            return None, None

    def update_trajectory(self, p1, p2):
        self.traj_p1 = np.array(p1).tolist()
        self.traj_p2 = np.array(p2).tolist()