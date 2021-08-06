from shapely.geometry import Polygon, LineString, Point
from shapely import affinity
import math

def getStep(a, MinClip):
    return round(float(a) / MinClip) * MinClip

class ReachableSet:
    def __init__(self, obstacle_size=1):
      self.obs_size = obstacle_size
      self.ego_position = (0, 0)
      self.ego_heading = 0

    """
    compute_environment takes in a list of tracked objects and returns a list of polygons that are in the same position as the tracked objects and rotated to match the objects heading

    :param tracked_objects: a list of tracked objects
    :param lanes: an array containing the upper and lower lane bounds
    :return: a list of polygons
    """ 
    def compute_environment(self, tracked_objects, lanes=None):
        
        # Save the ego_position
        ego_pos = tracked_objects[0].position[-1]
        self.ego_position = (ego_pos[0], ego_pos[1])
        self.ego_heading = tracked_objects[0].current_heading

        obstacles = []
        # For each object
        for t in tracked_objects:
            # get the position
            pos = t.position[-1]
            x = pos[0]
            y = pos[1]
            s = self.obs_size
            # Create a polygon around the obstacle
            obs_instance = Polygon([(x-(2*s), y-s), (x+(2*s), y-s), (x+(2*s), y+s), (x-(2*s), y+s)])
            # Rotate obstacle around center
            rotation = math.degrees(t.current_heading)
            obs_instance = affinity.rotate(obs_instance, rotation, 'center')
            # Save for printing
            obstacles.append(obs_instance)

        # Create an upper and lower bound if the lanes are given
        if lanes is not None:
            # Get the x and y values
            x1 = lanes[0][0][0]
            x2 = lanes[0][1][0]
            y1 = lanes[0][0][1]
            y2 = lanes[1][0][1]

            # Create the obstacles
            l1 = Polygon([(x1, y1), (x2, y1),(x2, y1 + 0.001), (x1, y1 + 0.001)])
            l2 = Polygon([(x1, y2), (x2, y2),(x2, y2 - 0.001), (x1, y2 - 0.001)])

            obstacles.append(l1)
            obstacles.append(l2)
             
        return obstacles

    """
    estimate_raw_reachset creates an estimation of the reachable set from the ego vehicle. It estimates the reachable set using a set of lines starting from the center of the ego vehicle

    :param total_lines: The total number of lines required for the estimated reachable set
    :param steering_angle: The maximum steering angle of the ego vehicle (determines how broad the reach set will be)
    :param max_distance: the maximum distance of each of the lines
    :return: a list of lines
    """ 
    def estimate_raw_reachset(self, total_lines=40, steering_angle=40, max_distance=30):
        # Create the output
        lines = []

        # Convert steering angle to radians
        steering_angle = math.radians(steering_angle)

        # Compute the intervals 
        intervals = 0 

        if total_lines > 1:
            intervals = (steering_angle * 2) / float(total_lines - 1)
        
        # Create each line
        for i in range(total_lines):
            # Compute the angle of the beam
            if total_lines > 1:
                theta = (-1 * steering_angle) + (i * intervals) +  self.ego_heading
            else:
                theta = self.ego_heading

            # Compute the new point
            p2 = (self.ego_position[0] + (max_distance * math.cos(theta)), self.ego_position[1] + (max_distance * math.sin(theta)))

            # Save the linestring
            l = LineString([self.ego_position, p2])
            lines.append(l)

        # Return the output
        return lines

    """
    estimate_true_reachset creates a better approximation of the reachable set but taking into consideration all the other obstacles in the world.

    :param polygons: A list of polygons in the world
    :param r_set: A list of lines 
    :return: a list of lines that do not pass through any of the other obstacles
    """ 
    def estimate_true_reachset(self, polygons, r_set):
        new_lines = []
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
            new_lines.append(true_l)

        return new_lines

    """
    vectorize_reachset turns a list of lines into a vector.

    :param lines: A list of lines 
    :param accuracy: The accuracy you want the vector to be. i.e. 0.5 will round to the closest 0.5
    :return: a vector which represents the reachable set
    """ 
    def vectorize_reachset(self, lines, accuracy=0.25):
        vector = []
        # For each line:
        for l in lines:
            l_len = l.length
            l_len = getStep(l_len, accuracy)
            l_len = round(l_len, 6)
            vector.append(l_len)

        return vector



    """
    Take a reachable set into a series of points that represent the possible readings 

    :param line: A list of lines
    :param accuracy: The accuracy you want the points to be. i.e. 0.5 will round to the closest 0.5
    :return: a list of lines where each line is described as a series of points
    """ 
    def line_to_points(self, lines, accuracy=0.25):
        segmented_lines = []
        # For each line:
        for l in lines:
            points = []
            current_dist = 0
            while current_dist <= l.length:
                new_point = l.interpolate(current_dist)
                current_dist += accuracy
                points.append(new_point)
            segmented_lines.append(points)

        return segmented_lines