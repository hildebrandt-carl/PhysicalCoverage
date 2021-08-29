from typing import List, Tuple, Union

import numpy as np
import copy
from highway_env_v2 import utils
from highway_env_v2.road.road import Road, LaneIndex, Route
from highway_env_v2.types import Vector
from highway_env_v2.vehicle.kinematics import Vehicle


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 0.5*TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1/3 * KP_HEADING  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)
        total_lines = 93
        self.code_coverage = np.zeros(total_lines)
        self.code_coverage[0] = 1
        self.target_lane_index = target_lane_index or self.lane_index
        self.code_coverage[1] = 1
        self.target_speed = target_speed or self.speed
        self.code_coverage[2] = 1
        self.route = route


    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        self.code_coverage[3] = 1
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        self.code_coverage[4] = 1
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        self.code_coverage[5] = 1
        try:
            self.code_coverage[6] = 1
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            self.code_coverage[7] = 1 ## The except
            self.code_coverage[8] = 1
            path = []
        self.code_coverage[9] = 1
        if path:
            self.code_coverage[10] = 1
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.code_coverage[11] = 1 # The else
            self.code_coverage[12] = 1
            self.route = [self.lane_index]
        self.code_coverage[13] = 1
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.code_coverage[14] = 1
        self.follow_road()
        self.code_coverage[15] = 1
        if action == "FASTER":
            self.code_coverage[16] = 1
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.code_coverage[17] = 1 ## The elif
            self.code_coverage[18] = 1
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            self.code_coverage[17] = 1 ## The elif
            self.code_coverage[19] = 1 ## The second elif
            self.code_coverage[20] = 1
            _from, _to, _id = self.target_lane_index
            self.code_coverage[21] = 1
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            self.code_coverage[22] = 1
            self.code_coverage[23] = 1
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.code_coverage[24] = 1
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            self.code_coverage[17] = 1 ## The elif
            self.code_coverage[19] = 1 ## The second elif
            self.code_coverage[25] = 1 ## The third elif
            self.code_coverage[26] = 1
            _from, _to, _id = self.target_lane_index
            self.code_coverage[27] = 1
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            self.code_coverage[28] = 1
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.code_coverage[29] = 1
                self.target_lane_index = target_lane_index

        self.code_coverage[30] = 1
        action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.speed_control(self.target_speed)}
        self.code_coverage[31] = 1
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        self.code_coverage[32] = 1
        super().act(action)

    def follow_road(self) -> None:
        self.code_coverage[33] = 1
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.code_coverage[34] = 1
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        self.code_coverage[35] = 1
        target_lane = self.road.network.get_lane(target_lane_index)
        self.code_coverage[36] = 1
        lane_coords = target_lane.local_coordinates(self.position)
        self.code_coverage[37] = 1
        lane_next_coords = lane_coords[0] + self.speed * self.PURSUIT_TAU
        self.code_coverage[38] = 1
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        self.code_coverage[39] = 1
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        self.code_coverage[40] = 1
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        self.code_coverage[41] = 1
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        self.code_coverage[42] = 1
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        self.code_coverage[43] = 1
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,-1, 1))
        self.code_coverage[44] = 1                                   
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        self.code_coverage[45] = 1
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        self.code_coverage[0] = 1
        return self.KP_A * (target_speed - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [self.route[0:index+1] + [(self.route[index][1], destination, self.route[index][2])]
                  for destination in next_destinations_from]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.randint(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                     for t in times]))

# This is the vehicle that is running in the random tests
class MDPVehicle(ControlledVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_COUNT: int = 3  # []
    SPEED_MIN: float = 15  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None) -> None:
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.code_coverage[46] = 1 # The super
        self.code_coverage[47] = 1
        self.speed_index = self.speed_to_index(self.target_speed)
        self.code_coverage[48] = 1
        self.target_speed = self.index_to_speed(self.speed_index)
        self.code_coverage[49] = 1
        self.crash_ends_test = True
        self.code_coverage[50] = 1
        self.collided = False

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        self.code_coverage[51] = 1
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
    
        elif action == "SLOWER":
            self.code_coverage[52] = 1 ## The if statement above was executed
            self.code_coverage[53] = 1
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            self.code_coverage[52] = 1 ## The if statement above was executed
            self.code_coverage[54] = 1 ## The else statement above was executed
            self.code_coverage[55] = 1
            super().act(action)
            self.code_coverage[57] = 1
            return
        self.code_coverage[58] = 1
        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.code_coverage[59] = 1
        self.target_speed = self.index_to_speed(self.speed_index)
        self.code_coverage[60] = 1
        super().act()

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        self.code_coverage[61] = 1
        if self.SPEED_COUNT > 1:
            self.code_coverage[62] = 1
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            self.code_coverage[63] = 1 # Else
            self.code_coverage[64] = 1
            return self.SPEED_MIN

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        self.code_coverage[65] = 1
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        self.code_coverage[66] = 1
        return np.int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        self.code_coverage[67] = 1
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        self.code_coverage[68] = 1
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        self.code_coverage[69] = 1
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states

    def check_collision(self, other: Union['Vehicle', 'RoadObject']) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        """
        self.code_coverage[70] = 1
        if not self.crash_ends_test:
            self.code_coverage[71] = 1
            self.crashed = False

        self.code_coverage[72] = 1
        if self.crashed or other is self:
            self.code_coverage[73] = 1
            return

        self.code_coverage[74] = 1
        if isinstance(other, Vehicle):
            self.code_coverage[75] = 1
            if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED:
                self.code_coverage[76] = 1
                return

            self.code_coverage[77] = 1
            if self._is_colliding(other):
                self.code_coverage[78] = 1
                self.speed = other.speed = min([self.speed, other.speed], key=abs)
                self.code_coverage[79] = 1
                self.collided = True
                self.code_coverage[80] = 1
                if self.crash_ends_test:
                    self.code_coverage[81] = 1
                    self.crashed = True
                self.code_coverage[82] = 1
                self.incident_vehicle_kinematic_history = other.kinematic_history
        
        elif isinstance(other, Obstacle):
            self.code_coverage[83] = 1 ## The elif was executed
            self.code_coverage[84] = 1
            if not self.COLLISIONS_ENABLED:
                self.code_coverage[85] = 1
                return

            self.code_coverage[86] = 1
            if self._is_colliding(other):
                self.code_coverage[87] = 1
                self.speed = min([self.speed, 0], key=abs)
                self.code_coverage[88] = 1
                self.collided = True
                self.code_coverage[89] = 1
                if self.crash_ends_test:
                    self.code_coverage[90] = 1
                    self.crashed = other.hit = True

        elif isinstance(other, Landmark):
            self.code_coverage[91] = 1 ## The elif was executed
            if self._is_colliding(other):
                self.code_coverage[92] = 1
                other.hit = True

# There are vehicles used when I want to manually control them with a PID
class ManualVehicle(Vehicle):

    SPEED_MIN: float = 10  # [m/s]
    SPEED_MAX: float = 30  # [m/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)

    @classmethod
    def create_from(cls, vehicle: "ManualVehicle") -> "ManualVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """

        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def act(self, action = None) -> None:
        if action is None:
            super().act()
        else:
            action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
            super().act(action)

    def check_collision(self, other: Union['Vehicle', 'RoadObject']) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        """
        # self.COLLISIONS_ENABLED = False
        if self.crashed or other is self:
            return

        # Ignore if a manual vehicle crashes into another manual vehicle
        if isinstance(other, ManualVehicle):
            return

        if isinstance(other, Vehicle):
            if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = other.speed = min([self.speed, other.speed], key=abs)
                self.crashed = other.crashed = True
        elif isinstance(other, Obstacle):
            if not self.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = min([self.speed, 0], key=abs)
                self.crashed = other.hit = True
        elif isinstance(other, Landmark):
            if self._is_colliding(other):
                other.hit = True