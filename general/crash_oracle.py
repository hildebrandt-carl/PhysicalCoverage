import numpy as np

def round_to_base(x, base):
    return base * round(x/base)

def hash_crash(crash_ego_magnitude=None, crash_veh_magnitude=None, crash_incident_angle=None, base=1):
    # Check you got everything you needed
    assert(crash_ego_magnitude is not None)
    assert(crash_veh_magnitude is not None)
    assert(crash_incident_angle is not None)

    # Round all the values to ints 
    crash_ego_magnitude = round_to_base(crash_ego_magnitude, base=base)
    crash_veh_magnitude = round_to_base(crash_veh_magnitude, base=base)
    crash_incident_angle = round_to_base(crash_incident_angle, base=base)

    return hash(tuple([crash_ego_magnitude, crash_veh_magnitude, crash_incident_angle]))

class CrashOracle:
    def __init__(self, scenario):

        self.max_possible_crashes = 10 

        # The base is defined by the scenario
        self.base = -1
        if scenario == "highway_random" or scenario == "highway_generated":
            self.base = 1
        elif scenario == "beamng_random" or scenario == "beamng_generated":
            self.base = 5
        else:
            print("Error: Unknown scenario base")
            exit()