import numpy as np

def hash_crash(crash_ego_magnitude=None, crash_veh_magnitude=None, crash_incident_angle=None):
    # Check you got everything you needed
    assert(crash_ego_magnitude is not None)
    assert(crash_veh_magnitude is not None)
    assert(crash_incident_angle is not None)

    # Round all the values to ints 
    crash_ego_magnitude = int(np.round(crash_ego_magnitude, 0))
    crash_veh_magnitude = int(np.round(crash_veh_magnitude, 0))
    crash_incident_angle = int(np.round(crash_incident_angle, 0))

    return hash(tuple([crash_ego_magnitude, crash_veh_magnitude, crash_incident_angle]))