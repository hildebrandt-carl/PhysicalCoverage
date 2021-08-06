class HighwayKinematics:
    def __init__(self):
        self.steering_angle  = 30 # Deg
        self.max_velocity    = 30 # m/s

class RSRConfig:
    def __init__(self, beam_count = 3):
        self.beam_count     = beam_count 
        self.accuracy       = 5 