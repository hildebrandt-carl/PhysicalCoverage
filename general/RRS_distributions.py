
class linear_distribution():
    def __init__(self, scenario):
      self.scenario = scenario
      if (self.scenario != "highway") or (self.scenario != "beamng"):
        print("Error: 1002 - Unknown scenario")
        exit()

    def get_predfined_points(self):
        predfined_point_distribution = None
        if "highway" in self.scenario:
            predfined_point_distribution = {
            2:  np.array([5,10]),
            3:  np.array([5, 6, 10]),
            4:  np.array([5, 6, 7, 10]),
            5:  np.array([5, 6, 7, 8, 10]),
            6:  np.array([5, 6, 7, 8, 9, 10]),
            7:  np.array([5, 6, 7, 8, 9, 10, 11]),
            8:  np.array([5, 6, 7, 8, 9, 10, 11, 12]),
            9:  np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]),
            10: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            11: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            12: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            13: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            14: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            15: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            16: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            17: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])}
        elif "beamng" in self.scenario:
            predfined_point_distribution = {
            2:  np.array([5,10]),
            3:  np.array([5, 6, 10]),
            4:  np.array([5, 6, 7, 10]),
            5:  np.array([5, 6, 7, 8, 10]),
            6:  np.array([5, 6, 7, 8, 9, 10]),
            7:  np.array([5, 6, 7, 8, 9, 10, 11]),
            8:  np.array([5, 6, 7, 8, 9, 10, 11, 12]),
            9:  np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]),
            10: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            11: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            12: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            13: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            14: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            15: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            16: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            17: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])}

        return predfined_point_distribution

    def get_sample_distribution(self):
        sample_distribution = None
        if "highway" in self.scenario:
            sample_distribution = {
                1:  np.array([15]),
                2:  np.array([5, 5]),
                3:  np.array([3, 4, 3]),
                4:  np.array([2, 3, 3, 2]),
                5:  np.array([2, 2, 2, 2, 2]),
                6:  np.array([2, 2, 2, 2, 2, 2]),
                7:  np.array([2, 2, 2, 2, 2, 2, 2]),
                8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
                10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])}
        elif "beamng" in self.scenario:
            sample_distribution = {
                1:  np.array([8]),
                2:  np.array([3, 3]),
                3:  np.array([2, 3, 2]),
                4:  np.array([2, 2, 2, 2]),
                5:  np.array([2, 2, 2, 2, 2]),
                6:  np.array([2, 2, 2, 2, 2, 2]),
                7:  np.array([2, 2, 2, 2, 2, 2, 2]),
                8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
                10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])}

        return sample_distribution

    def get_angle_distribution(self):
        angle_distribution = None
        if "highway" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-6, 6],
                3:  [-10, 0, 10],
                4:  [-15, -6, 6, 15],
                5:  [-15, -8, 0, 8, 15],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-20, -12, -6, 0, 6, 12, 20],
                8:  [-20, -14, -8, 4, 4, 8, 14, 20],
                9:  [-26, -18, -12, 6, 0, 6, 12, 18, 26],
                10: [-28, -22, -12, -6, -2, 2, 6, 12, 22, 28]}
        elif "beamng" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-7, 7],
                3:  [-11, 0, 11],
                4:  [-17, -7, 7, 17],
                5:  [-20, -8, 0, 8, 20],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-24, -14, -7, 0, 7, 14, 24],
                8:  [-26, -16, -10, 4, 4, 10, 16, 26],
                9:  [-28, -18, -12, 6, 0, 6, 12, 18, 28],
                10: [-30, -24, -15, -7, -2, 2, 7, 15, 24, 30]}
        return angle_distribution

class center_close_distribution():
    def __init__(self, scenario):
      self.scenario = scenario
      if (self.scenario != "highway") or (self.scenario != "beamng"):
        print("Error: 1002 - Unknown scenario")
        exit()

    def get_predfined_points(self):
        predfined_point_distribution = None
        if "highway" in self.scenario:
            predfined_point_distribution = {
            2:  np.array([5,10]),
            3:  np.array([5, 6, 10]),
            4:  np.array([5, 6, 7, 10]),
            5:  np.array([5, 6, 7, 8, 10]),
            6:  np.array([5, 6, 7, 8, 9, 10]),
            7:  np.array([5, 6, 7, 8, 9, 10, 11]),
            8:  np.array([5, 6, 7, 8, 9, 10, 11, 12]),
            9:  np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]),
            10: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            11: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            12: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            13: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            14: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            15: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            16: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            17: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])}
        elif "beamng" in self.scenario:
            predfined_point_distribution = {
            2:  np.array([5,10]),
            3:  np.array([5, 6, 10]),
            4:  np.array([5, 6, 7, 10]),
            5:  np.array([5, 6, 7, 8, 10]),
            6:  np.array([5, 6, 7, 8, 9, 10]),
            7:  np.array([5, 6, 7, 8, 9, 10, 11]),
            8:  np.array([5, 6, 7, 8, 9, 10, 11, 12]),
            9:  np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]),
            10: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            11: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            12: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            13: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            14: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            15: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            16: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            17: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])}

        return predfined_point_distribution

    def get_sample_distribution(self):
        sample_distribution = None
        if "highway" in self.scenario:
            sample_distribution = {
                1:  np.array([15]),
                2:  np.array([5, 5]),
                3:  np.array([3, 4, 3]),
                4:  np.array([2, 3, 3, 2]),
                5:  np.array([2, 2, 2, 2, 2]),
                6:  np.array([2, 2, 2, 2, 2, 2]),
                7:  np.array([2, 2, 2, 2, 2, 2, 2]),
                8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
                10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])}
        elif "beamng" in self.scenario:
            sample_distribution = {
                1:  np.array([8]),
                2:  np.array([3, 3]),
                3:  np.array([2, 3, 2]),
                4:  np.array([2, 2, 2, 2]),
                5:  np.array([2, 2, 2, 2, 2]),
                6:  np.array([2, 2, 2, 2, 2, 2]),
                7:  np.array([2, 2, 2, 2, 2, 2, 2]),
                8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
                10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])}

        return sample_distribution

    def get_angle_distribution(self):
        angle_distribution = None
        if "highway" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-6, 6],
                3:  [-10, 0, 10],
                4:  [-15, -6, 6, 15],
                5:  [-15, -8, 0, 8, 15],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-20, -12, -6, 0, 6, 12, 20],
                8:  [-20, -14, -8, 4, 4, 8, 14, 20],
                9:  [-26, -18, -12, 6, 0, 6, 12, 18, 26],
                10: [-28, -22, -12, -6, -2, 2, 6, 12, 22, 28]}
        elif "beamng" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-7, 7],
                3:  [-11, 0, 11],
                4:  [-17, -7, 7, 17],
                5:  [-20, -8, 0, 8, 20],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-24, -14, -7, 0, 7, 14, 24],
                8:  [-26, -16, -10, 4, 4, 10, 16, 26],
                9:  [-28, -18, -12, 6, 0, 6, 12, 18, 28],
                10: [-30, -24, -15, -7, -2, 2, 7, 15, 24, 30]}
        return angle_distribution

class center_mid_distribution():
    def __init__(self, scenario):
      self.scenario = scenario
      if (self.scenario != "highway") or (self.scenario != "beamng"):
        print("Error: 1002 - Unknown scenario")
        exit()

    def get_predfined_points(self):
        predfined_point_distribution = None
        if "highway" in self.scenario:
            predfined_point_distribution = {
            2:  np.array([5,10]),
            3:  np.array([5, 6, 10]),
            4:  np.array([5, 6, 7, 10]),
            5:  np.array([5, 6, 7, 8, 10]),
            6:  np.array([5, 6, 7, 8, 9, 10]),
            7:  np.array([5, 6, 7, 8, 9, 10, 11]),
            8:  np.array([5, 6, 7, 8, 9, 10, 11, 12]),
            9:  np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]),
            10: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            11: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            12: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            13: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            14: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            15: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            16: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            17: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])}
        elif "beamng" in self.scenario:
            predfined_point_distribution = {
            2:  np.array([5,10]),
            3:  np.array([5, 6, 10]),
            4:  np.array([5, 6, 7, 10]),
            5:  np.array([5, 6, 7, 8, 10]),
            6:  np.array([5, 6, 7, 8, 9, 10]),
            7:  np.array([5, 6, 7, 8, 9, 10, 11]),
            8:  np.array([5, 6, 7, 8, 9, 10, 11, 12]),
            9:  np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]),
            10: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            11: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            12: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            13: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            14: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            15: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            16: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            17: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])}

        return predfined_point_distribution

    def get_sample_distribution(self):
        sample_distribution = None
        if "highway" in self.scenario:
            sample_distribution = {
                1:  np.array([15]),
                2:  np.array([5, 5]),
                3:  np.array([3, 4, 3]),
                4:  np.array([2, 3, 3, 2]),
                5:  np.array([2, 2, 2, 2, 2]),
                6:  np.array([2, 2, 2, 2, 2, 2]),
                7:  np.array([2, 2, 2, 2, 2, 2, 2]),
                8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
                10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])}
        elif "beamng" in self.scenario:
            sample_distribution = {
                1:  np.array([8]),
                2:  np.array([3, 3]),
                3:  np.array([2, 3, 2]),
                4:  np.array([2, 2, 2, 2]),
                5:  np.array([2, 2, 2, 2, 2]),
                6:  np.array([2, 2, 2, 2, 2, 2]),
                7:  np.array([2, 2, 2, 2, 2, 2, 2]),
                8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
                10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])}

        return sample_distribution

    def get_angle_distribution(self):
        angle_distribution = None
        if "highway" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-6, 6],
                3:  [-10, 0, 10],
                4:  [-15, -6, 6, 15],
                5:  [-15, -8, 0, 8, 15],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-20, -12, -6, 0, 6, 12, 20],
                8:  [-20, -14, -8, 4, 4, 8, 14, 20],
                9:  [-26, -18, -12, 6, 0, 6, 12, 18, 26],
                10: [-28, -22, -12, -6, -2, 2, 6, 12, 22, 28]}
        elif "beamng" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-7, 7],
                3:  [-11, 0, 11],
                4:  [-17, -7, 7, 17],
                5:  [-20, -8, 0, 8, 20],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-24, -14, -7, 0, 7, 14, 24],
                8:  [-26, -16, -10, 4, 4, 10, 16, 26],
                9:  [-28, -18, -12, 6, 0, 6, 12, 18, 28],
                10: [-30, -24, -15, -7, -2, 2, 7, 15, 24, 30]}
        return angle_distribution
