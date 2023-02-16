import numpy as np

class center_close_distribution():
    def __init__(self, scenario):
      self.scenario = scenario
      if not (("highway" in self.scenario) or ("beamng" in self.scenario) or ("waymo" in self.scenario)):
        print("Error: 1002 - Unknown scenario ({})".format(self.scenario))
        exit()

    def get_predefined_points(self):
        predefined_point_distribution = None
        if "highway" in self.scenario:
            predefined_point_distribution = {
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
            predefined_point_distribution = {
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
        elif "waymo" in self.scenario:
            predefined_point_distribution = {
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

        return predefined_point_distribution

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
                1:  np.array([10]),
                2:  np.array([5, 5]), 
                3:  np.array([3, 4, 3]),
                4:  np.array([2, 4, 4, 2]),
                5:  np.array([2, 2, 3, 2, 2]),
                6:  np.array([2, 2, 2, 2, 2, 2]),
                7:  np.array([2, 2, 2, 2, 2, 2, 2]),
                8:  np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                9:  np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),
                10: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])}
        elif "waymo" in self.scenario:
            sample_distribution = {
                1:  np.array([10]),
                2:  np.array([5, 5]), 
                3:  np.array([3, 4, 3]),
                4:  np.array([2, 4, 4, 2]),
                5:  np.array([2, 2, 3, 2, 2]),
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
                8:  [-26, -16, -10, -4, 4, 10, 16, 26],
                9:  [-28, -18, -12, -6, 0, 6, 12, 18, 28],
                10: [-30, -24, -15, -7, -2, 2, 7, 15, 24, 30]}
        elif "waymo" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-7, 7],
                3:  [-11, 0, 11],
                4:  [-17, -7, 7, 17],
                5:  [-20, -8, 0, 8, 20],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-24, -14, -7, 0, 7, 14, 24],
                8:  [-26, -16, -10, -4, 4, 10, 16, 26],
                9:  [-28, -18, -12, -6, 0, 6, 12, 18, 28],
                10: [-30, -24, -15, -7, -2, 2, 7, 15, 24, 30]}
        return angle_distribution

class center_full_distribution():
    def __init__(self, scenario):
      self.scenario = scenario
      if not (("highway" in self.scenario) or ("beamng" in self.scenario) or ("waymo" in self.scenario)):
        print("Error: 1002 - Unknown scenario ({})".format(self.scenario))
        exit()

    def get_predefined_points(self):
        predefined_point_distribution = None
        if "highway" in self.scenario:
            predefined_point_distribution = {
            2:  np.array([5.00, 30.00]),
            3:  np.array([5.00, 17.50, 30.00]),
            4:  np.array([5.00, 13.33, 21.67, 30.00]),
            5:  np.array([5.00, 11.25, 17.50, 23.75, 30.00]),
            6:  np.array([5.00, 10.00, 15.00, 20.00, 25.00, 30.00]),
            7:  np.array([5.00,  9.17, 13.33, 17.50 ,21.67, 25.83, 30.00]),
            8:  np.array([5.00,  8.57, 12.14, 15.71, 19.29, 22.86, 26.43, 30.00]),
            9:  np.array([5.00,  8.12, 11.25, 14.38, 17.50, 20.62, 23.75, 26.88, 30.00]),
            10: np.array([5.00,  7.78, 10.56, 13.33, 16.11, 18.89, 21.67, 24.44, 27.22, 30.00]),
            11: np.array([5.00,  7.50, 10.00, 12.50, 15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00]),
            12: np.array([5.00,  7.27,  9.55, 11.82, 14.09, 16.36, 18.64, 20.91, 23.18, 25.45, 27.73, 30.00]),
            13: np.array([5.00,  7.08,  9.17, 11.25, 13.33, 15.42, 17.50, 19.58, 21.67, 23.75, 25.83, 27.92, 30.00]),
            14: np.array([5.00,  6.92,  8.85, 10.77, 12.69, 14.62, 16.54, 18.46, 20.38, 22.31, 24.23, 26.15, 28.08, 30.00]),
            15: np.array([5.00,  6.79,  8.57, 10.36, 12.14, 13.93, 15.71, 17.50, 19.29, 21.07, 22.86, 24.64, 26.43, 28.21, 30.00]),
            16: np.array([5.00,  6.67,  8.33, 10.00, 11.67, 13.33, 15.00, 16.67, 18.33, 20.00, 21.67, 23.33, 25.00, 26.67, 28.33, 30.00]),
            17: np.array([5.00,  6.56,  8.12,  9.69, 11.25, 12.81, 14.38, 15.94, 17.50, 19.06, 20.62, 22.19, 23.75, 25.31, 26.88, 28.44, 30.00])}
        elif "beamng" in self.scenario:
            predefined_point_distribution = {
            2:  np.array([ 5.00, 35.00]),
            3:  np.array([ 5.00, 20.00, 35.00]),
            4:  np.array([ 5.00, 15.00, 25.00, 35.00]),
            5:  np.array([ 5.00, 12.50, 20.00, 27.50, 35.00 ]),
            6:  np.array([ 5.00, 11.00, 17.00, 23.00, 29.00, 35.00]),
            7:  np.array([ 5.00, 10.00, 15.00, 20.00, 25.00, 30.00, 35.00]),
            8:  np.array([ 5.00,  9.29, 13.57, 17.86, 22.14, 26.43, 30.71, 35.00]),
            9:  np.array([ 5.00,  8.75, 12.50, 16.25, 20.00, 23.75, 27.50, 31.25, 35.00]),
            10: np.array([ 5.00,  8.33, 11.67, 15.00, 18.33, 21.67, 25.00, 28.33, 31.67, 35.00]),
            11: np.array([ 5.00,  8.00, 11.00, 14.00, 17.00, 20.00, 23.00, 26.00, 29.00, 32.00, 35.00]),
            12: np.array([ 5.00,  7.73, 10.45, 13.18, 15.91, 18.64, 21.36, 24.09, 26.82, 29.55, 32.27, 35.00]),
            13: np.array([ 5.00,  7.50, 10.00, 12.50, 15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00]),
            14: np.array([ 5.00,  7.31,  9.62, 11.92, 14.23, 16.54, 18.85, 21.15, 23.46, 25.77, 28.08, 30.38, 32.69, 35.00]),
            15: np.array([ 5.00,  7.14,  9.29, 11.43, 13.57, 15.71, 17.86, 20.00, 22.14, 24.29, 26.43, 28.57, 30.71, 32.86, 35.00]),
            16: np.array([ 5.00,  7.00,  9.00, 11.00, 13.00, 15.00, 17.00, 19.00, 21.00, 23.00, 25.00, 27.00, 29.00, 31.00, 33.00, 35.00]),
            17: np.array([ 5.00,  6.88,  8.75, 10.62, 12.50, 14.38, 16.25, 18.12, 20.00, 21.88, 23.75, 25.62, 27.50, 29.38, 31.25, 33.12, 35.00])}
        elif "waymo" in self.scenario:
            predefined_point_distribution = {
            2:  np.array([ 5.00, 35.00]),
            3:  np.array([ 5.00, 20.00, 35.00]),
            4:  np.array([ 5.00, 15.00, 25.00, 35.00]),
            5:  np.array([ 5.00, 12.50, 20.00, 27.50, 35.00 ]),
            6:  np.array([ 5.00, 11.00, 17.00, 23.00, 29.00, 35.00]),
            7:  np.array([ 5.00, 10.00, 15.00, 20.00, 25.00, 30.00, 35.00]),
            8:  np.array([ 5.00,  9.29, 13.57, 17.86, 22.14, 26.43, 30.71, 35.00]),
            9:  np.array([ 5.00,  8.75, 12.50, 16.25, 20.00, 23.75, 27.50, 31.25, 35.00]),
            10: np.array([ 5.00,  8.33, 11.67, 15.00, 18.33, 21.67, 25.00, 28.33, 31.67, 35.00]),
            11: np.array([ 5.00,  8.00, 11.00, 14.00, 17.00, 20.00, 23.00, 26.00, 29.00, 32.00, 35.00]),
            12: np.array([ 5.00,  7.73, 10.45, 13.18, 15.91, 18.64, 21.36, 24.09, 26.82, 29.55, 32.27, 35.00]),
            13: np.array([ 5.00,  7.50, 10.00, 12.50, 15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00]),
            14: np.array([ 5.00,  7.31,  9.62, 11.92, 14.23, 16.54, 18.85, 21.15, 23.46, 25.77, 28.08, 30.38, 32.69, 35.00]),
            15: np.array([ 5.00,  7.14,  9.29, 11.43, 13.57, 15.71, 17.86, 20.00, 22.14, 24.29, 26.43, 28.57, 30.71, 32.86, 35.00]),
            16: np.array([ 5.00,  7.00,  9.00, 11.00, 13.00, 15.00, 17.00, 19.00, 21.00, 23.00, 25.00, 27.00, 29.00, 31.00, 33.00, 35.00]),
            17: np.array([ 5.00,  6.88,  8.75, 10.62, 12.50, 14.38, 16.25, 18.12, 20.00, 21.88, 23.75, 25.62, 27.50, 29.38, 31.25, 33.12, 35.00])}
            

        return predefined_point_distribution

    def get_sample_distribution(self):
        sample_distribution = None
        if "highway" in self.scenario:
            sample_distribution = {
                1:  np.array([15]),
                2:  np.array([5, 5]), 
                3:  np.array([3, 4, 3]),
                4:  np.array([2, 3, 3, 4]),
                5:  np.array([4, 4, 4, 4, 4]),
                6:  np.array([4, 4, 4, 4, 4, 4]),
                7:  np.array([4, 4, 4, 4, 4, 4, 4]),
                8:  np.array([4, 4, 4, 4, 4, 4, 4, 4]),
                9:  np.array([4, 4, 4, 4, 4, 4, 4, 4, 4]),
                10: np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])}
        elif "beamng" in self.scenario:
            sample_distribution = {
                1:  np.array([10]),
                2:  np.array([5, 5]), 
                3:  np.array([3, 4, 3]),
                4:  np.array([4, 4, 4, 4]),
                5:  np.array([4, 4, 3, 4, 4]),
                6:  np.array([4, 4, 4, 4, 4, 4]),
                7:  np.array([4, 4, 4, 4, 4, 4, 4]),
                8:  np.array([4, 4, 4, 4, 4, 4, 4, 4]),
                9:  np.array([4, 4, 4, 4, 4, 4, 4, 4, 4]),
                10: np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])}
        elif "waymo" in self.scenario:
            sample_distribution = {
                1:  np.array([10]),
                2:  np.array([5, 5]), 
                3:  np.array([3, 4, 3]),
                4:  np.array([4, 4, 4, 4]),
                5:  np.array([4, 4, 3, 4, 4]),
                6:  np.array([4, 4, 4, 4, 4, 4]),
                7:  np.array([4, 4, 4, 4, 4, 4, 4]),
                8:  np.array([4, 4, 4, 4, 4, 4, 4, 4]),
                9:  np.array([4, 4, 4, 4, 4, 4, 4, 4, 4]),
                10: np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])}

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
                8:  [-26, -16, -10, -4, 4, 10, 16, 26],
                9:  [-28, -18, -12, -6, 0, 6, 12, 18, 28],
                10: [-30, -24, -15, -7, -2, 2, 7, 15, 24, 30]}
        elif "waymo" in self.scenario:
            angle_distribution = {
                1:  [0],
                2:  [-7, 7],
                3:  [-11, 0, 11],
                4:  [-17, -7, 7, 17],
                5:  [-20, -8, 0, 8, 20],
                6:  [-20, -10, -6, 6, 10, 20],
                7:  [-24, -14, -7, 0, 7, 14, 24],
                8:  [-26, -16, -10, -4, 4, 10, 16, 26],
                9:  [-28, -18, -12, -6, 0, 6, 12, 18, 28],
                10: [-30, -24, -15, -7, -2, 2, 7, 15, 24, 30]}
        return angle_distribution
