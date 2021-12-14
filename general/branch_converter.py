import numpy as np

class BranchConverter:
    def __init__(self, environment):
        self.environment = environment

        if self.environment == "highway":
            self.coverage_array = np.zeros(96)
            # These branches cant be hit because they are irrelevant in code
            self.impossible_branches = set([89, 96, 90, 73, 80, 50, 83, 52, 53, 54, 86, 87, 88, 92, 94])
            for i in self.impossible_branches:
                self.coverage_array[i-1] = -1
        else:
            print("requires implementation")


    def compute_branch_coverage(self, lines):

        if self.environment == "highway":

            # Break it up into two sets. One is the car_controller code and the other environment code
            cl    = set()
            el    = set()

            # Sort out the lines
            for l in lines:
                if l <= 2000:
                    el.add(l - 1000)
                else:
                    cl.add(l - 2000)

            # manually mark off different branches
            # b1
            if (15 in cl) or (17 in cl):
                self.coverage_array[0] += 1

            #b2
            if (14 in cl):
                self.coverage_array[1] += 1

            #b3
            if (15 in cl):
                self.coverage_array[2] += 1

            #b4
            if (17 in cl):
                self.coverage_array[3] += 1

            # b5 -- Impossible to tell with current metric
            self.coverage_array[4] = -1

            #b6
            if (21 in cl):
                self.coverage_array[5] += 1

            #b7
            if (28 in cl):
                self.coverage_array[6] += 1

            #b8 -- Impossible to tell with current metric
            self.coverage_array[7] = -1

            #b9
            if (41 in cl):
                self.coverage_array[8] += 1

            #b10 -- Impossible to tell with current metric
            self.coverage_array[9] = -1

            #b11
            if (47 in cl):
                self.coverage_array[10] += 1

            #b12 -- Impossible to tell with current metric
            self.coverage_array[11] = -1

            #b13
            if (49 in cl):
                self.coverage_array[12] += 1

            #b14 -- Impossible to tell with current metric
            self.coverage_array[13] = -1

            #b15
            if (50 in cl):
                self.coverage_array[14] += 1

            #b16
            if (54 in cl):
                self.coverage_array[15] += 1

            #b17
            if (51 in cl):
                self.coverage_array[16] += 1

            #b18 -- Impossible to tell with current metric
            self.coverage_array[17] = -1

            #b19
            if (53 in cl):
                self.coverage_array[18] += 1

            #b20 -- Impossible to tell with current metric
            self.coverage_array[19] = -1

            #b21
            if (57 in cl):
                self.coverage_array[20] += 1

            #b22
            if (61 in cl):
                self.coverage_array[21] += 1

            #b23
            if (58 in cl):
                self.coverage_array[22] += 1

            #b24
            if (66 in cl):
                self.coverage_array[23] += 1

            #b25 -- Impossible to tell with current metric
            self.coverage_array[24] = -1

            #b26
            if (64 in cl):
                self.coverage_array[25] += 1

            #b27 -- Impossible to tell with current metric
            self.coverage_array[26] = -1

            #b28
            if (60 in cl):
                self.coverage_array[27] += 1

            #b29 -- Impossible to tell with current metric
            self.coverage_array[28] = -1

            #b30
            if (68 in cl):
                self.coverage_array[29] += 1

            #b31
            if (72 in cl):
                self.coverage_array[30] += 1

            #b32
            if (69 in cl):
                self.coverage_array[31] += 1

            #b33
            if (77 in cl):
                self.coverage_array[32] += 1

            #b34
            if (73 in cl):
                self.coverage_array[33] += 1

            #b35 -- Impossible to tell with current metric
            self.coverage_array[34] = -1

            #b36
            if (71 in cl):
                self.coverage_array[35] += 1

            #b37 -- Impossible to tell with current metric
            self.coverage_array[36] = -1

            #b38 
            if (75 in cl):
                self.coverage_array[37] += 1

            #b39 -- Impossible to tell with current metric
            self.coverage_array[38] = -1

            #b40
            if (79 in cl):
                self.coverage_array[39] += 1

            #b41
            if (84 in cl):
                self.coverage_array[40] += 1

            #b42
            if (80 in cl):
                self.coverage_array[41] += 1
            
            #b43 -- Impossible to tell with current metric
            self.coverage_array[42] = -1

            #b44
            if (82 in cl):
                self.coverage_array[43] += 1

            #b45 -- Impossible to tell with current metric
            self.coverage_array[44] = -1

            #b46
            if (86 in cl):
                self.coverage_array[45] += 1

            #b47 -- Impossible to tell with current metric
            self.coverage_array[46] = -1

            #b48
            if (87 in cl):
                self.coverage_array[47] += 1

            #b49 -- Impossible to tell with current metric
            self.coverage_array[48] = -1

            #b50
            if (34 in el):
                self.coverage_array[49] += 1

            #b51 -- Impossible to tell with current metric
            self.coverage_array[50] = -1

            #b52
            if (8 in el):
                self.coverage_array[51] += 1

            #b53
            if (12 in el):
                self.coverage_array[52] += 1

            #b54
            if (10 in el):
                self.coverage_array[53] += 1

            #b55
            if (17 in el):
                self.coverage_array[54] += 1

            #b56
            if (16 in el):
                self.coverage_array[55] += 1

            #b57
            if (19 in el):
                self.coverage_array[56] += 1

            #b58
            if (18 in el):
                self.coverage_array[57] += 1

            #b59
            if (25 in el):
                self.coverage_array[58] += 1

            #b60
            if (20 in el):
                self.coverage_array[59] += 1

            #b61 -- Impossible to tell with current metric
            self.coverage_array[60] = -1

            #b62
            if (24 in el):
                self.coverage_array[61] += 1

            #b63 -- Impossible to tell with current metric
            self.coverage_array[62] = -1

            #b64
            if (28 in el):
                self.coverage_array[63] += 1

            #b65 -- Impossible to tell with current metric
            self.coverage_array[64] = -1

            #b66
            if (29 in el):
                self.coverage_array[65] += 1

            #b67
            if (52 in el):
                self.coverage_array[66] += 1

            #b68 -- Impossible to tell with current metric
            self.coverage_array[67] = -1

            #b69
            if (54 in el):
                self.coverage_array[68] += 1

            #70
            if (53 in el):
                self.coverage_array[69] += 1

            #b71 -- Impossible to tell with current metric
            self.coverage_array[70] = -1

            #b72
            if (55 in el):
                self.coverage_array[71] += 1

            #b73
            if (64 in el):
                self.coverage_array[72] += 1

            #b74
            if (62 in el):
                self.coverage_array[73] += 1

            #b75 -- Impossible to tell with current metric
            self.coverage_array[74] = -1

            #b76 -- Unlisted
            self.coverage_array[75] += 1

            #b77
            self.coverage_array[76] = -1 # What happens if we ignore check collision code
            # if (71 in el):
            #     self.coverage_array[76] += 1

            #b78
            self.coverage_array[77] = -1 # What happens if we ignore check collision code
            # if (74 in el):
            #     self.coverage_array[77] += 1

            #b79
            self.coverage_array[78] = -1 # What happens if we ignore check collision code
            # if (73 in el):
            #     self.coverage_array[78] += 1

            #b80
            self.coverage_array[79] = -1 # What happens if we ignore check collision code
            # if (83 in el):
            #     self.coverage_array[79] += 1

            #b81
            self.coverage_array[80] = -1 # What happens if we ignore check collision code
            # if (75 in el):
            #     self.coverage_array[80] += 1

            #b82
            self.coverage_array[81] = -1 # What happens if we ignore check collision code
            # if (77 in el):
            #     self.coverage_array[81] += 1

            #b83
            self.coverage_array[82] = -1 # What happens if we ignore check collision code
            # if (76 in el):
            #     self.coverage_array[82] += 1

            #b84
            self.coverage_array[83] = -1 # What happens if we ignore check collision code
            # if (78 in el):
            #     self.coverage_array[83] += 1

            #b85 -- Impossible to tell with current metric
            self.coverage_array[84] = -1

            #b86
            self.coverage_array[85] = -1 # What happens if we ignore check collision code
            # if (81 in el):
            #     self.coverage_array[85] += 1

            #b87
            self.coverage_array[86] = -1 # What happens if we ignore check collision code
            # if (84 in el):
            #     self.coverage_array[86] += 1

            #b88
            self.coverage_array[87] = -1 # What happens if we ignore check collision code
            # if (91 in el):
            #     self.coverage_array[87] += 1

            #b89
            self.coverage_array[88] = -1 # What happens if we ignore check collision code
            # if (86 in el):
            #     self.coverage_array[88] += 1

            #b90
            self.coverage_array[89] = -1 # What happens if we ignore check collision code
            # if (85 in el):
            #     self.coverage_array[89] += 1

            #b91 -- Impossible to tell with current metric
            self.coverage_array[90] = -1

            #b92
            self.coverage_array[91] = -1 # What happens if we ignore check collision code

            #b93 -- Impossible to tell with current metric
            self.coverage_array[92] = -1

            #b94
            self.coverage_array[93] = -1 # What happens if we ignore check collision code
            # if (90 in el):
            #     self.coverage_array[93] += 1

            #b95 -- Impossible to tell with current metric
            self.coverage_array[94] = -1

            #b96
            self.coverage_array[95] = -1 # What happens if we ignore check collision code
            # if (92 in el):
            #     self.coverage_array[95] += 1

        self.coverage_array = np.clip(self.coverage_array, -1, 1)
        return self.coverage_array



        