import re
import ast
import glob
import argparse

from tqdm import tqdm
from general_functions import get_ignored_code_coverage_lines

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--scenario',           type=str, default="",     help="beamng/highway")
args = parser.parse_args()


files = glob.glob("../../PhysicalCoverageData/{}/random_tests/code_coverage/raw/*/*.txt".format(args.scenario))
print("Processing: {} files".format(len(files)))

all_lines_coverage = set()
all_possible_lines = set()
ignored_lines = get_ignored_code_coverage_lines(args.scenario)
coverage_array = []

for f in tqdm(files):

    f = open(f, "r")

    # Read the file
    for line in f: 
        if "Lines covered:" in line:
            covered_l = ast.literal_eval(line[15:])

        if "Total lines covered:" in line:
            total_covered_l = int(line[21:])
            assert(len(covered_l) == total_covered_l)

        if "All Lines:" in line:
            all_l = ast.literal_eval(line[11:])

        if "Total Lines:" in line:
            total_all_l = int(line[13:])
            assert(len(all_l) == total_all_l)

    # Close the file
    f.close()

    # Keep track of the lines covered
    all_lines_coverage = all_lines_coverage | set(covered_l)

    # Get all the possible lines
    if len(all_possible_lines) <= 0:
        all_possible_lines = set(all_l) - ignored_lines
    else:
        assert(all_possible_lines == (set(all_l) - ignored_lines))

    # Remove the ignored lines from all possible lines and lines covered
    all_lines_coverage = all_lines_coverage - ignored_lines
    all_possible_lines = all_possible_lines - ignored_lines

    # Compute the coverage
    coverage = (len(all_lines_coverage) / len(all_possible_lines)) * 100
    coverage_array.append(coverage)

lines_not_coverage = sorted(list(all_possible_lines - all_lines_coverage))
print("The lines not covered are: {}".format(lines_not_coverage))

plt.plot(coverage_array)
plt.ylim([-5,105])
plt.grid(alpha=0.5)
plt.yticks(np.arange(0, 100.01, step=5))
plt.xlabel("Number of tests")
plt.ylabel("Code Coverage (%)")
plt.title("Total Code Coverage: {}%".format(np.round(coverage_array[-1], 2)))
plt.show()