import re
import sys
import ast
import glob
import random
import argparse

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)


from tqdm import tqdm
from general_functions import get_ignored_code_coverage_lines

from general.branch_converter import BranchConverter

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--scenario',           type=str, default="",     help="beamng/highway")
args = parser.parse_args()


files = glob.glob("../../PhysicalCoverageData/{}/random_tests/code_coverage/raw/*/*.txt".format(args.scenario))
random.shuffle(files)
print("Processing: {} files".format(len(files)))

all_lines_coverage = set()
all_possible_lines = set()
ignored_lines = get_ignored_code_coverage_lines(args.scenario)
coverage_array = []

# Get the branch_converter
bc = BranchConverter(args.scenario)

# Init the branch coverage array
branch_coverage_array = None

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

    # Get the branch coverage
    branch_coverage = bc.compute_branch_coverage(covered_l)

    # Save the branch coverage
    if branch_coverage_array is None:
        branch_coverage_array = branch_coverage
    else:
        branch_coverage_array = np.maximum(branch_coverage, branch_coverage_array)

    # Get the count of all branches
    total_possible_branches = len(branch_coverage)

    # Get the total number of impossible to determine branches
    undetermined_branches = (branch_coverage <= -1).sum()

    # Get the count branches covered
    covered_branches = (branch_coverage >= 1).sum()

    # Get the count of branches uncovered
    uncovered_branches = (branch_coverage == 0).sum()

    # Get the coverage and save it
    cov = ((covered_branches) / (total_possible_branches - undetermined_branches)) * 100
    coverage_array.append(cov)

print("Total branches: {}".format(total_possible_branches))
print("Undetermined branches: {}".format(undetermined_branches))
print("Covered branches: {}".format(covered_branches))
print("Uncovered branches: {}".format(uncovered_branches))
print("--------------------")

# Print the uncovered branches (add 1 because on the diagram branches start from 1 not 0 :/ sigh)
uncovered_indices = set(np.argwhere(branch_coverage_array < 0.1).reshape(-1) + 1)
undetermined_indices = set(np.argwhere(branch_coverage_array < -0.1).reshape(-1) + 1)
print("undetermined_branches branch indices: {}".format(undetermined_indices))
print("Uncovered branch indices: {}".format(uncovered_indices - undetermined_indices))
print("--------------------")

lines_not_coverage = sorted(list(all_possible_lines - all_lines_coverage))
print("The lines not covered are: {}".format(lines_not_coverage))

plt.plot(coverage_array)
plt.ylim([-5,105])
plt.grid(alpha=0.5)
plt.yticks(np.arange(0, 100.01, step=5))
plt.xlabel("Number of tests")
plt.ylabel("Branch Coverage (%)")
plt.title("Total Branch Coverage: {}%".format(np.round(coverage_array[-1], 2)))
plt.show()