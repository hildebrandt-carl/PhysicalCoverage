import re
import glob
import random
import argparse
import multiprocessing

from tqdm import tqdm
from general_functions import get_lines_covered
from general_functions import clean_branch_data
from general_functions import get_ignored_lines
from general_functions import get_ignored_branches

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

# Get the files
file_path = "../../PhysicalCoverageData/{}/random_tests/code_coverage/processed/{}/*/*.txt".format(args.scenario, args.total_samples)
files = glob.glob(file_path)
random.shuffle(files)
print("Processing: {} files".format(len(files)))

# Keep track of all the different sets
lines_covered_set       = set()
all_lines_set           = set()
branches_covered_set    = set()
all_branches_set        = set()

# used to save the coverage
line_coverage_array = []
branch_coverage_array = []

# Get the ignored lines
ignored_lines       = set(get_ignored_lines(args.scenario))
ignored_branches    = set(get_ignored_branches(args.scenario))

# Start the multiprocessing
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
# for i in range(50000):
    # f = files[i]
for f in files:
    jobs.append(pool.apply_async(get_lines_covered, args=([f])))

# Get the results
for job in tqdm(jobs):
    result = job.get()

    # Expand the data
    lines_covered           = result[0]
    all_lines               = result[1]
    branches_covered        = result[2]
    all_branches            = result[3]

    # Make sure converting to a set was done correctly
    assert(len(lines_covered)       == len(set(lines_covered)))
    assert(len(all_lines)           == len(set(all_lines)))
    assert(len(branches_covered)    == len(set(branches_covered)))
    assert(len(all_branches)        == len(set(all_branches)))

    # Save them to the overall sets
    lines_covered_set       = lines_covered_set     | set(lines_covered)
    all_lines_set           = all_lines_set         | set(all_lines)
    branches_covered_set    = branches_covered_set  | set(branches_covered)
    all_branches_set        = all_branches_set      | set(all_branches)

    # Clean the branch data
    all_branches_set_clean, branches_covered_set_clean = clean_branch_data(all_branches_set, branches_covered_set)

    # Remove the ignored lines
    lines_covered_set           -= ignored_lines
    all_lines_set               -= ignored_lines
    all_branches_set_clean      -= ignored_branches
    branches_covered_set_clean  -= ignored_branches

    # Make sure it all makes sense
    assert(len(all_lines_set) == len(all_lines_set | lines_covered_set))
    assert(len(all_branches_set_clean) == len(all_branches_set_clean | branches_covered_set_clean))

    # Save the coverage
    line_coverage = (len(lines_covered_set) / len(all_lines_set)) * 100
    line_coverage_array.append(line_coverage)
    branch_coverage = (len(branches_covered_set_clean) / len(all_branches_set_clean)) * 100
    branch_coverage_array.append(branch_coverage)

# Close the pool
pool.close()

# Save the branch and all branches set
branches_covered_set = branches_covered_set_clean
all_branches_set = all_branches_set_clean

# Print the statistics
print("Total lines: {}".format(len(all_lines_set)))
print("Total lines covered: {}".format(len(lines_covered_set)))
print("Total branches: {}".format(len(all_branches_set)))
print("Total branches covered: {}".format(len(branches_covered_set)))

# Print line details
print("")
print("All lines: {}".format(sorted(list(all_lines_set))))
print("Lines covered: {}".format(sorted(list(lines_covered_set))))
print("Lines not covered: {}".format(sorted(list(all_lines_set - lines_covered_set))))

# Print branch details
print("")
print("All branches: {}".format(sorted(list(all_branches_set))))
print("Branches covered: {}".format(sorted(list(branches_covered_set))))
print("Branches not covered: {}".format(sorted(list(all_branches_set - branches_covered_set))))
print("")

plt.plot(line_coverage_array, label="Line Coverage")
plt.plot(branch_coverage_array, label="Branch Coverage")
plt.ylim([-5,105])
plt.grid(alpha=0.5)
plt.yticks(np.arange(0, 100.01, step=5))
plt.xlabel("Number of tests")
plt.ylabel("Coverage (%)")
plt.legend()
plt.show()