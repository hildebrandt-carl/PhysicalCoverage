import sys
from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import glob
import numpy as np
from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/media/carl/DataDrive/PhysicalCoverageData",     help="The location and name of the datafolder")
args = parser.parse_args()

results = glob.glob("/media/carl/DataDrive/PhysicalCoverageData/beamng/generated_tests/center_close/physical_coverage/processed/10000/traces_*.npy")

RRS_numbers = get_beam_number_from_file(results)
RRS_numbers = sorted(RRS_numbers)
trace_files = order_files_by_beam_number(results, RRS_numbers)

for trace_file, RRS in zip(trace_files, RRS_numbers):

    expected_RRS_list = []
    seen_RRS_list = []

    print("Analyzing RRS: {}".format(RRS))

    # Load the data file
    t = np.load(trace_file)

    # Get all test from this RRS number
    for i in range(np.shape(t)[0]):
        test_name = "/media/carl/DataDrive/PhysicalCoverageData/beamng/generated_tests/center_close/tests/{}_external_vehicles/test_{}.txt".format(RRS, i)
        expected_positions = []
        test = open(test_name, 'r')
        first_line = True
        for line in test:
            if first_line:
                expected_RRS = line[5:-1]
                first_line = False
            else:
                l = line.strip()
                l = l.split(',')
                x = float(l[1])
                y = float(l[2])
                expected_positions.append((x,y))
        test.close()

        expected_RRS_list.append(expected_RRS)
        
        # We now have expected positions and actual positions
        sensed_positions = t[i]

        seen_RRS = set()
        for s in sensed_positions:
            if not np.isnan(s).any():
                seen_RRS.add(tuple(s))

        seen_RRS_list.append(seen_RRS)

    seen_RRS_list_string = []
    for s in seen_RRS_list:
        new_RRS = []
        for i in list(s)[0]:
            new_RRS.append(int(i))
        seen_RRS_list_string.append(str(new_RRS))

    seen_RRS_list_string       = sorted(seen_RRS_list_string)
    expected_RRS_list          = sorted(expected_RRS_list)

    unseen_expected_RRS = []

    print("\nMatched:")
    for expected in expected_RRS_list:
        add = True
        for i, seen in enumerate(seen_RRS_list_string):
            if seen == expected:
                print("Expt: {}".format(expected))
                print("Seen: {}".format(seen))
                print("")
                seen_RRS_list_string.pop(i)
                add = False
                break
        if add:
            unseen_expected_RRS.append(expected)
        

    if len(unseen_expected_RRS) > 0:
        print("\nUnmatched Expected RRS")
        for s in unseen_expected_RRS:
            print(s)

    print("")

    if len(seen_RRS_list_string) > 0:
        print("Unmatched Seen RRS")
        for s in seen_RRS_list_string:
            print(s)

    print("------------------------------------------")