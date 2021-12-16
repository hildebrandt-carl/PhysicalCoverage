

import os
import glob
import copy
import argparse
import multiprocessing

import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm

def compute_coverage(file_name, save_path):
    global preprocessed_crash_arrays
    global preprocessed_file_arrays

    # Load the data
    data = ET.parse(file_name)

    # Save this files coverage data
    coverage_data = {}

    # Loop through the different files
    for c in data.iter('class'):

        all_lines = []
        lines_covered = []

        all_branches = []
        branches_covered = []

        # Get the current class
        current_class = c.attrib['filename']
        current_class = current_class[current_class.rfind("/")+1:]

        # Loop through each line
        for line in c.iter("line"):

            # Compute the line coverage
            line_number = int(line.attrib["number"])
            line_hit    = int(line.attrib["hits"])

            # Save the data
            all_lines.append(line_number)
            if line_hit:
                lines_covered.append(line_number)

            # Compute the branch coverage
            if "branch" in line.attrib.keys():
                branch_number     = str(line.attrib["number"])
                branch_coverage   = str(line.attrib["condition-coverage"])
                branch_hits       = int(branch_coverage[branch_coverage.rfind("(")+1:branch_coverage.rfind("/")])

                # Determine if we hit both branches
                if branch_hits == 2:
                    branches_covered.append(branch_number)
                elif branch_hits == 1:
                    # Determine which branch is missing
                    missing_branch = line.attrib["missing-branches"]
                    branches_covered.append(str(branch_number) + "_" + missing_branch)

                # Save this as part of all branches
                all_branches.append(branch_number)

        # Save this information
        coverage_data[current_class] = [lines_covered, all_lines, branches_covered, all_branches]
    
    # Compute the save name
    external_vehicles_str = file_name[file_name.rfind("raw/")+4:file_name.rfind("/")]
    save_name = file_name[file_name.rfind("/")+1:-4] + ".txt"
    code_coverage_save_name = save_path + external_vehicles_str + "/" + save_name

    # Figure out which crash array to look at:
    c_array = preprocessed_crash_arrays[external_vehicles_str]
    f_array = preprocessed_file_arrays[external_vehicles_str]

    # Get the array index
    arr_index = np.where(f_array == save_name)
    if f_array[arr_index][0] != save_name:
        print("Error: The name we associate with crashes does not match the code coverage file")
        exit()

    # Get the crash count    
    crash_count = np.sum(~np.isinf(c_array[arr_index]))

    return [coverage_data, code_coverage_save_name, crash_count]

parser = argparse.ArgumentParser()
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

print("----------------------------------")
print("----------Locating Files----------")
print("----------------------------------")

all_files = None
if args.scenario == "beamng_random":
    print("To be implemented")
    exit()
elif args.scenario == "beamng_generated":
    print("To be implemented")
    exit()
elif args.scenario == "highway_random":
    base = "../../PhysicalCoverageData/highway/random_tests"
    all_files = glob.glob(base + "/code_coverage/raw/*/*.xml")
    crash_info = glob.glob(base + "/physical_coverage/processed/{}/crash_hash*.npy".format(args.total_samples))
    file_info = glob.glob(base + "/physical_coverage/processed/{}/processed_files*.npy".format(args.total_samples))
elif args.scenario == "highway_generated":
    print("To be implemented")
    exit()
else:
    print("Error: Scenario not known")
    exit()

total_files = len(all_files)
print("Total files found: " + str(total_files))

# Select all of the files
file_names = all_files

# If no files are found exit
if len(file_names) <= 0:
    print("No files found")
    exit()

# Flatten the list
if len(np.shape(file_names)) > 1:
    file_names_flat = []
    for subl in file_names:
        for item in subl:
            file_names_flat.append(item)
    file_names = file_names_flat

# Get the file size
total_files = len(file_names)
print("Total files selected for processing: " + str(total_files))

if (len(crash_info) == 0) or (len(file_info) == 0):
    print("Please run the preprocessing data first before runing this")
    exit()

# Load the crash information
global preprocessed_crash_arrays
global preprocessed_file_arrays
preprocessed_crash_arrays = {}
preprocessed_file_arrays = {}
for i in range(10):
    # Find the index of the correct file
    c_index = [idx for idx, s in enumerate(crash_info) if '_b{}_'.format(i+1) in s][0]
    f_index = [idx for idx, s in enumerate(file_info) if '_b{}_'.format(i+1) in s][0]
    c_name = crash_info[c_index]
    f_name = file_info[f_index]
    preprocessed_crash_arrays["{}_external_vehicles".format(i+1)] = copy.deepcopy(np.load(c_name))
    preprocessed_file_arrays["{}_external_vehicles".format(i+1)] = copy.deepcopy(np.load(f_name))

print("----------------------------------")
print("-----Creating Output Location-----")
print("----------------------------------")

all_files = None
if args.scenario == "beamng_random":
    print("To be implemented")
    exit()
elif args.scenario == "beamng_generated":
    print("To be implemented")
    exit()
elif args.scenario == "highway_random":
    save_path = "../output/highway/random_tests/code_coverage/processed/{}/".format(args.total_samples)
elif args.scenario == "highway_generated":
    print("To be implemented")
    exit()
else:
    print("Error: Scenario not known")
    exit()

# # Create the output directory if it doesn't exists
for i in range(10):
    new_path = save_path + "{}_external_vehicles/".format(i+1)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

print("Location created: {}".format(save_path))

print("----------------------------------")
print("---------Processing files---------")
print("----------------------------------")

total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
for i in range(total_files):
    jobs.append(pool.apply_async(compute_coverage, args=([file_names[i], save_path])))

# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

print("----------------------------------")
print("---------Saving the data----------")
print("----------------------------------")

for r in tqdm(results):

    # Get the coverage data and save name
    coverage_data, code_coverage_save_name, total_physical_accidents = r 

    # Open the file for writing
    f = open(code_coverage_save_name, "w")

    # For each of the subfiles
    for key in coverage_data:

        # Get the data
        lines_covered       = coverage_data[key][0]
        all_lines           = coverage_data[key][1]
        branches_covered    = coverage_data[key][2]
        all_branches        = coverage_data[key][3]

        # Save the data
        f.write("-----------------------------\n")
        f.write("File: {}\n".format(key))
        f.write("-----------------------------\n")
        f.write("Lines covered: {}\n".format(sorted(list(lines_covered))))
        f.write("Total lines covered: {}\n".format(len(list(lines_covered))))
        f.write("-----------------------------\n")
        f.write("All lines: {}\n".format(sorted(list(all_lines))))
        f.write("Total lines: {}\n".format(len(list(all_lines))))
        f.write("-----------------------------\n")
        f.write("Branches covered: {}\n".format(sorted(list(branches_covered))))
        f.write("Total branches covered: {}\n".format(len(list(branches_covered))))
        f.write("-----------------------------\n")
        f.write("All branches: {}\n".format(sorted(list(all_branches))))
        f.write("Total branches: {}\n".format(len(list(all_branches))))

    # Save the total number of crashes
    f.write("-----------------------------\n")
    f.write("Total physical crashes: {}\n".format(total_physical_accidents))
    f.write("-----------------------------\n")
    f.close()