

import os
import glob
import argparse
import multiprocessing

import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm

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
    all_files = glob.glob("../../PhysicalCoverageData/highway/random_tests/code_coverage/raw/*/*.xml")
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

print("----------------------------------")
print("-----Creating Output Location------")
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

for i in tqdm(range(total_files)):

    # Get the file name
    file_name = file_names[i]

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
                branch_number   = int(line.attrib["number"])
                branch_coverage = str(line.attrib["condition-coverage"])
                branch_hits     = int(branch_coverage[branch_coverage.rfind("(")+1:branch_coverage.rfind("/")])
                
                # Save the data (Each branch has two branches (hence two numbers))
                all_branches.append(branch_number)
                all_branches.append(branch_number)
                for _ in range(branch_hits):
                    branches_covered.append(branch_number)

        # Compute the line coverage
        line_coverage = (len(lines_covered) / len(all_lines)) * 100
        line_coverage = np.round(line_coverage, 2)
        # print("Line Coverage: {}/{} - {}%".format(len(lines_covered), len(all_lines), line_coverage))
        # Compute the branch coverage
        branch_coverage = (len(branches_covered) / len(all_branches)) * 100
        branch_coverage = np.round(branch_coverage, 2)
        # print("Branch Coverage: {}/{} - {}%".format(len(branches_covered), len(all_branches), branch_coverage))

        # Save this information
        coverage_data[current_class] = [lines_covered, all_lines, branches_covered, all_branches]

    # Save the file
    save_name = file_name[file_name.rfind("raw/")+4:-4] + ".txt"
    code_coverage_save_name = save_path + save_name

    f = open(code_coverage_save_name, "w")

    # For each of the subfiles
    for key in coverage_data:

        # Get the data
        lines_covered       = coverage_data[key][0]
        all_lines           = coverage_data[key][1]
        branches_covered    = coverage_data[key][2]
        all_branches        = coverage_data[key][3]

        total_physical_accidents = 0

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
        f.write("-----------------------------\n")


    f.close()