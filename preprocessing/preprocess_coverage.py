

import os
import ast
import glob
import time
import copy
import hashlib
import argparse
import multiprocessing

from ast import literal_eval

import numpy as np
import xml.etree.ElementTree as ET

from ordered_set import OrderedSet

from tqdm import tqdm

def is_simple(A):
    if len(A) == 1 or len(A) == 2:
        return True

    if len(A) == 3:
        if (A[0] != A[1]) and (A[1] != A[2]):
            return True
        else:
            return False

    path = A
    if A[0] == A[-1]:
        path = A[1:-1]

    p1 = list(OrderedSet(path)) 
    p2 = path
    return np.array_equal(p1, p2)

def is_prime_path(A, B):
    # Two pointers to traverse the arrays
    i = 0; j = 0
    # Traverse both arrays simultaneously
    while (i < len(A) and j < len(B)):
        # If element matches
        # increment both pointers
        if (A[i] == B[j]):
            i += 1
            j += 1
            # If array B is completely
            # traversed
            if (j == len(A)):
                return False
        # If not,
        # increment i and reset j
        else:
            i = i - j + 1
            j = 0
    return True

def find_prime_paths(paths):
    prime_paths = []
    # Loop through the paths
    for i, p1 in enumerate(paths):
        is_prime = True
        for j, p2 in enumerate(paths):
            if i != j:
                if not is_simple(p1):
                    is_prime = False
                    break
                if not is_prime_path(p1, p2):
                    is_prime = False
                    break
        if is_prime:
            prime_paths.append(p1)
    return prime_paths

def is_float(input):
    try:
        num_int = int(input)
        return True
    except ValueError:
        return False

def intraprocedural_path_coverage(path_taken, loops_allowed=True, prime_paths=False):

    # For the paths
    intraprocedural_path    = {}

    # Process the path
    function_stack          = []
    for j, b in enumerate(path_taken):

        # Compute the print tabs
        print_tabs = ""
        for i in range(len(function_stack)):
            print_tabs += "  "

        # Check if its a string
        if is_float(b) == False:

            # Check if this is the start or end of a function
            if "enter_" in b:
                # Get the function name
                f_name = b[6:]
                # Append this function to the stack and the current path to a stack
                function_stack.append(([], f_name))
            elif "exit_" in b:
                # Pop the stack
                print_tabs = print_tabs[:-2]
                i_path, f_name = function_stack.pop()

                # Remove loops if loops are not allowed
                if not loops_allowed:
                    # i_path = list(OrderedSet(i_path))
                    i_path = sorted(list(set(i_path)))
                
                # Insert this into the dictionary
                if f_name in intraprocedural_path:
                    intraprocedural_path[f_name].append(i_path)
                else:
                    intraprocedural_path[f_name] = [i_path]

                # print("{}{} end - {}..{}".format(print_tabs, f_name, i_path[0:5], i_path[-5:-1]))
            else:
                print("Error: string not understood: {}".format(b))
                exit()
        # It must be a branch number
        else:
            # Get the true branch for comparing against start and end
            try:
                branch_number = int(b)
                branch = get_true_branch_number(branch_number)
            
                # Push the current branch to the most recent function in the stack
                function_stack[-1][0].append(branch_number)
                # print("{}Stack size: {} - Processed: {}".format(print_tabs, len(function_stack), branch_number))
            except Exception as e:
                print("----------------")
                print(path_taken[j])
                print("Error: {} - Path: {}".format(b, path_taken[j-5:j+5]))
                exit()

    # Compute the true intraprocedural path
    for key in intraprocedural_path:
        path_set = set()
        for path in intraprocedural_path[key]:
            path_set.add(str(path))
        intraprocedural_path[key] = path_set
    
    if prime_paths:
        # Convert each of the intraprocedural paths to prime paths
        for key in intraprocedural_path:
            list_of_paths = []
            for path in intraprocedural_path[key]:
                list_of_paths.append(literal_eval(path))
            prime_paths = find_prime_paths(list_of_paths)
            intraprocedural_path[key] = []
            for p in prime_paths:
                intraprocedural_path[key].append(str(p))

    # Convert to list (make sure that both the order of the functions, and the order of the paths are always sorted the same)
    all_keys = sorted(list(intraprocedural_path.keys()))
    final_intraprocedural_path = []
    for key in all_keys:
        final_intraprocedural_path.append(sorted(list(intraprocedural_path[key])))

    return final_intraprocedural_path
        
def get_true_branch_number(branch_number):
    if branch_number <= 0:
        return branch_number
    else:
        return branch_number if (branch_number % 2) == 0 else  branch_number - 1

def compute_coverage_beamng(file_name, save_path):

    # Compute the save name
    save_name = file_name[file_name.rfind("/")+1:]
    external_vehicles_str = file_name[:file_name.find("_external_vehicles")+18]
    external_vehicles_str = external_vehicles_str[external_vehicles_str.rfind("/")+1:] + "/"
    code_coverage_save_name = save_path + external_vehicles_str + save_name

    # Create the coverage line
    coverage_data = {}
    lines_covered = []
    all_lines = []
    branches_covered = []
    all_branches = []
    crash_count = -1

    # Read the file
    with open(file_name, "r") as f:
        for line in f:
            # Get the data
            if "Lines covered:" in line[0:15]:
                lines_covered       = ast.literal_eval(line[15:-1])
            elif "All lines:" in line[0:11]:
                all_lines           = ast.literal_eval(line[11:-1])
            elif "Branches covered:" in line[0:18]:
                branches_covered    = ast.literal_eval(line[18:-1])
            elif "All branches:" in line[0:14]:
                all_branches        = ast.literal_eval(line[14:-1])
            elif "Total physical crashes:" in line[0:23]:
                crash_count         = int(line[23:-1])
            elif "Path Taken:" in line[0:12]:
                path_taken          = line[12:-1]  

    # Get the path taken
    path_taken = path_taken.split(", ")
    # Remove all 0's
    index = 0
    while (path_taken[index] != '0') and (index < len(path_taken) - 1):
        index += 1  
    path_taken = path_taken[:index]

    # Make sure the last part of the path is "exit_isDriving"
    if(path_taken[-1] == "exit_isDriving"):

        # Compute intraprocedural path coverage with loops
        intra_path_taken = intraprocedural_path_coverage(path_taken, loops_allowed=True, prime_paths=False)

        # Get the intraprocedural path_signature
        intra_path_string = ''.join([str(x) + "," for x in intra_path_taken])
        intra_path_signature = hashlib.md5(intra_path_string.encode()).hexdigest()
        # Memory cleanup
        intra_path_string = None

        # Compute intraprocedural path coverage with loops
        intra_prime_path_taken = intraprocedural_path_coverage(path_taken, loops_allowed=True, prime_paths=True)

        # Get the intraprocedural path_signature
        intra_prime_path_string = ''.join([str(x) + "," for x in intra_prime_path_taken])
        intra_prime_path_signature = hashlib.md5(intra_prime_path_string.encode()).hexdigest()
        # Memory cleanup
        intra_prime_path_string = None

        # Get the absolute path_signature
        absolute_path_string = ''.join([str(x) + "," for x in path_taken])
        absolute_path_signature = hashlib.md5(absolute_path_string.encode()).hexdigest()
    else:
        # The path does not end with isDriving and thus we cant compute the absolute path signature.
        intra_path_signature = None
        intra_prime_path_signature = None
        absolute_path_signature = None

    # Convert the branch data to strings to match the highwayenv data
    branches_covered = [str(x) for x in branches_covered]
    all_branches = [str(x) for x in all_branches]

    coverage_data["ai.lua"] = [lines_covered, all_lines, branches_covered, all_branches, intra_prime_path_signature, intra_path_signature, absolute_path_signature]
    return [coverage_data, code_coverage_save_name, crash_count]

def compute_coverage_highway(file_name, save_path):
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
        coverage_data[current_class] = [lines_covered, all_lines, branches_covered, all_branches, None, None, None]
    
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

    # This will throw an error we need a isNone function
    # Get the crash count    
    crash_count = np.sum(~np.isnan(c_array[arr_index]))

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
    base = "/media/carl/DataDrive/PhysicalCoverageData/beamng/random_tests"
    all_files = glob.glob(base + "/code_coverage/raw/*/*.txt")
elif args.scenario == "beamng_generated":
    print("To be implemented")
    exit()
elif args.scenario == "highway_random":
    base = "/media/carl/DataDrive/PhysicalCoverageData/highway/random_tests"
    all_files = glob.glob(base + "/code_coverage/raw/*/*.xml")
    crash_info = glob.glob(base + "/physical_coverage/processed/center_close/{}/crash_hash*.npy".format(args.total_samples))
    file_info = glob.glob(base + "/physical_coverage/processed/center_close/{}/processed_files*.npy".format(args.total_samples))
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

if "highway" in args.scenario:
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

all_files = None
if args.scenario == "beamng_random":
    save_path = "../output/beamng/random_tests/code_coverage/processed/{}/".format(args.total_samples)
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

print("----------------------------------")
print("---------Processing files---------")
print("----------------------------------")

print("\n\nComputing line, branch and path coverage")
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
for i in range(total_files):
    if "highway" in args.scenario:
        jobs.append(pool.apply_async(compute_coverage_highway, args=([file_names[i], save_path])))
    elif "beamng" in args.scenario:
        jobs.append(pool.apply_async(compute_coverage_beamng, args=([file_names[i], save_path])))

# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

# Close the pool
pool.close()

print("----------------------------------")
print("---------Saving the data----------")
print("----------------------------------")

# Create the output directory if it doesn't exists
for i in range(10):
    new_path = save_path + "{}_external_vehicles/".format(i+1)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

print("Location created: {}".format(save_path))

for r in tqdm(results):

    # Get the coverage data and save name
    coverage_data, code_coverage_save_name, total_physical_accidents = r 

    # Open the file for writing
    f = open(code_coverage_save_name, "w")

    # For each of the subfiles
    for key in coverage_data:

        # Get the data
        lines_covered                   = coverage_data[key][0]
        all_lines                       = coverage_data[key][1]
        branches_covered                = coverage_data[key][2]
        all_branches                    = coverage_data[key][3]
        intra_prime_path_signature      = coverage_data[key][4]
        intra_path_signature            = coverage_data[key][5]
        absolute_path_signature         = coverage_data[key][6]

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
        f.write("Intraprocedural prime path signature: {}\n".format(intra_prime_path_signature))
        f.write("Intraprocedural path signature: {}\n".format(intra_path_signature))
        f.write("Absolute path signature: {}\n".format(absolute_path_signature))

    # Save the total number of crashes
    f.write("-----------------------------\n")
    f.write("Total physical crashes: {}\n".format(total_physical_accidents))
    f.write("-----------------------------\n")
    f.close()