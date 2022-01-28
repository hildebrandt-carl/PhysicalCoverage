

import os
import ast
import glob
import copy
import hashlib
import argparse
import multiprocessing

import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm


def intraprocedural_path_coverage(path_taken):
    # These functions are declared when processing the AI file in beamng
    start_end_branch_numbers = {}
    start_end_branch_numbers["stateChanged"]        = [0, 1]
    start_end_branch_numbers["setSpeedMode"]        = [2, 3]
    start_end_branch_numbers["driveToTarget"]       = [4, 41]
    start_end_branch_numbers["aiPosOnPlan"]         = [42, 53]
    start_end_branch_numbers["calculateTarget"]     = [54, 65]
    start_end_branch_numbers["pathExtend"]          = [66, 67]
    start_end_branch_numbers["inCurvature"]         = [68, 73]
    start_end_branch_numbers["getPathLen"]          = [74, 75]
    start_end_branch_numbers["waypointInPath"]      = [76, 79]
    start_end_branch_numbers["getPlanLen"]          = [80, 81]
    start_end_branch_numbers["buildNextRoute"]      = [82, 113]
    start_end_branch_numbers["mergePathPrefix"]     = [114, 123]
    start_end_branch_numbers["planAhead"]           = [124, 205]
    start_end_branch_numbers["chasePlan"]           = [206, 239]
    start_end_branch_numbers["updateGFX"]           = [240, 387]
    start_end_branch_numbers["setAvoidCars"]        = [388, 389]
    start_end_branch_numbers["driveInLane"]         = [390, 391]
    start_end_branch_numbers["setMode"]             = [392, 405]
    start_end_branch_numbers["reset"]               = [406, 407]

    print(path_taken[:200])

    # Create the dictionary to hold all paths
    function_starts = []
    function_ends   = []
    function_names  = []
    i_paths_taken = {}
    for key in start_end_branch_numbers:
        # Create the set of intraprocedural paths taken
        i_paths_taken[key] = set()
        # Save the start and end branches for easy lookup
        function_starts.append(get_true_branch_number(start_end_branch_numbers[key][0]))
        function_ends.append(get_true_branch_number(start_end_branch_numbers[key][1]))
        function_names.append(key)

    # Make sure nothing broke
    assert(len(function_starts) == len(start_end_branch_numbers))
    assert(len(function_ends) == len(start_end_branch_numbers))
    assert(len(function_names) == len(start_end_branch_numbers))

    # Process the path
    function_stack          = []
    current_function        = None
    start_recording_path    = False
    intraprocedural_path    = []
    counter = -100
    for b in path_taken:

        # Get the true branch for comparing against start and end
        branch = get_true_branch_number(b)
        branch_number = b

        print_tabs = ""
        for i in range(len(function_stack)-1):
            print_tabs += "  "

        # Check if this is the start or end of a function
        if branch in function_starts:
            # Get the function name
            index = function_starts.index(branch)
            f_name = function_names[index]

            print("{}{} start".format(print_tabs, f_name))
            
            # Append this function to the stack and the current path to a stack
            function_stack.append(([], f_name))

        print_tabs = ""
        for i in range(len(function_stack)-1):
            print_tabs += "  "

        # Push the current branch to the most recent function in the stack
        function_stack[-1][0].append(branch_number)
        print("{}Stack size: {} - Processed: {}".format(print_tabs, len(function_stack), branch_number))

        if branch in function_ends:
            # Pop the stack
            print_tabs = print_tabs[:-1]
            i_path, f_name = function_stack.pop()
            print("{}{} end - {}".format(print_tabs, f_name, i_path))

        counter += 1
        if counter > 0:
            exit()
        

def get_true_branch_number(branch_number):
    return branch_number if (branch_number % 2) == 0 else  branch_number - 1

def clean_path(path_taken, first_branch, last_branch, file_name):
    # Remove all 0's from the end of the array
    first_zero = np.where(path_taken==0)[0][0]
    path_taken = path_taken[:first_zero]

    # Branches are numbered with true as even, and false as odd
    # The same branches true and false path will only be 1 number away
    # Thus to make sure we start on the same branch, we need to check it is either true or false
    # e.g. if first_branch == 240... path_taken[0] = 240, or path_taken[1] = 241 is okay
    current_first_branch = get_true_branch_number(path_taken[0])
    current_last_branch = get_true_branch_number(path_taken[-1])

    # Assert they start with the same branch as all others
    assert first_branch == current_first_branch, "first: {} != {}".format(first_branch, current_first_branch)

    # If they do not end of the same branch, search back through the file until they do
    if last_branch != current_last_branch:
        path_taken = np.array(path_taken)
        # Find the last occurrence of the branch (either true of false)
        p1 = np.argwhere(path_taken == last_branch)
        p2 = np.argwhere(path_taken == (last_branch + 1))
        if np.size(p1) != 0 and np.size(p2) != 0:
            possible_indices = [p1.max(), p2.max()]
            print("Successfully found new ending")
        elif np.size(p1) != 0:
            possible_indices = [p1.max(), 0]
            print("Successfully found new ending")
        elif np.size(p1) != 0:
            possible_indices = [0, p2.max()]
            print("Successfully found new ending")
        else:
            print("Error in ".format(file_name))
            print("Error ({}): branch {} or {} can not be found anywhere in the path".format(file_name, last_branch, last_branch+1))
            possible_indices = [np.size(path_taken)]
        last_occurrence = max(possible_indices)
        path_taken = list(path_taken[0:last_occurrence])

    # # Assert they end with the same branch as all others
    # current_last_branch = path_taken[-1] if (path_taken[-1] % 2) == 0 else  path_taken[-1] - 1
    # assert last_branch == current_last_branch, "last: {} != {}".format(last_branch, current_last_branch)
    return path_taken

def most_common(lst):
    return max(set(lst), key=lst.count)

def identity_first_last_candidates(file_name):
    # Get the path_taken
    path_taken = None
    
    with open(file_name, "r") as f:
        for line in f:

            if "Path Taken:" in line[0:12]:
                path_taken = line[12:-1]
                break

    # Check if we couldnt find a path
    if path_taken is None:
        return None, None

    # Get the path taken
    path_taken = np.array(path_taken.split(", "), dtype=int)        
    
    # Remove all 0's from the end of the array
    first_zero = np.where(path_taken==0)[0][0]
    path_taken = path_taken[:first_zero]

    # Branches are numbered with true as even, and false as odd
    # The same branches true and false path will only be 1 number away
    # Thus to make sure we start on the same branch, we need to check it is either true or false
    # e.g. if first_branch == 240... path_taken[0] = 240, or path_taken[1] = 241 is okay
    # Make sure we return even numbers
    first_branch_candidate = get_true_branch_number(path_taken[0])
    last_branch_candidate = get_true_branch_number(path_taken[-1])

    return first_branch_candidate, last_branch_candidate

def get_first_last_branch(file_name, total_cores):
    total_processors = min(len(file_name), total_cores)
    pool =  multiprocessing.Pool(processes=total_processors)

    # Get the candidates for each file
    jobs = []
    for i in range(len(file_name)):
        jobs.append(pool.apply_async(identity_first_last_candidates, args=([file_names[i]])))

    # Get the results
    possible_starting_branches = []
    possible_ending_branches = []
    for job in tqdm(jobs):
        candidates = job.get()
        possible_starting_branches.append(candidates[0])
        possible_ending_branches.append(candidates[1])

    # Get the first and last branch
    first_branch = most_common(possible_starting_branches)
    last_branch = most_common(possible_ending_branches)

    # Close the pool
    pool.close()

    # Print out details
    print("Start branch candidates: {}".format(possible_starting_branches))
    print("Start branch selected candidate: {}".format(first_branch))
    print("Last branch candidates: {}".format(possible_ending_branches))
    print("Last branch selected candidate: {}".format(last_branch))

    # Return the first and last branch
    return first_branch, last_branch

def compute_coverage_beamng(file_name, save_path, first_branch, last_branch):

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
    path_taken = np.array(path_taken.split(", "), dtype=int)  
    path_taken = clean_path(path_taken, first_branch, last_branch, file_name)

    # Compute intraprocedural path coverage
    i_path_taken = intraprocedural_path_coverage(path_taken)

    # Get the intraprocedural path_signature
    i_path_string = ''.join([str(x) + "," for x in i_path_taken])
    i_path_signature = hashlib.md5(absolute_path_string.encode()).hexdigest()

    # Get the absolute path_signature
    absolute_path_string = ''.join([str(x) + "," for x in path_taken])
    absolute_path_signature = hashlib.md5(absolute_path_string.encode()).hexdigest()

    # Convert the branch data to strings to match the highwayenv data
    branches_covered = [str(x) for x in branches_covered]
    all_branches = [str(x) for x in all_branches]

    coverage_data["ai.lua"] = [lines_covered, all_lines, branches_covered, all_branches, i_path_signature, absolute_path_signature]
    return [coverage_data, code_coverage_save_name, crash_count]

def compute_coverage_highway(file_name, save_path, first_branch, last_branch):
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
        coverage_data[current_class] = [lines_covered, all_lines, branches_covered, all_branches, None, None]
    
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
    base = "../../PhysicalCoverageData/beamng/random_tests"
    all_files = glob.glob(base + "/code_coverage/raw/*/*.txt")
elif args.scenario == "beamng_generated":
    print("To be implemented")
    exit()
elif args.scenario == "highway_random":
    base = "../../PhysicalCoverageData/highway/random_tests"
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

print("----------------------------------")
print("-----Creating Output Location-----")
print("----------------------------------")

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

# Create the output directory if it doesn't exists
for i in range(10):
    new_path = save_path + "{}_external_vehicles/".format(i+1)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

print("Location created: {}".format(save_path))

print("----------------------------------")
print("---------Processing files---------")
print("----------------------------------")

# Get the first and last branch number
print("Identifying start and end branch candidates for path coverage")
first_branch, last_branch = get_first_last_branch(file_names[0:1], args.cores)


print("\n\nComputing line, branch and path coverage")
total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
for i in range(total_files):
    if "highway" in args.scenario:
        jobs.append(pool.apply_async(compute_coverage_highway, args=([file_names[i], save_path, first_branch, last_branch])))
    elif "beamng" in args.scenario:
        jobs.append(pool.apply_async(compute_coverage_beamng, args=([file_names[i], save_path, first_branch, last_branch])))

# Get the results
results = []
for job in tqdm(jobs):
    results.append(job.get())

# Close the pool
pool.close()

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
        path_signature      = coverage_data[key][4]

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
        f.write("Path signature: {}\n".format(path_signature))

    # Save the total number of crashes
    f.write("-----------------------------\n")
    f.write("Total physical crashes: {}\n".format(total_physical_accidents))
    f.write("-----------------------------\n")
    f.close()