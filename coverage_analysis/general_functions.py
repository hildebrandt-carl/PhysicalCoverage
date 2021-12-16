import ast

def get_beam_numbers(files):

    beam_numbers = []
    for f in files:
        f_name = f[f.rfind("/"):]
        if "random" in f_name:
            f_name = f_name[f_name.find("_random")+7:]
        elif "generated" in f_name:
            f_name = f_name[f_name.find("_generated")+10:]
        beam = f_name[f_name.find("_b")+2:-4]
        if beam.find("_") != -1:
            beam = beam[:beam.find("_")]
        beam_numbers.append(int(beam))
    return beam_numbers

def order_by_beam(files, beam_number):
    resultant_files = []
    # Go through each beam
    for b in beam_number:
        found = False
        for f in files:
            if "_b{}_".format(b) in f or "_b{}.".format(b) in f:
                found = True
                resultant_files.append(f)
        if not found:
            resultant_files.append("")
    
    return resultant_files

def get_filename_prefix(filename):
    if filename == "car_controller.py":
        return 1000
    elif filename == "controller.py":
        return 2000
    else:
        print("Error! File not known.")
        exit()

def get_lines_covered(filename):
    # Open the file
    f = open(filename, "r")

    line_data = {}
    branch_data = {}
    current_key = None
    total_crashes = None

    # Read the file
    for line in f: 
        # Get the file we are working on
        if ".py" in line:
            current_key = line[6:-1]
            line_data[current_key] = {}
            branch_data[current_key] = {}

        # Line coverage
        if "Lines covered:" in line:
            covered_l = ast.literal_eval(line[15:])
            line_data[current_key]["lines_covered"] = covered_l

        if "Total lines covered:" in line:
            total_covered_l = int(line[21:])
            assert(len(covered_l) == total_covered_l)

        if "All lines:" in line:
            all_l = ast.literal_eval(line[11:])
            line_data[current_key]["all_lines"] = all_l

        if "Total lines:" in line:
            total_all_l = int(line[13:])
            assert(len(all_l) == total_all_l)

        # Branch coveraged
        if "Branches covered:" in line:
            covered_b = ast.literal_eval(line[18:])
            branch_data[current_key]["branches_covered"] = covered_b

        if "Total branches covered:" in line:
            total_covered_b = int(line[24:])
            assert(len(covered_b) == total_covered_b)

        if "All branches:" in line:
            all_b = ast.literal_eval(line[14:])
            branch_data[current_key]["all_branches"] = all_b

        if "Total branches:" in line:
            total_all_b = int(line[15:])
            assert(len(all_b) == total_all_b)

        if "Total physical crashes:" in line:
            total_crashes = int(line[23:])

    # Close the file
    f.close()

    # Group the line and branch information into a single set of lines
    all_lines_covered = []
    all_lines = []
    for key in line_data:
        file_adder = get_filename_prefix(key)
        for line in line_data[key]["lines_covered"]:
            all_lines_covered.append(line + file_adder)
        for line in line_data[key]["all_lines"]:
            all_lines.append(line + file_adder)
    all_branches_covered = []
    all_branches = []
    for key in branch_data:
        file_adder = get_filename_prefix(key)
        for branch in branch_data[key]["branches_covered"]:
            if "_" not in branch:
                branch = str(int(branch) + file_adder)
            else:
                prefix = int(branch[:branch.find("_")]) + file_adder
                suffix = int(branch[branch.rfind("_")+1:]) + file_adder
                branch = str(prefix) + "_" + str(suffix)
            all_branches_covered.append(branch)
        for branch in branch_data[key]["all_branches"]:
            branch = str(int(branch) + file_adder)
            all_branches.append(branch)

    # Compile the results
    results = [all_lines_covered, all_lines, all_branches_covered, all_branches, total_crashes]
    return results

def clean_branch_data(all_branches_set, branches_covered_set):
    # Fix the allbranches set
    all_branches_set_new = set()
    for branch in all_branches_set:
        all_branches_set_new.add(branch + "a")
        all_branches_set_new.add(branch + "b")

    # Compute all the branch numbers seen
    branch_numbers_seen = set()
    for branch in branches_covered_set:
        prefix = branch
        if "_" in branch:
            prefix = branch[:branch.find("_")]
        branch_numbers_seen.add(prefix)

    # Go through all branches and find if two of the same prefix exist
    branches_covered_set_new = set()
    for branch_number in branch_numbers_seen:
        occurrences = []
        for branch in branches_covered_set:
            prefix = branch
            if "_" in branch:
                prefix = branch[:branch.find("_")]
            # If the prefix matches the branch number
            if prefix == branch_number:
                occurrences.append(branch)

        if len(occurrences) > 3:
            print("Error, why have you seen more than 3 branches (true, false, truefalse)")
            exit()

        # Check if any of the occurrences were both:
        saw_both = False
        for o in occurrences:
            if "_" not in o:
                saw_both = True

        if (saw_both) or (len(occurrences) == 2):
            branches_covered_set_new.add(branch_number + "a")
            branches_covered_set_new.add(branch_number + "b")
        elif len(occurrences) == 1:
            branches_covered_set_new.add(branch_number + "a")
        else:
            print("This case should not exist")
            exit()   

    return all_branches_set_new, branches_covered_set_new

def ignored_lines_definition(scenario):
    if scenario == "highway":
        ignored_lines = {}
        ignored_lines["car_controller.py"] = [1,2,3,4,5,6,77,8,9,10,11,12,13,14,28,143,144,145,146,147,148,149,150,151,152,157]
        ignored_lines["controller.py"] = [1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23,24,25,26,27,28,29,31,45,46,55,58,59,60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 200, 204, 205, 206, 208, 222, 242, 254, 264, 265 , 275, 276, 279, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411]
    elif scenario == "beamng":
        print("Error 4000001")
        exit()
    return ignored_lines

def get_ignored_lines(scenario):
    ignored_lines = ignored_lines_definition(scenario)
    # Convert into the format we are using
    result = []
    for key in ignored_lines:
        prefix = get_filename_prefix(key)
        for l in ignored_lines[key]:
            result.append(l + prefix)
    return result

def get_ignored_branches(scenario):
    ignored_lines = ignored_lines_definition(scenario)
    # Convert into the format we are using
    result = []
    for key in ignored_lines:
        prefix = get_filename_prefix(key)
        for l in ignored_lines[key]:
            result.append(str(l + prefix) + "a")
            result.append(str(l + prefix) + "b")
    return result
