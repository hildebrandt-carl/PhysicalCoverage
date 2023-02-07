import glob
from os.path import basename
from pathlib import Path


# Find all the files to instrument
all_files = glob.glob("./original_code/*.lua")

total_lines     = 756
total_branches  = 471
path_max_len    = 10000000

# List all methods to ignore
ignored_functions = [ 'aistatus',
                    'updatePlayerData',
                    'newManualPath',
                    'validateUserInput',
                    'warningAIDisabled',
                    'debugDraw',
                    'spanMap',
                    'onDeserialized',
                    'startRecording',
                    'stopRecording',
                    
                    'resetSpeedModeAndValue',
                    'setAggressionMode',
                    'posOnPlan',
                    'targetsCompatible',
                    'pickAiWp',
                    'projectileSqSpeedToRangeRatio',
                    'doTrafficActions',
                    'getMapEdges',
                    'fleePlan',
                    'setVehicleDebugMode',
                    'setState',
                    'setPath',
                    'driveUsingPath',
                    'dumpCurrentRoute',
                    'startFollowing',
                    'scriptStop',
                    'scriptState',
                    'setScriptDebugMode']

all_functions = ['aistatus', 'getState', 'stateChanged', 'setSpeed', 'setSpeedMode', 'resetSpeedModeAndValue', 'setAggressionInternal', 'setAggressionExternal', 'setAggressionMode', 'resetAggression', 'setTargetObjectID', 'updatePlayerData', 'driveCar', 'driveToTarget', 'posOnPlan', 'aiPosOnPlan', 'calculateTarget', 'targetsCompatible', 'getMinPlanLen', 'pickAiWp', 'pathExtend', 'projectileSqSpeedToRangeRatio', 'inCurvature', 'getPathLen', 'waypointInPath', 'doTrafficActions', 'getPlanLen', 'buildNextRoute', 'mergePathPrefix', 'planAhead', 'resetMapAndRoute', 'getMapEdges', 'newManualPath', 'validateUserInput', 'fleePlan', 'chasePlan', 'warningAIDisabled', 'offRoadFollowControl', 'updateGFX', 'debugDraw', 'setAvoidCars', 'driveInLane', 'setMode', 'reset', 'resetLearning', 'setVehicleDebugMode', 'setState', 'setTarget', 'setPath', 'driveUsingPath', 'spanMap', 'setCutOffDrivability', 'onDeserialized', 'dumpCurrentRoute', 'startRecording', 'stopRecording', 'startFollowing', 'scriptStop', 'scriptState', 'setScriptDebugMode', 'isDriving']


for file_name in all_files:
    
    # Init variables
    inside_function = False
    header_added = False
    inside_comment = False
    public_interface = False
    insert_inside_loop = False
    while_loop = False
    repeat_loop = True
    insert_inside_loop_branches = [0, 0]
    loop_line = ""
    line_counter    = 0
    branch_counter  = 0
    func_name=  None
    add_enter_exit_to_end = True

    print("Processing: \t\t{}".format(file_name))

    # Create the output file
    output_file_name = Path("./instrumented_code/{}/".format(basename(file_name)))
    print("Instrumented file: \t{}\n".format(output_file_name))
    out = open(output_file_name, "w")

    # Open the file
    with open(file_name) as f:
        # Read the line
        for line in f:

            # Create the new line
            new_line = line

            # Check if its an empty line
            if ("\n" == line):
                # Do nothing unless we haven't added the header
                if not header_added:
                    print("Header added\n")
                    header_added = True

                    # Init the variables
                    out.write("\n\n-- [[ Coverage functions ]] --\n")
                    out.write("local control_variables = {}\n")
                    out.write("control_variables['line_count']         = {}\n".format(total_lines))
                    out.write("control_variables['branches_count']     = {}\n".format(total_branches))
                    out.write("control_variables['current_max_length'] = {}\n".format(path_max_len))
                    out.write("control_variables['current_path_index'] = {}\n".format(1))
                    out.write("control_variables['start_coverage']     = false\n\n")

                    out.write("local coverage = {}\n")
                    out.write("coverage['line'] = {}\n")
                    out.write("coverage['branch'] = {}\n")
                    out.write("coverage['path'] = {}\n\n")


                    # Add the line coverage table
                    out.write("for i=0, control_variables['line_count'] do\n")
                    out.write("  coverage['line'][i] = 0\n")
                    out.write("end\n\n")

                    # Add the branch coverage table
                    out.write("for i=0, control_variables['branches_count'] do\n")
                    out.write("  coverage['branch'][i] = 0\n")
                    out.write("end\n\n")

                    # Add the path coverage table
                    out.write("for i=0, control_variables['current_max_length'] do\n")
                    out.write("  coverage['path'][i] = 0\n")
                    out.write("end\n\n")

                    # Get the line coverage
                    out.write("local function getLineCoverageArray()\n")
                    out.write("  local cov_out = ''\n")
                    out.write("  for i, line in ipairs(coverage['line']) do\n")
                    out.write("    cov_out = cov_out ..' ' .. tostring(line)\n")
                    out.write("  end\n")
                    out.write("  return cov_out\n")
                    out.write("end\n\n")

                    # Get the branch coverage
                    out.write("local function getBranchCoverageArray()\n")
                    out.write("  local branch_out = ''\n")
                    out.write("  for i, branch in ipairs(coverage['branch']) do\n")
                    out.write("    branch_out = branch_out ..' ' .. tostring(branch)\n")
                    out.write("  end\n")
                    out.write("  return branch_out\n")
                    out.write("end\n\n")

                    # Get the path coverage
                    out.write("local function getPathCoverageArray()\n")
                    out.write("  local path_out = table.concat(coverage['path'], ', ')\n")
                    out.write("  return path_out\n")
                    out.write("end\n\n")

                    # Reset the coverage
                    out.write("local function resetCoverageArrays()\n")
                    out.write("for i=0, control_variables['line_count'] do\n")
                    out.write("coverage['line'][i] = 0\n")
                    out.write("end\n")
                    out.write("for i= 0, control_variables['branches_count'] do\n")
                    out.write("coverage['branch'][i] = 0\n")
                    out.write("end\n")
                    out.write("for i=0, control_variables['current_max_length'] do\n")
                    out.write("coverage['path'][i] = 0\n")
                    out.write("end\n")
                    out.write("control_variables['current_path_index'] = 1\n")
                    out.write("-- Turn off coverage until its started again\n")
                    out.write("control_variables['start_coverage'] = false\n")
                    out.write("end\n\n")

                    # Start the coverage
                    out.write("local function startCoverage()\n")
                    out.write("  control_variables['start_coverage'] = true\n")
                    out.write("end\n\n")

                    # Used to update the coverage
                    out.write("local function update_path_coverage(line_branch_number)\n")
                    out.write("  if control_variables['start_coverage'] then\n")
                    out.write("    if control_variables['current_path_index'] < control_variables['current_max_length'] then\n")
                    out.write("      coverage['path'][control_variables['current_path_index']] = line_branch_number\n")
                    out.write("      control_variables['current_path_index'] = control_variables['current_path_index'] + 1\n")
                    out.write("    end\n")
                    out.write("  end\n")
                    out.write("end\n\n")

                    out.write("local function enter_or_exit_function(e_string, func_name)\n")
                    out.write("  if control_variables['start_coverage'] then\n")
                    out.write("    if e_string == 'enter' then\n")
                    out.write("      coverage['path'][control_variables['current_path_index']] = 'enter_' .. func_name\n")
                    out.write("      control_variables['current_path_index'] = control_variables['current_path_index'] + 1\n")
                    out.write("    end\n")
                    out.write("    if e_string == 'exit' then\n")
                    out.write("      coverage['path'][control_variables['current_path_index']] = 'exit_' .. func_name\n")
                    out.write("      control_variables['current_path_index'] = control_variables['current_path_index'] + 1\n")
                    out.write("    end\n")
                    out.write("  end\n")
                    out.write("end\n\n")

                    # Add line and path coverage function
                    out.write("local function process_line(line_number)\n")
                    out.write("  if control_variables['start_coverage'] then\n")
                    out.write("    coverage['line'][line_number] = 1;\n")
                    out.write("  end\n")
                    out.write("end\n\n")

                    # Add the predicate analyzer for if statements
                    out.write("local function predicate_analyzer(predicate, t_branch, f_branch)\n")
                    out.write("  if control_variables['start_coverage'] then\n")
                    out.write("    if (predicate) then\n")
                    out.write("      coverage['branch'][t_branch] = 1\n")
                    out.write("      update_path_coverage(t_branch)\n")
                    out.write("    else\n")
                    out.write("      coverage['branch'][f_branch] = 1\n")
                    out.write("      update_path_coverage(f_branch)\n")
                    out.write("    end\n")
                    out.write("  end\n")
                    out.write("  return predicate\n")
                    out.write("end\n\n")

                    # Pair analyzer for for loop
                    out.write("local function for_loop_pair(i, data, t_branch, f_branch)\n")
                    out.write("  if control_variables['start_coverage'] then\n")
                    out.write("    local count = 0\n")
                    out.write("    for _ in pairs(data) do count = count + 1 end\n")
                    out.write("    if (i < count) then\n")
                    out.write("      coverage['branch'][t_branch] = 1\n")
                    out.write("      update_path_coverage(t_branch)\n")
                    out.write("    else\n")
                    out.write("      coverage['branch'][f_branch] = 1\n")
                    out.write("      update_path_coverage(f_branch)\n")
                    out.write("    end\n")
                    out.write("  end\n")
                    out.write("end\n\n")

                    # Range analyzer for for loop
                    out.write("local function for_loop_range(data1, data2, t_branch, f_branch)\n")
                    out.write("  if control_variables['start_coverage'] then\n")
                    out.write("    if (data1 < data2) then\n")
                    out.write("      coverage['branch'][t_branch] = 1\n")
                    out.write("      update_path_coverage(t_branch)\n")
                    out.write("    else\n")
                    out.write("      coverage['branch'][f_branch] = 1\n")
                    out.write("      update_path_coverage(f_branch)\n")
                    out.write("    end\n")
                    out.write("  end\n")
                    out.write("end\n")

                    out.write("---------------------------------\n\n")

            # Check multi line comments
            elif "--[[" == line.strip()[0:4]:
                inside_comment = True

            # Close multiline comments
            elif inside_comment:
                if "--]]" in line:
                    inside_comment = False

            # Check if we are at the end of the file where we declare public interfaces
            elif "-- public interface" == line.strip():
                public_interface = True
                out.write("-- additional public interface\n")
                out.write("M.getLineCoverageArray = getLineCoverageArray\n")
                out.write("M.getBranchCoverageArray = getBranchCoverageArray\n")
                out.write("M.resetCoverageArrays = resetCoverageArrays\n")
                out.write("M.startCoverage = startCoverage\n")
                out.write("M.getPathCoverageArray = getPathCoverageArray\n")
                pass

            # Check single line comments
            elif "--" == line.strip()[0:2]:
                pass

            # Determine if this is a function
            elif "local function" in line:

                # Get the function name
                func_name = line[15:line.rfind("(")]
                
                # Make sure we want to instrument this code
                if func_name not in ignored_functions:
                    # Add the the start of functions
                    whitespace = len(line) - len(line.lstrip())
                    new_line = line + line[:whitespace] + "  enter_or_exit_function('enter', '{}')\n".format(func_name)

                    print("Instrumenting: {}".format(func_name))
                    # We are inside a function
                    inside_function = True

            # Ignore else
            elif ("else\n" in line) or (("end\n" in line) and (len(line) > 4)):
                pass

            # Check when we are finished processing a function
            elif ("end\n" == line) and inside_function:
                # Add to the end of the function
                if add_enter_exit_to_end:
                    pass
                    # TODO
                    whitespace = len(line) - len(line.lstrip())
                    new_line = line[:whitespace] + "  enter_or_exit_function('exit', '{}')\n".format(func_name) + line
                
                print("Finished: {}\n".format(func_name))
                inside_function = False

            # Otherwise check if we are inside a function and need to edit the lines
            elif inside_function:

                # Check if the current line is the last return
                if line[0:8] == "  return":
                    add_enter_exit_to_end = False
                else:
                    add_enter_exit_to_end = True       

                # Compute the whitespace of the line
                whitespace = len(line) - len(line.lstrip())

                # Check if we are in a return statement
                return_statement = False


                if (line.strip()[0:6] == "return"):
                    pass
                    # In this case this is the last return and we shouldn't add an exit or enter function to the end
                    return_statement = True
                    return_string = line[:whitespace] + "enter_or_exit_function('exit', '{}')\n".format(func_name)  

                # Check if we are in a branch
                branch = False
                if ("if " == line.strip()[0:3]) or ("elseif " in line.strip()[0:7]):
                    branch = True

                # If we are in a branch
                if branch:
                    # Get the predicate
                    predicate = line[line.find("if")+3:line.rfind("then")-1]
                    # Update the function to use the predicate analyzer
                    line = line[:line.find("if")+3] + "(predicate_analyzer({}, {}, {}))".format(predicate, branch_counter, branch_counter+1) + line[line.rfind("then")-1:]
                    branch_counter += 2
                
                # Check if we are in a while loop
                while_loop = False
                if ("while " in line.strip()[:7]):
                    while_loop = True

                # Handle the while loop
                if while_loop:
                    # get the loop predicate
                    while_loop_predicate = line[line.find("while")+6:line.rfind("do")-1]

                    # Add the loop predicate
                    line = line[:line.find("while")+5] + "(predicate_analyzer({}, {}, {}))".format(while_loop_predicate, branch_counter, branch_counter+1) + line[line.rfind("do")-1:]
                    branch_counter += 2
                    while_loop = False

                repeat_loop = False
                if ("until" in line.strip()[:6]):
                    repeat_loop = True

                # Handle the repeat loop
                if repeat_loop:
                    # get the loop predicate
                    repeat_loop_predicate = line[line.find("until")+6:-1]

                    # Add the loop predicate
                    line = line[:line.find("until")+5] + "(predicate_analyzer({}, {}, {}))".format(repeat_loop_predicate, branch_counter, branch_counter+1)
                    branch_counter += 2
                    repeat_loop = False                

                # Check if we are in a for loop
                for_loop = False
                if ("for " in line.strip()[:5]):
                    for_loop = True

                # Insert inside loop
                if insert_inside_loop:
                    insert_inside_loop = False
                    loop_predicate = loop_predicate_all
                    # There are two main ways they loop here. Either with pairs or with range
                    if ("in pairs(" in loop_line) or ("in ipairs(" in loop_line):
                        # We cant handle two cases
                        if not (("pairs(mapData.graph[path[nextPathIdx]])" in loop_line) or ("pairs(mapData.graph[nid1]" in loop_line)):
                            # Refine the predicate
                            loop_variable = loop_predicate[:loop_predicate.find(",")]
                            loop_predicate = loop_predicate[loop_predicate.find("(")+1:loop_predicate.rfind(")")]
                            # Create the new line
                            additional_line = line[:whitespace] + "for_loop_pair({}, {}, {}, {})\n".format(loop_variable, loop_predicate, insert_inside_loop_branches[0], insert_inside_loop_branches[1])
                            out.write(additional_line)
                    elif ("=" in loop_line) and ("," in loop_line):
                        # Refine the predicate
                        loop_variable = loop_predicate[:loop_predicate.find(" =")]
                        loop_predicate = loop_predicate[loop_predicate.find("=")+1:]
                        loop_predicate = loop_predicate.strip()
                        loop_predicate = loop_predicate.split(", ")
                        assert(len(loop_predicate) >= 2)
                        if len(loop_predicate) == 2:
                            additional_line = line[:whitespace] + "for_loop_range({}, {}, {}, {})\n".format(loop_variable, loop_predicate[1], insert_inside_loop_branches[0], insert_inside_loop_branches[1])
                        elif len(loop_predicate) == 3:
                            assert(loop_predicate[2] == "-1")
                            additional_line = line[:whitespace] + "for_loop_range({}, {}, {}, {})\n".format(loop_predicate[1], loop_variable, insert_inside_loop_branches[0], insert_inside_loop_branches[1])
                        # Create the new line
                        out.write(additional_line)
                    else:
                        print("Unknown type of for loop")

                # If we are in a loop
                if for_loop:
                    # Get the loop predicate
                    loop_predicate_all = line[line.find("for")+4:line.rfind("do")-1]
                    loop_predicate = line[line.find("for")+4:line.rfind("do")-1]
                    insert_inside_loop = True
                    loop_line = line

                    # This goes above the for loop
                    # There are two main ways they loop here. Either with pairs or with range
                    if ("in pairs(" in loop_line) or ("in ipairs(" in loop_line):
                        # Refine the predicate
                        loop_variable = loop_predicate[:loop_predicate.find(",")]
                        loop_predicate = loop_predicate[loop_predicate.find("(")+1:loop_predicate.rfind(")")]
                        # Create the new line
                        additional_line = line[:whitespace] + "for_loop_pair({}, {}, {}, {})\n".format(0, loop_predicate, branch_counter, branch_counter+1)
                        insert_inside_loop_branches[0] = branch_counter
                        insert_inside_loop_branches[1] = branch_counter+1
                        branch_counter += 2
                        out.write(additional_line)
                    elif ("=" in loop_line) and ("," in loop_line):
                        # Refine the predicate
                        loop_variable = loop_predicate[:loop_predicate.find(" =")]
                        loop_predicate = loop_predicate[loop_predicate.find("=")+1:]
                        loop_predicate = loop_predicate.strip()
                        loop_predicate = loop_predicate.split(", ")
                        assert(len(loop_predicate) >= 2)
                        if len(loop_predicate) == 2:
                            additional_line = line[:whitespace] + "for_loop_range({}, {}, {}, {})\n".format(loop_predicate[0], loop_predicate[1], branch_counter, branch_counter+1)
                        elif len(loop_predicate) == 3:
                            assert(loop_predicate[2] == "-1")
                            additional_line = line[:whitespace] + "for_loop_range({}, {}, {}, {})\n".format(loop_predicate[1], loop_predicate[0], branch_counter, branch_counter+1)
                        
                        insert_inside_loop_branches[0] = branch_counter
                        insert_inside_loop_branches[1] = branch_counter+1
                        # Create the new line
                        branch_counter += 2
                        out.write(additional_line)
                    else:
                        print("Unknown type of for loop")

                # Add line coverage
                if (for_loop) or (branch) or (while_loop) or line.strip() == "else":
                    new_line = line
                else:
                    hook = "process_line({}); ".format(line_counter)
                    if return_statement:
                        new_line = return_string + line[:whitespace] + hook + line[whitespace:]
                    else:
                        new_line = line[:whitespace] + hook + line[whitespace:]
                    line_counter += 1


            # Write to the file
            out.write(new_line)

    # Close the output file
    out.close()

print("Total lines instrumented: {}".format(line_counter))
print("Total branches instrumented: {}".format(branch_counter))