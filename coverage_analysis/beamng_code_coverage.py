import glob
import re
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("./output/random_tests/code_coverage/*/*.txt")
print("Processing: {} files".format(len(files)))

all_lines_coverage = set([680, 681, 682, 683, 684, 685, 687, 688, 689, 690, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668,592,593,594,595,596,597,598,599,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650])
# all_lines_coverage = set()
all_possible_lines = set()

coverage_array = []

for f in files:

    f = open(f, "r")

    lines_coverage = []
    total_lines_coverage = 0

    all_lines = []
    total_all_lines = 0

    # Read the file
    for line in f: 
        if "Lines covered:" in line:
            result = re.sub('[^0-9^\s]','', line)
            result = result.split(' ')
            for r in result:
                try:
                    lines_coverage.append(int(r))
                except:
                    pass

        if "Total lines covered:" in line:
            total_lines_coverage = int(line[21:])
            assert(len(lines_coverage) == total_lines_coverage)

        if "All Lines:" in line:
            result = re.sub('[^0-9^\s]','', line)
            result = result.split(' ')
            for r in result:
                try:
                    all_lines.append(int(r))
                except:
                    pass

        if "Total Lines:" in line:
            total_all_lines = int(line[13:])
            assert(len(all_lines) == total_all_lines)

    # Close the file
    f.close()
     
    # Check the all possible lines
    if len(all_possible_lines) <= 0:
        all_possible_lines = set(all_lines)
    else:
        for line_number in all_lines:
            if line_number not in all_possible_lines:
                print("error")
                break

    # Keep track of the coverage:
    for line_number in lines_coverage:
        all_lines_coverage.add(line_number)

    coverage_percentage = (len(all_lines_coverage) / len(all_possible_lines)) * 100.0
    coverage_array.append(coverage_percentage)


print("The lines not covered are:")
lines_not_coverage = sorted(list(all_possible_lines - all_lines_coverage))
print(lines_not_coverage)

plt.plot(coverage_array)
plt.ylim([-5,105])
plt.grid()
plt.yticks(np.arange(0, 100.01, step=5))
plt.xlabel("Number of tests")
plt.ylabel("Code Coverage (%)")
plt.title("Total Code Coverage: {}%".format(np.round(coverage_array[-1], 2)))
plt.show()