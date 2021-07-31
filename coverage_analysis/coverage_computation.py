import argparse
import numpy as np
import glob as glob
import matplotlib.pyplot as plt

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i + 1, y[i], str(y[i]) + "%", ha = 'center')

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_tests',    type=int, default=1,      help="The number of tests used while computing coverage")
parser.add_argument('--coverage_type',      type=str, default="line", help="Line or branch coverage")
parser.add_argument('--scenario',           type=str, default="",     help="beamng/highway")
args = parser.parse_args()

# Normal
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/coverage/' + str(args.number_of_tests) + "_tests/"

if args.coverage_type == "line":
    files = glob.glob(base_path + "coverage_results/*/*-coverage.txt")
elif args.coverage_type == "branch":
    files = glob.glob(base_path + "coverage_results_branch/*/*-coverage-branch.txt")
else:
    exit()
    
# Get the vehicle numbers
vehicle_count = []
for f in files:
    f = f.split('/')[-1]
    count = f[:f.find('-')]
    vehicle_count.append(int(count))

# Sort both lists according to the vehicle count
vehicle_count, files = zip(*sorted(zip(vehicle_count, files)))

coverage_percentage = []

# Go through each and get the coverage
for f in files:
    file1 = open(f, 'r')
    lines = file1.readlines()
    file1.close()

    # Get the line we are interested in:
    for l in lines:
        # Get the coverage
        if "car_controller.py" in l:
            l = l.split()
            coverage_percentage.append(int(l[-1][:-1]))


fig = plt.figure()
plt.bar(vehicle_count, coverage_percentage)
addlabels(vehicle_count, coverage_percentage)
plt.xticks(np.arange(0,len(vehicle_count)) + 1, vehicle_count, rotation='horizontal')
plt.grid(b=True, which='major', color='#445577', linestyle=':')
plt.ylabel(args.coverage_type.capitalize() + " coverage (%)")
plt.xlabel("Number of traffic vehicles")
plt.title("Running " + str(args.number_of_tests) + " tests")
plt.ylim([0, 105])
plt.show()