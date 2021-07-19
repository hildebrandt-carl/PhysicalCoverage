import glob as glob

# Branch
# files = glob.glob("coverage_results_branch/*/*-coverage-branch.txt")

# # Normal
files = glob.glob("coverage_results/*/*-coverage.txt")

# Get the vehicle numbers
vehicle_count = []
for f in files:
    count = f[f.find('/')+1:f.find('-')]
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

print(coverage_percentage)

