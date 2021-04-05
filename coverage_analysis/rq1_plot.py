import argparse
import numpy as np
import matplotlib.pyplot as plt


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom')

def myround(x, base=5):
    return round(base * round(x/base), 5)

def plot_number_of_test_suites(number_tests_data, plot_number):
    ax = fig.add_subplot(2, 3, plot_number)
    x = []
    y = []
    for key in number_tests_data:
        x.append(key)
        y.append(len(num_test_data[key]))
    rects = ax.bar(x, y)
    plt.xlabel("Physical Coverage (%)")
    plt.ylabel("Number of Test Suites")
    autolabel(rects, ax)

def plot_coverage_vs_crashes(crash_data, plot_number):
    # Creat the box plot
    ax = fig.add_subplot(2, 3, plot_number)
    plot_data = []
    label_data = []
    for key in crash_data:
        plot_data.append(crash_data[key])
        label_data.append(key)
    # Creating plot 
    bp = plt.boxplot(plot_data) 
    plt.xticks(np.arange(1, len(label_data) + 1), label_data)
    plt.xlabel("Physical Coverage (%)")
    plt.ylabel("Crashes")

def plot_coverage_vs_number_tests(num_test_data, plot_number):
    # Creat the box plot
    ax = fig.add_subplot(2, 3, plot_number)
    plot_data = []
    label_data = []
    for key in num_test_data:
        plot_data.append(num_test_data[key])
        label_data.append(key)
    # Creating plot 
    bp = plt.boxplot(plot_data) 
    plt.xticks(np.arange(1, len(label_data) + 1), label_data)
    plt.xlabel("Physical Coverage (%)")
    plt.ylabel("Number of Tests in Test Suite")

def plot_number_tests_vs_crashes(num_test_data, crash_data, plot_number):
    # Creat the scatter plot
    ax = fig.add_subplot(2, 3, plot_number)
    x = []
    y = []
    # Get all the data
    for key in num_test_data:
        tmp = num_test_data[key]
        for i in tmp:
            x.append(i)
        tmp = crash_data[key]
        for i in tmp:
            y.append(i)
    # Creating plot 
    plt.scatter(x, y, s=1) 
    plt.xlabel("Number of Tests in Test Suite")
    plt.ylabel("Number of Crashes")

def plot_coverage_vs_crashes_scatter(coverage_data, crash_data, plot_number):
    # Creat the scatter plot
    ax = fig.add_subplot(2, 3, plot_number)
    x = []
    y = []
    # Get all the data
    for key in coverage_data:
        tmp = coverage_data[key]
        for i in tmp:
            x.append(i)
        tmp = crash_data[key]
        for i in tmp:
            y.append(i)
    # Creating plot 
    plt.scatter(x, y, s=1) 
    plt.xlabel("Coverage (%)")
    plt.ylabel("Number of Crashes")

def plot_crashes_sorted_by_crashes(crash_data, plot_number):
    # Creat the scatter plot
    ax = fig.add_subplot(2, 3, plot_number)
    x = []
    y = []
    sort_array = []
    # Get all the data
    counter = 0
    for key in crash_data:
        for i in range(len(crash_data[key])):
            counter += 1
            x.append(counter)
            y.append(crash_data[key][i])
            sort_array.append(crash_data[key][i])
    # Sort based on coverage
    sort_array = np.array(sort_array)
    y = np.array(y)
    inds = sort_array.argsort()
    y_sorted = y[inds]
    # Creating plot 
    plt.scatter(x, y_sorted, s=1, label="Sorted by total crashes") 
    plt.xlabel("Test Suite Index")
    plt.ylabel("Number of Crashes")
    plt.legend()

def plot_crashes_sorted_by_test_size(crash_data, num_test_data, plot_number):
    # Creat the scatter plot
    ax = fig.add_subplot(2, 3, plot_number)
    x = []
    y = []
    sort_array = []
    # Get all the data
    counter = 0
    for key in crash_data:
        for i in range(len(crash_data[key])):
            counter += 1
            x.append(counter)
            y.append(crash_data[key][i])
            sort_array.append(num_test_data[key][i])
    # Sort based on coverage
    sort_array = np.array(sort_array)
    y = np.array(y)
    inds = sort_array.argsort()
    y_sorted = y[inds]
    # Creating plot 
    plt.scatter(x, y_sorted, s=1, label="Sorted by number of tests in test suite") 
    plt.xlabel("Test Suite Index")
    plt.ylabel("Number of Crashes")
    plt.legend()

def plot_crashes_sorted_by_coverage(crash_data, coverage_data, plot_number):
    # Creat the scatter plot
    ax = fig.add_subplot(2, 3, plot_number)
    x = []
    y = []
    sort_array = []
    # Get all the data
    counter = 0
    for key in crash_data:
        for i in range(len(crash_data[key])):
            counter += 1
            x.append(counter)
            y.append(crash_data[key][i])
            sort_array.append(coverage_data[key][i])
    # Sort based on coverage
    sort_array = np.array(sort_array)
    y = np.array(y)
    inds = sort_array.argsort()
    y_sorted = y[inds]
    # Creating plot 
    plt.scatter(x, y_sorted, s=1, label="Sorted by coverage") 
    plt.xlabel("Test Suite Index")
    plt.ylabel("Number of Crashes")
    plt.legend()


parser = argparse.ArgumentParser()
parser.add_argument('--steering_angle',             type=int, default=30,    help="The steering angle used to compute the reachable set")
parser.add_argument('--beam_count',                 type=int, default=5,     help="The number of beams used to vectorize the reachable set")
parser.add_argument('--max_distance',               type=int, default=30,    help="The maximum dist the vehicle can travel in 1 time step")
parser.add_argument('--accuracy',                   type=int, default=5,     help="What each vector is rounded to")
parser.add_argument('--total_samples',              type=int, default=1000,  help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',                   type=str, default="",    help="beamng/highway")
parser.add_argument('--total_random_test_suites',   type=int, default=1000,  help="Total random test suites to be generated")
parser.add_argument('--cores',                      type=int, default=4,     help="number of available cores")
args = parser.parse_args()

min_tests_per_group = 1000
interval_size = 1

results = np.load("../results/rq1_" + args.scenario + ".npy")

# Get the coverage / crashes / and number of tests
coverage = np.zeros(len(results))
crashes = np.zeros(len(results))
num_tests = np.zeros(len(results))
for i in range(len(results)):
    coverage[i] = results[i][0]
    crashes[i] = results[i][1]
    num_tests[i] = results[i][2]

# Compute the interval size
max_coverage = np.max(coverage) + interval_size
box_intervals = np.arange(0, max_coverage, interval_size)

# Create the dictionaries which will hold the data broken into intervals
coverage_data = {}
crash_data = {}
num_test_data = {}
for interval in box_intervals:
    coverage_data[str(myround(interval, interval_size))] = []
    crash_data[str(myround(interval, interval_size))] = []
    num_test_data[str(myround(interval, interval_size))] = []

# Sort the data into each of the dictionaries based on coverage
for i in range(coverage.shape[0]):
    c = myround(coverage[i], interval_size)
    coverage_data[str(c)].append(coverage[i])
    crash_data[str(c)].append(crashes[i])
    num_test_data[str(c)].append(num_tests[i])


# Plotting before cliping
fig = plt.figure("Before Clipping", figsize =(18, 5))
plot_number_of_test_suites(num_test_data, 1)
plot_coverage_vs_crashes(crash_data, 2)
plot_coverage_vs_number_tests(num_test_data, 3)
plot_number_tests_vs_crashes(num_test_data, crash_data, 4)
plot_coverage_vs_crashes_scatter(coverage_data, crash_data, 5)
plot_crashes_sorted_by_test_size(crash_data, num_test_data, 6)
plot_crashes_sorted_by_coverage(crash_data, coverage_data, 6)
plot_crashes_sorted_by_crashes(crash_data, 6)

# Remove groups with too few values
keys = list(coverage_data.keys())
for key in keys:
    if len(coverage_data[key]) < min_tests_per_group:
        print("Removing: " + str(key) + " which only had " + str(len(coverage_data[key])) + " tests")
        coverage_data.pop(key)
        crash_data.pop(key)
        num_test_data.pop(key)

# Recompute number per box
number_per_box = []
for key in coverage_data:
    number_per_box.append(len(coverage_data[key]))

# Limiting each test set to have only a set number of tests
min_number_tests = min(number_per_box)

# Crop the number of tests
for key in coverage_data:
    coverage_data[key] = coverage_data[key][0:min_number_tests]
    crash_data[key] = crash_data[key][0:min_number_tests]
    num_test_data[key] = num_test_data[key][0:min_number_tests]

# Plotting before cliping
fig = plt.figure("After Clipping", figsize =(18, 5))
plot_number_of_test_suites(num_test_data, 1)
plot_coverage_vs_crashes(crash_data, 2)
plot_coverage_vs_number_tests(num_test_data, 3)
plot_number_tests_vs_crashes(num_test_data, crash_data, 4)
plot_coverage_vs_crashes_scatter(coverage_data, crash_data, 5)
plot_crashes_sorted_by_test_size(crash_data, num_test_data, 6)
plot_crashes_sorted_by_coverage(crash_data, coverage_data, 6)
plot_crashes_sorted_by_crashes(crash_data, 6)
plt.show()