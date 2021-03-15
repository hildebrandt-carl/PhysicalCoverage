import glob
import random 
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import argparse
import copy
import multiprocessing
import time
import itertools

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom')

def myround(x, base=5):
    return round(base * round(x/base), 5)

scenario = "beamng"
min_tests_per_group = 200
interval_size = 0.5

# Get the results
results = None
if scenario == "highway":
    results = np.load("../results/Final/crashes_vs_coverage/50k_crash_variance_highway.npy")
elif scenario == "beamng":
    results1 = np.load("../results/Final/crashes_vs_coverage/10k_1_crash_variance_beamng.npy")
    results2 = np.load("../results/Final/crashes_vs_coverage/10k_2_crash_variance_beamng.npy")
    results3 = np.load("../results/Final/crashes_vs_coverage/30k_crash_variance_beamng.npy")
    results = np.concatenate([results1, results2, results3], axis=0)


results = np.load("../results/crash_variance_highway.npy")


# Get the coverage / crashes / and number of tests
coverage = np.zeros(len(results))
crashes = np.zeros(len(results))
num_tests = np.zeros(len(results))
for i in range(len(results)):
    coverage[i] = results[i][0]
    crashes[i] = results[i][1]
    num_tests[i] = results[i][2]


# Plot the differnt relationships as a scatter plot
fig = plt.figure(1)
ax = fig.add_subplot(1, 3, 1)
ax.scatter(coverage, crashes, alpha=0.8, edgecolors='none', s=30)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of Crashes")
plt.title('Matplot scatter plot')
ax = fig.add_subplot(1, 3, 2)
ax.scatter(num_tests, crashes, alpha=0.8, edgecolors='none', s=30)
plt.xlabel("Number of tests in test suite")
plt.ylabel("Number of Crashes")
plt.title('Matplot scatter plot')
ax = fig.add_subplot(1, 3, 3)
ax.scatter(coverage, num_tests, alpha=0.8, edgecolors='none', s=30)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of tests in test suite")
plt.title('Matplot scatter plot')

# Create a box plot of the data
max_coverage = np.max(coverage) + interval_size
box_intervals = np.arange(0, max_coverage, interval_size)

# Create a dictionary to hold all the data
coverage_data = {}
crash_data = {}
for interval in box_intervals:
    coverage_data[str(myround(interval, interval_size))] = []
    crash_data[str(myround(interval, interval_size))] = []

# Break the data up into groups
for i in range(coverage.shape[0]):
    c = myround(coverage[i], interval_size)
    coverage_data[str(c)].append(coverage[i])
    crash_data[str(c)].append(crashes[i])


# Plot the distribution of values
print("-----------------")
print("Before normalizing test suite size")
print('-----------------')
number_per_box = []
coverage_box = []
for key in coverage_data:
    coverage_box.append(key)
    number_per_box.append(len(coverage_data[key]))
    print("Key: " + str(key) + " - " + str(len(coverage_data[key])))

# Plot the number of tests in each plot
fig = plt.figure(2, figsize =(10, 7))
ax = fig.add_subplot(1, 1, 1)
rects = ax.bar(coverage_box, number_per_box)
plt.title("Before normalizing number of tests")
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of test suites")
autolabel(rects)
    
# Create box plot
plot_data = []
label_data = []
for key in crash_data:
    plot_data.append(crash_data[key])
    label_data.append(key)

# Creat the box plot
fig = plt.figure(3, figsize =(10, 7)) 
# Creating plot 
bp = plt.boxplot(plot_data) 
plt.xticks(np.arange(1, len(label_data) + 1), label_data)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Crashes")
plt.title("All test suites")





print("-----------------")
print("Removing coverage data with too few points")
print('-----------------')

# Remove groups with too few values
keys = list(coverage_data.keys())
for key in keys:
    if len(coverage_data[key]) < min_tests_per_group:
        print("Removing: " + str(key) + " which only had " + str(len(coverage_data[key])) + " tests")
        coverage_data.pop(key)
        crash_data.pop(key)

number_per_box = []
coverage_box = []
for key in coverage_data:
    coverage_box.append(key)
    number_per_box.append(len(coverage_data[key]))
    print("Key: " + str(key) + " - " + str(len(coverage_data[key])))

# Limiting each test set to have only a set number of tests
min_number_tests = min(number_per_box)

# Crop the number of tests
for key in coverage_data:
    coverage_data[key] = coverage_data[key][0:min_number_tests]
    crash_data[key] = crash_data[key][0:min_number_tests]

# Plot the distribution of values
print("-----------------")
print("After normalizing test suite size")
print('-----------------')
number_per_box = []
coverage_box = []
for key in coverage_data:
    coverage_box.append(key)
    number_per_box.append(len(coverage_data[key]))
    print("Key: " + str(key) + " - " + str(len(coverage_data[key])))

# Plot the number of tests in each plot
fig = plt.figure(4, figsize =(10, 7))
ax = fig.add_subplot(1, 1, 1)
rects = ax.bar(coverage_box, number_per_box)
plt.title("After normalizing number of tests")
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Number of test suites")
autolabel(rects)
    
# Create box plot
plot_data = []
label_data = []
for key in crash_data:
    plot_data.append(crash_data[key])
    label_data.append(key)

# Creat the box plot
fig = plt.figure(5, figsize =(10, 7)) 
# Creating plot 
bp = plt.boxplot(plot_data) 
plt.xticks(np.arange(1, len(label_data) + 1), label_data)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Crashes")
plt.title("Each group contains " + str(min_number_tests) + " tests suites")


# show plot 
plt.show() 