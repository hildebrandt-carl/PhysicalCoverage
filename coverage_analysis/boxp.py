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

def myround(x, base=5):
    return round(base * round(x/base), 5)

results1 = np.load('data1.npy')
results2 = np.load('data2.npy')
# results3 = np.load('data3.npy')
results = np.concatenate([results1, results2], axis=0)
print(results.shape)

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
interval_size = 0.1
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
    
# Create box plot
plot_data = []
label_data = []
for key in crash_data:
    plot_data.append(crash_data[key])
    label_data.append(key)

# Creat the box plot
fig = plt.figure(2, figsize =(10, 7)) 
# Creating plot 
bp = plt.boxplot(plot_data) 
plt.xticks(np.arange(1, len(label_data) + 1), label_data)
plt.xlabel("Physical Coverage (%)")
plt.ylabel("Crashes")
# show plot 
plt.show() 
