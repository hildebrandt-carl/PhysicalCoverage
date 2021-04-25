import re
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def processFile(f):
    try:
        timing_information = []
        data_started = False

        # Go through each line
        for line in f: 

            # Wait for the real data:
            if not data_started:
                if "----------------" in line:
                    data_started = True
                continue 

            # Get the number of external vehicles
            if "Time: " in line:
                t = float(line[line.find(": ")+2:])
                timing_information.append(t)
    except Exception as e:
        print("Error for file: " + f.name)
        print(e)
    return timing_information

parser = argparse.ArgumentParser()
args = parser.parse_args()

number_beams = ["1","2","3","4","5","10"]
scenarios = ["highway", "beamng"]

beamng_data = []
highway_data = []

# For each beam and scenario
for num_beam in number_beams:
    for scenario in scenarios:
        
        print("----------------------------------------------------")
        print("Processing: Beam Count - " + str(num_beam))
       
        all_files = None
        if scenario == "beamng":
            all_files = glob.glob("../../PhysicalCoverageData/beamng/processed_timing/b" + str(num_beam) + "/*/*.txt")
        elif scenario == "highway":
            all_files = glob.glob("../../PhysicalCoverageData/highway/raw_timing/b" + str(num_beam) + "/*/*.txt")
        else:
            print("No scenario declared")
            exit()

        total_files = len(all_files)
        print("Total files found: " + str(total_files))

        # Create the array
        timing_array = np.array([])

        if total_files > 0:
            # Create the numpy array 
            timing_information_per_file = []

            # For each file
            file_count = 0
            for i in range(total_files):
                # Get the filename
                file_name = all_files[i]

                # Process the file
                f = open(file_name, "r")    
                t_info = processFile(f)
                f.close()

                timing_information_per_file.append(t_info)

            # Flatten the list
            flat_list = []
            for sublist in timing_information_per_file:
                for item in sublist:
                    flat_list.append(item)

            timing_array = np.array(flat_list)
        
        # Save the data
        if scenario == "beamng":
            beamng_data.append(timing_array)
        else:
            highway_data.append(timing_array)

        # Compute mean per file
        total_per_file = []
        for file_data in timing_information_per_file:
            total_per_file.append(np.sum(file_data))

        # Compute mean per scenario
        mean_per_scenario = np.mean(timing_array)
        median_per_scenario = np.median(timing_array)

        # Compute overall mean per file
        mean_per_file = np.mean(total_per_file)
        median_per_file = np.median(total_per_file)

        print("Scenario: " + str(scenario))
        print("Beam Count: " + str(num_beam))
        print("Mean time per scenario: " + str(np.round(mean_per_scenario, 4)))
        print("Median time per scenario: " + str(np.round(median_per_scenario, 4)))
        print("Mean time per file: " + str(np.round(mean_per_file, 4)))
        print("Median time per file: " + str(np.round(median_per_file, 4)))

# Create the plot
plt.figure("Final")
bpl = plt.boxplot(highway_data, positions=np.array(range(len(highway_data)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(beamng_data, positions=np.array(range(len(beamng_data)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='HighwayEnv')
plt.plot([], c='#2C7BB6', label='BeamNG')
plt.legend()

plt.xticks(range(0, len(number_beams) * 2, 2), number_beams)
plt.xlim(-2, len(number_beams)*2)
plt.xlabel("Total Vectors")
plt.ylabel("Time per Scenario (s)")
plt.tight_layout()

# Display the other data
plt.figure("Highway")
b = plt.boxplot(highway_data, sym='')
plt.xticks(range(1, len(number_beams) +1, 1), number_beams)
plt.plot([], c='#D7191C', label='HighwayEnv')
set_box_color(b, '#D7191C')
plt.xlabel("Total Vectors")
plt.ylabel("Time per Scenario (s)")
plt.tight_layout()

plt.figure("BeamNG")
b = plt.boxplot(beamng_data, sym='')
plt.xticks(range(1, len(number_beams) +1, 1), number_beams)
plt.plot([], c='#2C7BB6', label='BeamNG')
set_box_color(b, '#2C7BB6')
plt.xlabel("Total Vectors")
plt.ylabel("Time per Scenario (s)")
plt.tight_layout()

plt.show()