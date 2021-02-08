import glob
import numpy as np
import matplotlib.pyplot as plt

file_names = glob.glob("./output/*.txt")

seen_before = []
total_observations = 0
coverage_plot_y = []
coverage_plot_color = []

vehicle_count_order = [1, 2, 5, 10, 15, 20, 25, 50]
colors = {1:'C0',
        2:'C1',
        5:'C2',
        10:'C3',
        15:'C4',
        20:'C5',
        25:'C6',
        50:'C7'}

for order in vehicle_count_order:
    for f_name in file_names:
        f = open(f_name, "r") 
        count_found = False
        for line in f: 

            if "External Vehicles: " in line:
                vehicle_count = int(line[line.find(": ")+2:])
                if vehicle_count != order:
                    break

            if "Vector: " in line:
                vector_str = line[line.find(": ")+3:-2]
                vector = np.fromstring(vector_str, dtype=float, sep=', ')

                # See if we have already seen this before
                already_seen = False
                for d in seen_before:
                    if np.array_equal(d,vector):
                        already_seen = True
                        break
                
                # If we havent seen it, add it
                if not already_seen:
                    seen_before.append(vector)

                total_observations += 1

        if vehicle_count == order:
            coverage_plot_y.append(len(seen_before))
            coverage_plot_color.append(colors[vehicle_count])

        f.close()

print("total observations: " + str(total_observations))
print("total unique observations: " + str(len(seen_before)))

# Compute the number of unique possible observations
vector_length = len(seen_before[0])
accuracy = 0.5
maximum = 15
unique_observations_per_cell = (maximum / float(accuracy)) + 1.0
total_possible_observations = pow(unique_observations_per_cell, vector_length)
print("total possible unique observations: " + str(total_possible_observations))

# Show the data
total_plotted = 0
for i in range(len(vehicle_count_order)):
    c = colors[vehicle_count_order[i]]
    y_data = []
    for j in range(len(coverage_plot_color)):
        if coverage_plot_color[j] == c:
            y_data.append(coverage_plot_y[j])

    x_data = np.arange(total_plotted, len(y_data)+total_plotted)
    total_plotted = x_data[-1] + 1
    plt.scatter(x_data, y_data, color=c, marker='o', label=str(vehicle_count_order[i]) + " vehicle(s)")

plt.legend(loc='upper left')
plt.show()