import glob
import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.widgets import Button



class callbacks:
    def __init__(self):
        self.labels = []

    def highway(self, event):
        print("highway")
        self.labels.append(1)
        plt.close()

    def neighborhood(self, event):
        print("neighborhood")
        self.labels.append(2)
        plt.close()

    def intersection(self, event):
        print("intersection")
        self.labels.append(3)
        plt.close()

    def single_lane(self, event):
        print("single_lane")
        self.labels.append(4)
        plt.close()

    def multi_lane(self, event):
        print("multi_lane")
        self.labels.append(5)
        plt.close()


# city
# parkinglot

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',     type=str,  default="/mnt/extradrive3/PhysicalCoverageData",          help="The path to the data")
args = parser.parse_args()

all_scenarios = glob.glob("{}/waymo/random_tests/physical_coverage/additional_data/scenario*".format(args.data_path))
all_scenarios = sorted(all_scenarios)



cb = callbacks()
for i, folder in enumerate(all_scenarios):

    print("Processing: {}".format(folder[folder.rfind("/")+1:]))

    # Create the subplot
    fig, axs = plt.subplots(1, 5, figsize=(35, 5))

    # Get all the pictures
    all_pictures = glob.glob("{}/camera_data/camera*.png".format(folder))
    all_pictures = sorted(all_pictures)

    # Select pictures from a wide range of views
    selected_pictures = [1, 50, 100, 150, -1]

    # Plot the pictures
    for j, picture_number in enumerate(selected_pictures):
        img = mpimg.imread(all_pictures[picture_number])
        axs[j].imshow(img)
        axs[j].axis('off')

    
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Add buttons
    highway_ax          = fig.add_axes([0.3, 0.05, 0.1, 0.075])
    neighborhood_ax     = fig.add_axes([0.4, 0.05, 0.1, 0.075])
    intersection_ax     = fig.add_axes([0.5, 0.05, 0.1, 0.075])
    single_lane_ax      = fig.add_axes([0.6, 0.05, 0.1, 0.075])
    multi_lane_ax       = fig.add_axes([0.7, 0.05, 0.1, 0.075])

    highway_btn         = Button(highway_ax, 'Highway')
    highway_btn.on_clicked(cb.highway)
    neighborhood_btn    = Button(neighborhood_ax, 'Neighborhood')
    neighborhood_btn.on_clicked(cb.neighborhood)
    intersection_btn    = Button(intersection_ax, 'Intersection')
    intersection_btn.on_clicked(cb.intersection)
    single_lane_btn     = Button(single_lane_ax, 'Single Lane')
    single_lane_btn.on_clicked(cb.single_lane)
    multi_lane_btn      = Button(multi_lane_ax, 'Multi Lane')
    multi_lane_btn.on_clicked(cb.multi_lane)

    print("")
    plt.show()


# Save the labels
with open("labels.txt", 'w') as output:
    for row in cb.labels:
        output.write(str(row) + '\n')