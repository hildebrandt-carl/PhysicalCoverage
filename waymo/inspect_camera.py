import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.widgets import Button

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',     type=str,  default="/mnt/extradrive3/PhysicalCoverageData",          help="The path to the data")
parser.add_argument('--index',         type=str,  default="0",                                                help="The index of the image you want to view")
args = parser.parse_args()



index_list = []
for index in args.index.split(', '):
    
    index = int(index)

    folder = "{}/waymo/random_tests/physical_coverage/additional_data/scenario{:03d}".format(args.data_path, index)

    # Get all the pictures
    all_pictures = glob.glob("{}/camera_data/camera*.png".format(folder))
    all_pictures = sorted(all_pictures)


    # Create the subplot
    fig, axs = plt.subplots(2, 5, figsize=(19, 4))

    # Select pictures from a wide range of views
    selected_pictures =  np.linspace(0, len(all_pictures)-1, 10, dtype=int)

    # Plot the pictures
    for j, picture_number in enumerate(selected_pictures):
        img = mpimg.imread(all_pictures[picture_number])

        if j < 5 :
            axs[0, j].imshow(img)
            axs[0, j].axis('off')
        else:
            axs[1, j-5].imshow(img)
            axs[1, j-5].axis('off')
        
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)


plt.show()