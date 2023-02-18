import os
import cv2
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,  default="",                  help="The path to the data")
parser.add_argument('--output',         type=str,  default="output",            help="The path to the data")
args = parser.parse_args()

image_folder = '{}'.format(args.data_path)
video_name = '{}.avi'.format(args.output)

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = sorted(images)
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in tqdm(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()