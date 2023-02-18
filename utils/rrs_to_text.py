import numpy as np
import cv2
import copy

background = cv2.imread('back.jpg',cv2.IMREAD_UNCHANGED)

traces = np.load("traces.npy")
trace = traces[8]

for i, rrs in enumerate(trace):
    print(rrs)
    new_background = copy.deepcopy(background)
    position = (10,500)
    cv2.putText(new_background, "{}".format(rrs), position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0, 255), 3) 
    cv2.imwrite('rrs_text/rrs{0:03d}.png'.format(i), new_background)