import glob
import re
import numpy as np
import matplotlib.pyplot as plt

coverage_array = [10,20,30,40,50,60,70,80,90]

plt.plot(coverage_array)
plt.ylim([-5,105])
plt.grid()
plt.yticks(np.arange(0, 100.01, step=5))
plt.xlabel("Number of tests")
plt.ylabel("Code Coverage (%)")
plt.title("Total Code Coverage: {}%".format(np.round(coverage_array[-1], 2)))
plt.show()