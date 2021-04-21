import numpy as np
import glob

coverage_files = glob.glob("./raw_data/*_coverage_*.npy")

coverage_data = None
crash_data = None

for i in range(len(coverage_files)):

    f_cov = coverage_files[i]
    f_cra = f_cov.replace("coverage", "crashes")

    tmp_cov = np.load(f_cov)
    tmp_cra = np.load(f_cra)

    if coverage_data is None:
        coverage_data = tmp_cov
        crash_data = tmp_cra
    else:
        coverage_data = np.concatenate((coverage_data, tmp_cov), axis=1)
        crash_data = np.concatenate((crash_data, tmp_cra), axis=1)

np.save("rq3_coverage_highway.npy", coverage_data)
np.save("rq3_crashes_highway.npy", crash_data)
