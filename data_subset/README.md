# Data Subset

## Prerequisites

We require that you have 7-Zip installed in order to extract the data. 7-Zip, a free, open-source file compression tool that support many different formats. You can learn more about 7-Zip [on their website](https://www.7-zip.org/).

You can install 7-Zip on ubuntu using:
```bash
sudo apt install p7zip-full p7zip-rar
```

## Downloading the Data

We have provided the data as a zip file. The ziped files are 10.2GB when extracted they are roughly 81.6GB.

To download the data run the `extract_data.sh` script. An example of running it is shown below:

```bash
cd /data_subset
./extract_data.sh
```

The script's output will look as follows:
```
Do you have 7z Installed? [y/N] y
You are about to download a 10GB zip file. Are you sure? [y/N] y
It will then extract the data taking up an additional 80GB of space. Are you sure? [y/N] y
data.zip      90%[=====================================>             ]   8.56G   112MB/s    eta 9s
...
```

Once it has completed you should have a `highway`, `beamng` and `waymo` folder.


## Understanding the data

The data subset contains both the raw data output from the simulators, as well as the output of our RRS pipeline. In general our data is structured in the following way:


```bash
├── environment name
├── feasibility
│   ├── raw
│   │   └── # The raw feasibility files
│   └── processed
│       └── # The processed feasibility files
├── random_tests
│   ├── code_coverage
│   |   ├── raw
│   │   │   └── # The code coverage output from the simulator
│   │   └── processed
│   │       └── # The code coverage after processing into a standard format by our tool
│   └── physical_coverage
│   |   ├── lidar/frames
│   │   │   └── # The sensor data output from the simulator
│   |   ├── raw
│   │   │   └── # The data in our standard format
│   │   └── processed
│   │       └── # The RRS data structured in numpy arrays
└── README.md
```

## Physical Coverage

### Lidar/frames

This is not present in the `highway` environment. The highway environment automatically outputs the standard format used by our tool. The `beamng` environment LiDAR data as a csv file which is stored in the `lidar` folder. The `waymo` environment stores LiDAR and camera data in as `frames` and such they are stored in the `frames` directory.

### Raw

The raw format contains a version of the PhysCov vectors. For example if you look at `data_subset/highway/random_tests/physical_coverage/raw/1_external_vehicles/1-1639594889-21ox.txt` you will see the following:

```
Name: 1-1639594889-21ox
Date: 15/12/2021
Time: 14:1:30
External Vehicles: 1
Reach set total lines: 31
Reach set steering angle: 30
Reach set max distance: 30
------------------------------
Vector: [11.998, 12.778, 13.685, 14.749, 16.014, 17.54, 19.413, 21.764, 24.797, 28.854, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 29.235, 26.692, 24.583, 22.809, 21.298, 19.998]
Ego Position: [312.7561   4.    ]
Ego Velocity: [24.2558  0.    ]
Crash: False
Collided: False
Operation Time: 0.012143
Total Wall Time: 0.030084
Total Simulated Time: 0.25

Vector: [11.998, 12.778, 13.685, 14.749, 16.014, 17.54, 19.413, 21.764, 24.797, 28.854, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 29.235, 26.692, 24.583, 22.809, 21.298, 19.998]
Ego Position: [317.7302   4.    ]
Ego Velocity: [25.9657  0.    ]
Crash: False
Collided: False
Operation Time: 0.011229
Total Wall Time: 0.059326
Total Simulated Time: 0.5

...
```

Here you can see the date and time the test was run. What parameters were used for the initial computation. Our pipeline works by creating a vector 30 wide, which we then need to convert into the appropriate RRS vectors. -->

### Processed

This holds a series of files. The files include:
* `crash_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains up to 10 hashes for each test, where each hash describes a crash based on the velocity and angle of incident.
* `ego_positions_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the ego position in the world frame for each test.
* `processed_files_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the original file name used to create each row in the numpy arrays.
* `stall_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains up to 10 hashes for each test, where each hash describes a stall based on the angle and distance to the closest vehicle.
* `time_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the total time of each test.
* `traces_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the RRS vectors for each step of each test.
* `vehicles_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the total number of traffic vehicles in each test.

To view the 671st test from RRS-5 values in from the `waymo` environment, you can run the following in your terminal:

```bash
cd ./data_subset/waymo/random_tests/physical_coverage/processed/center_full/798
python3
>>> import numpy as np
>>> rrs = np.load("traces_waymo_random_s33_b5_d35_t798.npy")
>>> print(np.shape(rrs))
(798, 200, 5)
>>> print(rrs[670])
[[15. 25. 35. 35. 35.]
 [15. 25. 35. 35. 35.]
 ...
 [15. 35. 20. 35. 35.]
 [15. 35. 20. 35. 35.]
 [nan nan nan nan nan]
 [nan nan nan nan nan]]
>>> exit()
```