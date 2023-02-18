# Data Subset

We have provided the data as a zip file. The ziped files are 10.2GB when extracted they are roughly 81.5GB.



<!-- This folder contains a subset of the raw data, as well as the output of the RRS pipeline from all data processed in our study.

The data is structured in the following way:

```bash
├── environment name
├── feasibility
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── random_tests
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir3
├── file_in_root.ext
└── README.md
```

















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
