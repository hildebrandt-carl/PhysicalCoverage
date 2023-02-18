# Processing Pipeline

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

Here you can see the date and time the test was run. What parameters were used for the initial computation. Our pipeline works by creating a vector 30 wide, which we then need to convert into the appropriate RRS vectors.

To do that we use the `preprocessing` folder.

## Highway

### Physical Coverage

To generate the physical coverage on `highway-env` you need to run the `preprocess_highway_random_physcov_coverage` script. The script has the following parameters
* Number of tests: In our datasubset we give you 1000
* Distribution: In our study we use center_close
* Path to data: The __FULL__ path to the data folder

An example of running the script is below:
```bash
cd preprocessing/
./scripts/preprocess_highway_random_physcov_coverage.sh 1000 center_close <full path to folder>/PhysicalCoverage/data_subset
```

If everything is run correctly you should have the following output on your terminal
![highway physcov preprocessing](./misc/highway/physcov_preprocessing.png)

Once this is done running you will have an `output` folder. This script will have generated several `npy` files in the folder `output/highway/random_tests/physical_coverage/processed/center_close/1000/`. You will have the following numpy files:

* `crash_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains up to 10 hashes for each test, where each hash describes a crash based on the velocity and angle of incident.
* `ego_positions_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the ego position in the world frame for each test.
* `processed_files_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the original file name used to create each row in the numpy arrays.
* `stall_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains up to 10 hashes for each test, where each hash describes a stall based on the angle and distance to the closest vehicle.
* `time_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the total time of each test.
* `traces_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the RRS vectors for each step of each test.
* `vehicles_hash_highway_random_s{X}_b{Y}_d{Z}_t{W}.npy`: Contains the total number of traffic vehicles in each test.

Where `X` is the steering angle. `Y` is the RRS number. `Z` is the maximum distance the vehicle can travel in 1 time step. `W` is the number of tests used.

__Note__: This data in output will match what was given to you in the `data_subset`.

### Code Coverage

To generate the code coverage on `highway-env` you need to run the `preprocess_highway_random_code_coverage` script. The script has the following parameters
* Number of tests: In our datasubset we give you 1000
* Path to data: The __FULL__ path to the data folder
An example of running the script is below:
```bash
cd preprocessing/
./scripts/preprocess_highway_random_code_coverage.sh 1000 <full path to folder>/PhysicalCoverage/data_subset
```

If everything is run correctly you should have the following output on your terminal
![highway codecov preprocessing](./misc/highway/codecov_preprocessing.png)

Once this is done running you will have an `output` folder. This script will have generated several `txt` files in the folder `output/highway/random_tests/code_coverage/processed/1000`. Each file generated is for a single test. Each file contains:

* `File name`: The file(s) which was monitored using code coverage
* `Lines covered`: The lines which were covered
* `All lines`: All lines in the file
* `Branches covered`: The branches which were taken
* `All branches`: All branches 
* `Intraprocedural prime path signature`: The hash of the intraprocedural prime path
* `Intraprocedural path signature`: The hash of the intraprocedural path
* `Absolute path signature`: The hash of the absolute path

__Note__: This data in output will match what was given to you in the `data_subset`.

## BeamNG

Will be completed by Sunday Feb 19th

## Waymo

Will be completed by Sunday Feb 19th