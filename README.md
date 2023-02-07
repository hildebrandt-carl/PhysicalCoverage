# PhysCov: Physical Test Coverage for Autonomous Vehicles

Adequately exercising the behaviors of autonomous vehicles is fundamental to their validation. However, quantifying an autonomous vehicle's testing adequacy is challenging as the system's behavior is influenced both by its *state* as well as its *physical environment*. To address this challenge, our work builds on two insights. First, data sensed by the autonomous vehicle provides a unique spatial signature of the physical environment inputs.Second, given its current state, inputs residing outside the autonomous vehicle's physically reachable regions are less relevant to its behavior. Building on those insights, we introduce an abstraction that enables the computation of a physical environment-based coverage metric, *PhysCov*. The abstraction combines the sensor readings with a physical reachability analysis based on the vehicle's state and dynamics to determine the region of the environment that may affect the autonomous vehicle. It then characterizes that  region through a parameterizable geometric approximation that can trade quality for cost. Tests with the same characterizations are deemed to have had similar internal states and exposed to similar environments, and thus likely to exercise the same set of behaviors, while tests with distinct characterizations  will increase *PhysCov*. A study on two simulated and one real system examines *PhysCovs*'s ability to quantify an autonomous vehicle's test suite, showcases its characterization cost and precision, and investigates its correlation with failures found and potential for test selection.

# Prerequisites

All notes in this document are related to everything other than BeamNG. To run BeamNG please refer to the [BeamNG Environment README](./environments/beamng/README.md).

## Hardware

This software was developed and run on the following machine:

__Computer 1:__
* Operating System: Ubuntu 20.04
* CPU: AMD Ryzen Threadripper 3990X
* CPU Cores: 128
* RAM: 128 GB

However the entire artifact has also been tested on a machine with lower specifications:

__Computer 2:__
* Operating System: Ubuntu 20.04
* CPU: Intel Core i7-10750H
* CPU Cores: 12
* RAM: 16 GB

## Software

We require several python packages to run our software. To install them you can use the package manager `pip`. 

```bash
sudo apt install python3-pip
```

Next install the following packages:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install gym==0.18
python3 -m pip install matplotlib
python3 -m pip install coverage==5.5
python3 -m pip install shapely
python3 -m pip install /environments/highway/highway_env_v2
python3 -m pip install numpy==1.19.5
python3 -m pip install pandas==1.2.0
```

# Environments

### Running an example

Our study was run on three different environments. We give details on how to run each of the environments below:

## Highway-Env

Before we run the study, we can run a test scenario to showcase the highway environment in action.

```
cd environments/highway
python3 run_random_scenario.py --environment_vehicles 5 --save_name test
```

You should get the following output on your screen:

![highway environment](./readme_data/highway/highway_example.gif)

This should also create an `output` folder. The output folder will contain the physical coverage in a standardized format `raw`, as well as the code coverage in a `raw` format.

### Creating study data

In our study we generated 1,000,000 tests. To do that we can use the scripts provided in the in `environments/highway/scripts` folder. The script you want to run is `run_random_scenario.sh`. This script will generate 1000 tests (100 tests with 1 traffic vehicle, 100 tests with 2 traffic vehicles, ..., 100 tests with 10 traffic vehicles).

__Note:__ make sure you delete the output folder generated in `Running an example`.

To run the script you can use:

```bash
cd environments/highway
./scripts/run_random_scenarios.sh
```

This will execute significantly faster than the example, as the GUI is turned off, and running faster than time is enabled. It took __computer 2__ X time to complete.

__Note:__ Multiple of these scripts can be run in parallel. Simply open multiple terminals and run them. For the study we ran 50 scripts in parallel on __computer 1__.

## BeamNG

Please refer to the [BeamNG Environment README](./environments/beamng/README.md).

## Waymo

