# BeamNG

This will take you through the steps of setting up the BeamNG environment and using it to generate data used for our study. Note this was run on a computer running Windows 11. Please refer to Prerequisites for more information.

## Prerequisites

This was tested on the following computer:

| Computer   | CPU              | CPU Cores | RAM       | GPU                           | Operating System  |
|------------|--------------	|-------	|-------	|---------------------------    |---------------    |
| Computer 3 |  i7-10750H       | 12        | 16 GB     | Nvidia GeForce RTX 2060       | Windows 11        |

The operating system was freshly installed, updated and had the following software prior to setup:
* Git
* VSCode
* Nvidia Geforce Experience
* 7-Zip
* Python3.10.10

### Downloading BeamNG Tech

We will be using [BeamNG.Tech](https://beamng.tech/), as our simulation software. To get access to this you need to contact their licensing department, whose information is available on their [website](https://beamng.tech/). The software has state of the art soft body physics, has realisitc rendering and sensors, as well as allows for full control of multiple vehicles.

![BeamNG Tech Banner](../../misc/beamng/beamngtech.gif)

When you are granted access you will also be sent a `tech.key`. This key needs to be placed in your workspace (More information below) for the software to find and verify it. The same access email will also give you a link to download different versions of the software. We have tested on multiple versions, however this document is written for `BeamNG.tech-0.23.5.1`. Download it and place it in `C:\Users\<name>\BeamNG\BeamNG.tech-0.23.5.1`.

**Note:** This software has been tested on **Version 0.23.5.1**, however we have seen it working on other versions

### Downloading BeamNGpy

We will be using [BeamNGpy](https://github.com/BeamNG/BeamNGpy), to give us API access to control the simulation [BeamNG.Tech](https://beamng.tech/). To install the software run the following:

```bash
git clone git@github.com:BeamNG/BeamNGpy.git
git checkout tags/v1.21.1
python -m pip install --upgrade pip
python -m pip install -e .\BeamNGpy\
```

To make sure you have everything installed run the following command:

```bash
python -m pip list
```

You should get the `beamngpy - 1.21.1`  listed as well as the location of the package.

**Note:** This software has been tested on **Version 1.21.1**, however we have seen it working on other versions

Finally we need to install a few additional python packages. Run the following:

```bash
python -m pip install shapely
```

### Setting up your workspace

We now describe how to setup the workspace. For the official instructions you can follow this [README](https://github.com/BeamNG/BeamNGpy/tree/v1.21.1#prerequisites)

Using BeamNG.tech version 0.23.5.1 the workspace needs to be set up by the user. This step needs to be repeated for every newly installed BeamNG.tech version and helps BeamNGpy to determine the correct user directory for mod deployment. However to do this you can follow these steps:

1) Create a workspace directory named `C:\Users\<name>\BeamNG\BeamNGWorkspace` and then place your license file into it.

2) Create an environment variable called `BNG_HOME`. You can do that by:
Edit the system environment variables -> Environment Variables -> System Variables (NEW) 
Varibale Name: `BNG_HOME `
Variable Value: `C:\Users\<name>\BeamNG\BeamNG.tech-0.23.5.1`

3) Create both workspaces by, first restarting the terminal (if you have just created the environment variable) and running:
```bash
beamngpy setup-workspace C:\Users\<name>\BeamNG\BeamNGWorkspace
```

**Note**: If you get any errors about not knowing the `beamngpy` command, refer to this [issue](https://github.com/BeamNG/BeamNGpy/issues/203#event-8563612735)

If everything worked you will notice that the `BeamNG` simulator opens and then closes.

# Using BeamNG

## Testing Setup

First we recommend testing that you have set everything up correctly using the test code we have provided. We provide tests for:
* Checking if the basic API connection is working
* Checking if you are able to manually control the vehicle
* Checking if the LiDAR is working

### Checking Basic API Control

This test will make sure that you have put the `tech.key` in the right folder and that your `BeamNGpy` was installed correctly. Please read the "Downloading BeamNG Tech" to see where we expect both the key and software to be placed. To run this test do the following.

```bash
cd ./environments/beamng/setup_tests
python basic_drive.py
```

If all went well you should see the following:

![BeamNG Basic Drive Example](../../misc/beamng/basic_drive.gif)

**Note:** The car is set to automatically drive around the world, and so it might turn left, right or go straight at the first intersection.

### Testing Manual Control

Next lets test that you are able to manually control the vehicle. This will test that you are able to load onto the highway which our testing will occur. To do that run the following:

```bash
cd ./environments/beamng/setup_tests
python manual_drive.py
```

If everything worked you should see the following. **Note** you can drive around with the arrow keys. Here you can also see the soft body physics at work!

![BeamNG Basic Drive Example](../../misc/beamng/manual_drive.gif)

### Checking LiDAR is Working

The final test is to make sure that we are able to use the LiDAR sensor and get data from it. To do that we can run the following test:

```bash
cd ./environments/beamng/setup_tests
python basic_lidar.py
```

You will know that it is working if you start getting the following output to your terminal:
```
Getting (296271,) lidar points
Getting (297126,) lidar points
Getting (294804,) lidar points
...
Getting (291318,) lidar points
Getting (287622,) lidar points
Getting (284004,) lidar points
```

Additionally you will notice the LiDAR on the simulation GUI, as shown below:

![BeamNG Basic LiDAR Example](../../misc/beamng/basic_lidar.gif)


## Adding Code Coverage to Lua

The next step is to update BeamNG py with the ability to perform code coverage. To do that please follow the `code_coverage_instrumentation` [README](./code_coverage_instrumentation/README.md).

## Running Study

Once you have confirmed that you are able to run each of the basic tests, and that you have added the code coverage functionality to BeamNGpy, you are finally ready to generate the study data. To do that first lets run a single example. To do that run the following code:

```bash
cd ./environments/beamng/random_tests
python .\run_random_tests.py --total_runs 2 --traffic_count 3 --ai_mode limit --port 64256
```

Gif coming soon.

This will also produce an output folder that contains both physical coverage, as well as code coverage. Both of these will match the type of data found in our `data_subset`.