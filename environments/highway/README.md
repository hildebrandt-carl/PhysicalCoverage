# Highway-Env

This readme describes how to both run examples of scenarios from HighwayEnv as well as how we generated data in the study

**Note:** Please make sure to have completed the Prerequisites from the original [Readme.MD](../../README.md).

## Running an example

Before we run the study, we can run a test scenario to showcase the highway environment in action.

```
cd environments/highway
python3 run_random_scenario.py --environment_vehicles 5 --save_name test
```

You should get the following output on your screen:

![highway environment](../../misc/highway/highway_example.gif)

This should also create an `output` folder. The output folder will contain the physical coverage in a standardized format `raw`, as well as the code coverage in a `raw` format.

## Creating study data

In our study we generated 1,000,000 tests. To do that we can use the scripts provided in the in `environments/highway/scripts` folder. The script you want to run is `run_random_scenario.sh`. This script will generate 100 tests:
* 10 tests with 1 traffic vehicle
* 10 tests with 2 traffic vehicle
* ...
* 10 tests with 10 traffic vehicle


__Note:__ make sure you delete the output folder generated in `Running an example`, otherwise you will end up with 101 tests (100 from the script and 1 from your test run).

To run the script you can use:

```bash
cd environments/highway
./scripts/run_random_scenarios.sh
```

This will execute significantly faster than the example, as the GUI is turned off, and the environment will be set to run faster than time. It took __computer 2__ 5 minutes to complete.

__Note:__ Multiple of these scripts can be run in parallel. Simply open multiple terminals and run them. For the study we ran 50 scripts in parallel on __computer 1__.