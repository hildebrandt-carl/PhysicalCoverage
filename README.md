# PhysicalStack

Below is the artifact for the paper.

## Installation

To install the simulator you need to do:

```bash
$ sudo apt install python3-pip -y
$ python3 -m pip install --upgrade pip
$ python3 -m pip install gym
$ python3 -m pip install numpy
$ python3 -m pip install matplotlib
$ python3 -m pip install tqdm
$ sudo apt install llvm-8
$ python3 -m pip install -e highway_env_v2
$ python3 -m pip install llvmlite==0.31.0
$ python3 -m pip install rl_agents_v2
$ python3 -m pip install networkx
```

Next run the following:
```bash
$ mkdir ~/Desktop/output
$ python3 main.py --environment_vehicles 10 --save_name test.txt
```

If this works you are ready to create the data

## Creating Data

You can create data using

```bash
$ cd highway
$ ./create_data.sh
```

This will save the data into you output folder

## Processing the data

Here we will explain how we converted the data

### Converting to Numpy array

First we need to convert the data into a numpy format. To do that you can run the following. For highwayEnv you need to run:
```bash
$ python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --beam_count 1
$ python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --beam_count 2
$ python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --beam_count 3
$ python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --beam_count 4
$ python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --beam_count 5
$ python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --beam_count 10
```

Next to convert beamng you need to run:
```bash
$ python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 1
$ python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 2
$ python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 3
$ python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 4
$ python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 5
$ python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 10
```

You will then need to move the output into the `highway/numpy_data` folder.

### Answering the research questions

To answer the research questions you need to run the following:

RQ1
```
$ python3 rq1_compute.py --steering_angle 30 --beam_count -1 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --cores 8
$ python3 rq1_compute.py --steering_angle 33 --beam_count -1 --max_distance=45 --accuracy 5 --total_samples 2000 --scenario beamng --cores 8 
```

RQ2
```
$ python3 rq2_compute.py
```

RQ3
```
$ python3 rq3_compute.py --steering_angle 30 --beam_count 3 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --cores 64
$ python3 rq3_compute.py --steering_angle 33 --beam_count 3 --max_distance=45 --accuracy 5 --total_samples 2000 --scenario beamng --cores 64
```

Computing the number of unique crashes
```
python3 rq3_compute_unique_crash_count.py --steering_angle 30 --beam_count 3 --max_distance=30 --accuracy 5 --total_samples 100000 --scenario highway --cores 10
python3 rq3_compute_unique_crash_count.py --steering_angle 33 --beam_count 3 --max_distance=45 --accuracy 5 --total_samples 2000 --scenario beamng --cores 10
```


Latest update

To start you will need to run the `./create_data.sh`. Each script will generate 1000 runs.
```bash
$ cd PhysicalCoverage/highway
$ ./scripts/create_data.sh
```

This will create an output folder which will contain the data from all the runs. The output folder will be `PhysicalCoverage/highway/output`. You can copy this output folder into the following space
```bash
$ cd PhysicalCoverage
$ cd ..
$ mkdir -p PhysicalCoverageData/highway/raw
$ mv PhysicalCoverage/highway/output/* PhysicalCoverageData/highway/raw
$ rm -r PhysicalCoverage/highway/output
```

Next we want to compute what is feasible and what is not feasible to do that you need to run the `./compute_feasibility.sh` script. You can do that using:
```bash
$ cd PhysicalCoverage/highway
$ ./scripts/compute_feasibility.sh
```

Once you are done that you will need to save its output into the same `PhysicalCoverageData` folder as the previous data. To do that you can use:
```bash
$ cd PhysicalCoverage
$ cd ..
$ mv PhysicalCoverage/highway/output/* PhysicalCoverageData/highway/
$ rm -r PhysicalCoverage/highway/output
```

Next we need to convert the data into a numpy file which we can process. To do that you need to run:
```bash
$ cd PhysicalCoverage/coverage_analysis
$ ./scripts/preprocess_highway.sh
```

Next we need to process the feasibility data. To do that we need to run:
```bash
$ cd PhysicalCoverage/coverage_analysis
$ ./scripts/preprocess_feasibility.sh
```

Now we need to move that data into the same `PhysicalCoverageData` folder as the previous data. To do that you can use:
```bash
$ cd PhysicalCoverage
$ cd ..
$ mkdir -p PhysicalCoverageData/highway/processed
$ mv PhysicalCoverage/coverage_analysis/output/processed PhysicalCoverageData/highway/feasibility
$ mv PhysicalCoverage/coverage_analysis/output/* PhysicalCoverageData/highway/processed
$ rm -r PhysicalCoverage/coverage_analysis/output
```

Now we can start using to the data and analyzing it. First we will start by running rq1. To do that you can run:
```bash
$ cd PhysicalCoverage/coverage_analysis
$ ./scripts/rq1_highway.sh
$ ./scripts/rq3_highway.sh
```

When you are ready you can generate new scenarios by first identifying scenarios which have not yet been seen. To do that you can run:
```bash
$ cd PhysicalCoverage/coverage_analysis
$ ./scripts/rq4_highway.sh
```

Then you need to move that data out of the folder into the data folder using:
```bash
$ cd PhysicalCoverage
$ cd ..
$ mkdir -p PhysicalCoverageData/highway/unseen
$ mv PhysicalCoverage/coverage_analysis/output/* PhysicalCoverageData/highway/unseen
$ rm -r PhysicalCoverage/coverage_analysis/output
```

Next we need to run those scenarios in highway to get the data from them. To do that we need to run. At this point we should know the number of samples we are using. In this case we are using 1000 samples.
```bash
$ cd PhysicalCoverage/highway
$ ./scripts/run_unseen_scenarios.sh 1000
```

Then we need to move the output data to the correct folder **Note you will need to figure out what the number is based on your output folder**
```bash
$ cd PhysicalCoverage
$ cd ..
$ mv  PhysicalCoverage/highway/output/50000/* PhysicalCoverageData/highway/unseen/50000/
$ rm -r PhysicalCoverage/highway/output
```

Next you need to preprocess the new data. You can do that using:
```bash
$ cd PhysicalCoverage/coverage_analysis
$ ./scripts/preprocess_highway_unseen.sh
```

Next we need to move that new data into the data folder. To do that you can use the following commands:
```bash
$ $ cd PhysicalCoverage
$ cd ..
$ mv PhysicalCoverage/coverage_analysis/output/* PhysicalCoverageData/highway/unseen/50000/processed
$ rm -r PhysicalCoverage/coverage_analysis/output
```