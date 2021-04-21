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













