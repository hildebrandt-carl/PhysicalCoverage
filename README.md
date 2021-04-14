# PhysicalStack


# Creating the Highway data
cd into highway
./create_data.sh

you need
python3 -m pip install --upgrade pip
python3 -m pip install gym
sudo apt install llvm-8
python3 -m pip install -e highway_env_v2
python3 -m pip install llvmlite==0.31.0
python3 -m pip install rl_agents_v2
python3 -m pip install networkx

Create an output folder on your desktop
then run
python3 main.py --environment_vehicles 10 --save_name test.txt

## Creating data
First you create the data using:

python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples -1 --scenario highway --beam_count 1
python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples -1 --scenario highway --beam_count 2
python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples -1 --scenario highway --beam_count 3
python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples -1 --scenario highway --beam_count 4
python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples -1 --scenario highway --beam_count 5
python3 pre_process_data.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples -1 --scenario highway --beam_count 10

python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 1
python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 2
python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 3
python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 4
python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 5
python3 pre_process_data.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples -1 --scenario beamng --beam_count 10

## Generating Graphs
Graphs can be generated using:


### Highway-Env

python3 rq1_compute.py --steering_angle 30 --beam_count -1 --max_distance=30 --accuracy 5 --total_samples 1000 --scenario highway --cores 8
python3 rq2_compute.py --steering_angle 30 --beam_count 3 --max_distance=30 --accuracy 5 --total_samples 1000000 --scenario highway --cores 64

### BeamNG

python3 rq1_compute.py --steering_angle 33 --beam_count -1 --max_distance=45 --accuracy 5 --total_samples 1000 --scenario beamng --cores 8
python3 rq2_compute.py --steering_angle 33 --beam_count 3 --max_distance=45 --accuracy 5 --total_samples 2000 --scenario beamng --cores 64

# Old


### Beamng

$ python3 total_coverage.py --steering_angle 33 --beam_count -1 --max_distance=45 --accuracy 5 --total_samples 1423 --scenario beamng
$ python3 greedy_coverage.py --steering_angle 33 --beam_count 5 --max_distance=45 --accuracy 5 --total_samples 1423 --greedy_sample 1000 --scenario beamng
$ python3 crash_variance_coverage.py --steering_angle 33 --beam_count 5 --max_distance=45 --accuracy 5 --total_samples 1423 --total_random_test_suites 100000 --scenario beamng
$ python3 unique_vector_count.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples 1423 --beam_count 5 --scenario beamng

### Highway-Env

$ python3 total_coverage.py --steering_angle 30 --beam_count -1 --max_distance=30 --accuracy 5 --total_samples 1000000 --scenario highway
$ python3 greedy_coverage.py --steering_angle 30 --beam_count 5 --max_distance=30 --accuracy 5 --total_samples 1000000 --greedy_sample 1000 --scenario highway
$ python3 crash_variance_coverage.py --steering_angle 30 --beam_count 5 --max_distance=30 --accuracy 5 --total_samples 1000000 --total_random_test_suites 100000 --scenario highway
$ python3 unique_vector_count.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 1000000 --beam_count 5 --scenario highway


## Prereq

sudo apt install python3-pip -y
python3 -m pip install numpy
python3 -m pip install matplotlib
python3 -m pip install tqdm
