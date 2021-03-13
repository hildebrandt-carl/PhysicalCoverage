# PhysicalStack

Graphs can be generated using:

BeamNG

$ python3 total_coverage.py --steering_angle 33 --beam_count -1 --max_distance=45 --accuracy 5 --total_samples 1423 --scenario beamng
$ python3 greedy_coverage.py --steering_angle 33 --beam_count 5 --max_distance=45 --accuracy 5 --total_samples 1423 --greedy_sample 1000 --scenario beamng
$ python3 crash_variance_coverage.py --steering_angle 33 --beam_count 5 --max_distance=45 --accuracy 5 --total_samples 1423 --total_random_test_suites 10000 --scenario beamng
$ python3 unique_vector_count.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples 1423 --beam_count 5 --scenario beamng



Highway

$ python3 total_coverage.py --steering_angle 30 --beam_count -1 --max_distance=30 --accuracy 5 --total_samples 1000000 --scenario highway
$ python3 greedy_coverage.py --steering_angle 30 --beam_count 5 --max_distance=30 --accuracy 5 --total_samples 1000000 --greedy_sample 1000 --scenario highway
$ python3 crash_variance_coverage.py --steering_angle 30 --beam_count 5 --max_distance=30 --accuracy 5 --total_samples 1000000 --total_random_test_suites 10000 --scenario highway
$ python3 unique_vector_count.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 1000000 --beam_count 5 --scenario highway


Done:
$ python3 total_coverage.py --steering_angle 33 --beam_count -1 --max_distance=45 --accuracy 5 --total_samples 1423 --scenario beamng
$ python3 greedy_coverage.py --steering_angle 33 --beam_count 5 --max_distance=45 --accuracy 5 --total_samples 1423 --greedy_sample 1000 --scenario beamng
$ python3 unique_vector_count.py --steering_angle 33 --max_distance=45 --accuracy 5 --total_samples 1423 --beam_count 5 --scenario beamng


$ python3 total_coverage.py --steering_angle 30 --beam_count -1 --max_distance=30 --accuracy 5 --total_samples 1000000 --scenario highway
$ python3 unique_vector_count.py --steering_angle 30 --max_distance=30 --accuracy 5 --total_samples 1000000 --beam_count 5 --scenario highway