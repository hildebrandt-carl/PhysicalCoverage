







# Creating the Artifcat
This stage describes how we can create the artifact

First you need to run the command:
```
python3 generate_artifact.py
```

This will generate a `PhysicalCoverageDataSubSet` on your desktop.

# Processing the Artifact

The code was set to run on a 4 core PC for review.
However all the code has been multithreaded and tested up to 128 cores.

## Creating the feasible vectors

Then you need to create the feasible vectors. To do that you need to run the following command:
```
cd preprocessing
cd ~/Desktop/PhysicalCoverage/preprocessing
./scripts/preprocess_highway_feasibility.sh /home/carl/Desktop/PhysicalCoverageDataSubSet
./scripts/preprocess_beamng_feasibility.sh /home/carl/Desktop/PhysicalCoverageDataSubSet
```

Next we need to move the folders from `output/<scenario>/feasibility/processed` into `/home/carl/Desktop/PhysicalCoverageDataSubSet/<scenario>/feasibility/processed`
You can then delete `output`

## Converting Lidar data into Raw data

Next we need to convert BeamNG's lidar data into a format that our pipeline accepts. To do that you need to run the following:
```
cd ~/Desktop/PhysicalCoverage/beamng_converter
python3 lidar_to_RRS.py --cores 4 --scenario beamng_random --data_path '/home/carl/Desktop/PhysicalCoverageDataSubSet'
```

Next we need to move those folders from `output/beamng/random_tests/physical_coverage/raw` into `/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/physical_coverage/raw`
You can then delete `output`

## Processing Physical Coverage

Next lets process the physical coverage. You can do that by running:
```
cd ~/Desktop/PhysicalCoverage/preprocessing 
./scripts/preprocess_highway_random_physcov_coverage.sh 100 center_close /home/carl/Desktop/PhysicalCoverageDataSubSet
./scripts/preprocess_beamng_random_physcov_coverage.sh 100 center_close /home/carl/Desktop/PhysicalCoverageDataSubSet
```

Next we need to move those folders from `output/<scenario>/random_tests/physical_coverage/processed` into `/home/carl/Desktop/PhysicalCoverageDataSubSet/<scenario>/random_tests/physical_coverage/processed`

## Processing the Code Coverage

Next lets process the strucutral code coverage. You can do that by running:
```
cd ~/Desktop/PhysicalCoverage/preprocessing 
./scripts/preprocess_highway_random_code_coverage.sh 100 /home/carl/Desktop/PhysicalCoverageDataSubSet
./scripts/preprocess_beamng_random_code_coverage.sh 100 /home/carl/Desktop/PhysicalCoverageDataSubSet
```

Next we need to move those folders from `output/<scenario>/random_tests/code_coverage/processed` into `/home/carl/Desktop/PhysicalCoverageDataSubSet/<scenario>/random_tests/code_coverage/processed`

# Computing Results

At this point we can start running some of the results shown in the paper.
Note these result will differ as we are using a significantly smaller dataset.

## RQ 1

We can compute the number of signatures using:
```
cd ~/Desktop/PhysicalCoverage/coverage_analysis
python3 RQ1_sigature_test.py --number_of_tests 100 --distribution center_close --scenario highway --cores 4 --data_path /home/carl/Desktop/PhysicalCoverageDataSubSet
python3 RQ1_sigature_test.py --number_of_tests 100 --distribution center_close --scenario beamng --cores 4 --data_path /home/carl/Desktop/PhysicalCoverageDataSubSet
```

We can then compute then compute the signatures vs failures graph using:
```
cd ~/Desktop/PhysicalCoverage/coverage_analysis
python3 RQ1_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 100 --distribution center_close --scenario highway --cores 4 --data_path /home/carl/Desktop/PhysicalCoverageDataSubSet    
python3 RQ1_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 100 --distribution center_close --scenario beamng --cores 4 --data_path /home/carl/Desktop/PhysicalCoverageDataSubSet 
```

## RQ 2

First lets compute the test selection process to do that you can run:
```
cd ~/Desktop/PhysicalCoverage/coverage_analysis
python3 RQ2_test_selection.py --number_of_tests 100 --distribution center_close --RRS_number 10 --scenario highway --cores 4 --data_path /home/carl/Desktop/PhysicalCoverageDataSubSet 
python3 RQ2_test_selection.py --number_of_tests 100 --distribution center_close --RRS_number 10 --scenario beamng --cores 4 --data_path /home/carl/Desktop/PhysicalCoverageDataSubSet 
```



## Other plots

RQ1 Tables
python3 RQ1_sigature_test.py --number_of_tests 10000 --distribution linear --scenario beamng --cores 50
python3 RQ1_sigature_test.py --number_of_tests 10000 --distribution center_close --scenario beamng --cores 50
python3 RQ1_sigature_test.py --number_of_tests 10000 --distribution center_mid --scenario beamng --cores 50

python3 RQ1_sigature_test.py --number_of_tests 1000000 --distribution linear --scenario highway --cores 50
python3 RQ1_sigature_test.py --number_of_tests 1000000 --distribution center_close --scenario highway --cores 50
python3 RQ1_sigature_test.py --number_of_tests 1000000 --distribution center_mid --scenario highway --cores 50

RQ1 Plots

python3 RQ2_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 10000 --distribution linear --scenario beamng --cores 50
python3 RQ2_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 10000 --distribution center_close --scenario beamng --cores 50
python3 RQ2_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 10000 --distribution center_mid --scenario beamng --cores 50

python3 RQ2_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 1000000 --distribution linear --scenario highway --cores 50
python3 RQ2_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 1000000 --distribution center_close --scenario highway --cores 50
python3 RQ2_coverage_vs_failures.py --random_test_suites 10 --number_of_tests 1000000 --distribution center_mid --scenario highway --cores 50


RQ3 Plots

python3 RQ3_test_selection.py --number_of_tests 10000 --distribution linear --RRS_number 5 --scenario beamng --cores 50
python3 RQ3_test_selection.py --number_of_tests 10000 --distribution center_close --RRS_number 5 --scenario beamng --cores 50
python3 RQ3_test_selection.py --number_of_tests 10000 --distribution center_mid --RRS_number 5 --scenario beamng --cores 50

python3 RQ3_test_selection.py --number_of_tests 1000000 --distribution linear --RRS_number 5 --scenario highway --cores 50
python3 RQ3_test_selection.py --number_of_tests 1000000 --distribution center_close --RRS_number 5 --scenario highway --cores 50
python3 RQ3_test_selection.py --number_of_tests 1000000 --distribution center_mid --RRS_number 5 --scenario highway --cores 50
