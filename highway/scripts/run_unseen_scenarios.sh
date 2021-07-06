#!/bin/bash

# Each of the beam counts
beamcount=(1 2 3 4 5 10)

# Run it for each of the total number of lines
for totallines in "${beamcount[@]}"
do
    # Find all the tests
    tests=($( ls ../../PhysicalCoverageData/highway/unseen/$1/tests/${totallines}_beams/* ))

    echo "Processing ${totallines} beams"

    # Run it 
    for testpath in "${tests[@]}"
    do
        # Get the file name
        testname="$(basename $testpath .npy)"
        echo "$testname"
        python3 run_test_scenario.py --no_plot --total_samples $1 --test_name=$testname --total_beams=$totallines
    done

done
