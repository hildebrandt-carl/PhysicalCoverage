#!/bin/bash

# Each of the beam counts
beamcount=(6)

# Launch counter
counter=0
total_cores=50

# Run it for each of the total number of lines
for totallines in "${beamcount[@]}"
do
    # Find all the tests
    tests=../../PhysicalCoverageData/highway/generated_tests/tests_single/$1/${totallines}_beams/*_points.npy

    echo "Processing ${totallines} beams"

    # Run it 
    for testpath in ${tests}
    do
        # Get the file name
        testname="$(basename $testpath .npy)"
        echo "$testname"

        python3 run_generated_scenario.py --no_plot --total_samples $1 --test_name=$testname --total_beams=$totallines &

        # Increment the counter
        counter=$((counter+1))

        if [[ "$counter" -ge total_cores ]]
        then
            wait
            counter=0
        fi

    done

done
