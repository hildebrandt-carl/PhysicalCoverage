#!/bin/bash

# Each of the beam counts
beamcount=(1 2 3 4 5)

# Launch counter
counter=0
total_cores=50

# Run it for each of the total number of lines
for totallines in "${beamcount[@]}"
do
    # Find all the tests
    tests=($( ls ../../PhysicalCoverageData/highway/unseen/$1/tests_single/${totallines}_beams/*_points.npy ))

    echo "Processing ${totallines} beams"

    # Run it 
    for testpath in "${tests[@]}"
    do
        # Get the file name
        testname="$(basename $testpath .npy)"
        echo "$testname"
        for i in {1..1}
        do
            python3 run_test_scenario.py --no_plot --total_samples $1 --test_name=$testname --total_beams=$totallines &
        done
        
        # Increment the counter
        counter=$((counter+1))

        if [[ "$counter" -ge total_cores ]]
        then
            wait
            counter=0
        fi

    done

done
