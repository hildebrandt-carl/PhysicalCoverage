#!/bin/bash

# Each of the beam counts
RRS_Number=(1 2 3 4 5 6 7 8 9 10)

# Launch counter
counter=0
total_cores=50

# Run it for each of the RRS
for RRS in "${RRS_Number[@]}"
do
    # Find all the tests
    tests=/media/carl/DataDrive/PhysicalCoverageData/highway/generated_tests/$1/tests/${RRS}_external_vehicles/*.txt

    echo "Processing RRS ${RRS}"

    # Run it 
    for testpath in ${tests}
    do
        # Get the file name
        testname="$(basename $testpath .txt)"
        echo "$testname"

        # Run the code
        python3 run_generated_scenario.py --number_traffic_vehicle ${RRS} --distribution $1 --test_name $testname &

        # Increment the counter
        counter=$((counter+1))

        if [[ "$counter" -ge total_cores ]]
        then
            wait
            counter=0
        fi

    done

done
