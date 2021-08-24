#!/bin/bash

# 10 different vehicle counts
vehicle_count=(1 2 3 4 5 6 7 8 9 10)

# Duplicate the code coverage file
cp ./config/code_coverage_config ./config/code_coverage_config$1 
sed -i "s/.coverage.tests/.coverage.tests$1/" ./config/code_coverage_config$1 

for tot_vehicle in "${vehicle_count[@]}"
do
    # Make the output directory
    mkdir -p ../output/random_tests/code_coverage/raw/external_vehicles_${tot_vehicle}

    # Run it 
    for value in {1..1800}
    do
    # For each vehicle count

        # Get the current time
        current_date=`date +%s`

        # Generate a random string to append to the front
        chars=abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ
        rand_string=""
        for i in {1..4} ; do
            rand_string="$rand_string${chars:RANDOM%${#chars}:1}"
        done
        
        # Create the save name
        save_name="$tot_vehicle-$current_date-$rand_string.txt"

        # Run the script
        coverage run --rcfile=./config/code_coverage_config$1 run_random_scenario.py --no_plot --environment_vehicles $tot_vehicle --save_name $save_name 
        sleep 0.01

        # Get the ID from the last run so you can match it to the process
        mv .coverage.tests$1.*  ../output/random_tests/code_coverage/raw/external_vehicles_${tot_vehicle}
        sleep 0.01
    done
done

rm ./config/code_coverage_config$1 
