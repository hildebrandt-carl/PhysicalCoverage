#!/bin/bash

# 12 different vehicle counts
vehicle_count=(1 2 3 4 5 6 7 8 9 10)

total_tests=100

mkdir coverage_results_branch
mkdir coverage_results

# For each vehicle count
for tot_vehicle in "${vehicle_count[@]}"
do
    # Run it 
    for value in $( seq 0 $total_tests )
    do
        # Get the current time
        current_date=`date +%s`

        # Generate a random string to append to the front
        chars=abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ
        rand_string=""
        for i in {1..4} ; do
            rand_string="$rand_string${chars:RANDOM%${#chars}:1}"
        done
        
        # Create the save name
        save_name="coverage_$tot_vehicle-$current_date-$rand_string.txt"

        # Run the script
        coverage run --omit='/usr/lib/*,*/.local/*' --parallel-mode main.py --no_plot --environment_vehicles $tot_vehicle --save_name $save_name
    done

    # Combine the coverage
    coverage combine

    # Get the results
    coverage report >> "$tot_vehicle-coverage.txt"
    coverage html

    mv htmlcov "$tot_vehicle-coverage"
    mv "$tot_vehicle-coverage.txt" "$tot_vehicle-coverage"
    mv "$tot_vehicle-coverage" coverage_results

    # delete the results
    rm .coverage
done


# For each vehicle count
for tot_vehicle in "${vehicle_count[@]}"
do
    # Run it 
    for value in $( seq 0 $total_tests )
    do
        # Get the current time
        current_date=`date +%s`

        # Generate a random string to append to the front
        chars=abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ
        rand_string=""
        for i in {1..4} ; do
            rand_string="$rand_string${chars:RANDOM%${#chars}:1}"
        done
        
        # Create the save name
        save_name="coverage_$tot_vehicle-$current_date-$rand_string.txt"

        # Run the script
        coverage run --omit='/usr/lib/*,*/.local/*' --parallel-mode --branch main.py --no_plot --environment_vehicles $tot_vehicle --save_name $save_name
    done

    # Combine the coverage
    coverage combine

    # Get the results
    coverage report >> "$tot_vehicle-coverage-branch.txt"
    coverage html

    mv htmlcov "$tot_vehicle-coverage-branch"
    mv "$tot_vehicle-coverage-branch.txt" "$tot_vehicle-coverage-branch"
    mv "$tot_vehicle-coverage-branch" coverage_results_branch

    # delete the results
    rm .coverage
done