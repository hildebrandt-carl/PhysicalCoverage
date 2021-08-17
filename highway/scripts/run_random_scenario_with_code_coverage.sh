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
    for value in {1..200}
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

# tot_vehicle=1 
# mkdir -p code_coverage_results/external_vehicles_${tot_vehicle}/raw

# # loop
# coverage run --omit='/usr/lib/*,*/.local/*' --parallel-mode --branch run_random_scenario.py --no_plot --environment_vehicles ${tot_vehicle} --save_name test1.txt
# mv .coverage*  code_coverage_results/external_vehicles_${tot_vehicle}/raw

# # Processing
# cp code_coverage_results/external_vehicles_${tot_vehicle}/raw/.coverage* ./
# coverage combine 
# cp .coverage  code_coverage_results/external_vehicles_${tot_vehicle}
# coverage report >> "code_coverage.txt"
# mv code_coverage.txt code_coverage_results/external_vehicles_${tot_vehicle}
# coverage html
# mv htmlcov code_coverage_results/external_vehicles_${tot_vehicle}
# mv code_coverage_results/external_vehicles_${tot_vehicle}/htmlcov code_coverage_results/external_vehicles_${tot_vehicle}/html 
# rm .coverage

# # Final Grouping
# mkdir -p code_coverage_results/all_coverage
# cp  code_coverage_results/external_vehicles_${tot_vehicle}/.coverage .coverage${tot_vehicle}
# coverage combine --append .coverage1 .coverage2
# coverage report >> "all_code_coverage.txt"
# mv all_code_coverage.txt code_coverage_results/all_coverage/
# coverage html
# mv htmlcov code_coverage_results/all_coverage
# mv code_coverage_results/all_coverage/htmlcov code_coverage_results/all_coverage/html 

# I have the coverage I am now trying to get it to work...


# Get the annotations
# mkdir output
# coverage annotate --directory=output

# Get hmtl version
# coverage html

# Get the report
# coverage report >> "coverage.txt"

# Combine the data (note this removes the raw data)
# coverage combine --keep

