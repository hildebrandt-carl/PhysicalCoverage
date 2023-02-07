#!/bin/bash

# 10 different vehicle counts
vehicle_count=(1 2 3 4 5 6 7 8 9 10)

# Run it 
for value in {1..10}
do
    for tot_vehicle in "${vehicle_count[@]}"
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
        save_name="$tot_vehicle-$current_date-$rand_string"

        # Run the script
        python3 run_random_scenario.py --no_plot --environment_vehicles $tot_vehicle --save_name $save_name 
        sleep 0.01
    done
done