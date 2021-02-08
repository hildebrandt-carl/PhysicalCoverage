#!/bin/bash

vehicle_count=(1 2 5 10 15 20 25 50)
output_count=$1

for value in {1..100}
do
    for tot_vehicle in "${vehicle_count[@]}"
    do
        # Get the current time
        current_date=`date +%s`
        # Count how many have been processed
        let "output_count+=1" 
        save_name="$current_date-$output_count.txt"

        # Run the script
        python3 main.py --no_plot --environment_vehicles $tot_vehicle --save_name $save_name
    done
done