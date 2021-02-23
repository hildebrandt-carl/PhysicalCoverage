#!/bin/bash

vehicle_count=(3 5 7 9 11 13 15 17 19 21 23 25)
output_count=$1

for value in {1..2}
do
    for tot_vehicle in "${vehicle_count[@]}"
    do
        # Get the current time
        current_date=`date +%s`
        # Count how many have been processed
        let "output_count+=1" 
        save_name="$tot_vehicle-$current_date-$output_count.txt"

        # Run the script
        python3 main.py --no_plot --environment_vehicles $tot_vehicle --save_name $save_name
    done
done