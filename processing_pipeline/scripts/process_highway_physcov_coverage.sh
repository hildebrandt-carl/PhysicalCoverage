#!/bin/bash

# Each of the beam counts
beamcount=(1 2 3 4 5 6 7 8 9 10)

# Run it for each of the total number of lines
for totallines in "${beamcount[@]}"
do
    # Run the script
    if [ $# -eq 2 ]
    then
        python3 process_physcov.py --scenario highway_random --cores 16 --total_samples $1  --beam_count $totallines --distribution $2
    fi

    if [ $# -eq 3 ]
    then
        python3 process_physcov.py --scenario highway_random --cores 16 --total_samples $1  --beam_count $totallines --distribution $2 --data_path $3
    fi

done