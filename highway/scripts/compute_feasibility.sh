#!/bin/bash

# Each of the beam counts
beamcount=(1 2 3 4 5 10)

# Run it for each of the total number of lines
for totallines in "${beamcount[@]}"
do
    # Run the script
    python3 determine_feasible_r.py --no_plot  --total_lines $totallines &
done
