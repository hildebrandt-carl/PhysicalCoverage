#!/bin/bash

# Each of the beam counts
beamcount=(1 2 3 4 5 6 7 8 9 10)

# Run it for each of the total number of lines
for totallines in "${beamcount[@]}"
do
    # Run the script
    python3 preprocess_data.py --scenario beamng_generated --cores 120 --total_samples $1 --beam_count $totallines
done
