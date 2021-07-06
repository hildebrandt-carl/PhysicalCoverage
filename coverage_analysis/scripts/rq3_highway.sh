#!/bin/bash

# Used to compute the number of unique crashes, as well as do the random and greedy test selection

python3 count_unique_crashes.py --beam_count 3 --total_samples $1 --scenario highway --cores 111

echo
echo 
echo -------------------------------------------------------------
echo
echo

python3 test_selection_compute.py --beam_count 3 --total_samples $1 --scenario highway --cores 111
