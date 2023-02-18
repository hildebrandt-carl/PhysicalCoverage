#!/bin/bash

if [ $# -eq 1 ]
  then
    python3 process_coverage.py --scenario highway_random --total_samples $1 --cores 16
fi

if [ $# -eq 2 ]
  then
    python3 process_coverage.py --scenario highway_random --total_samples $1 --cores 16 --data_path $2
fi
