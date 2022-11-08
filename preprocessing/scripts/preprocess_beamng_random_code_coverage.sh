#!/bin/bash

if [ $# -eq 1 ]
  then
    python3 preprocess_coverage.py --scenario beamng_random --total_samples $1 --cores 4
fi

if [ $# -eq 2 ]
  then
    python3 preprocess_coverage.py --scenario beamng_random --total_samples $1 --cores 4 --data_path $2
fi