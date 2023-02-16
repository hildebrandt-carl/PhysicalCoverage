#!/bin/bash

if [ $# -eq 0 ]
  then
    python3 preprocess_feasibility.py --scenario beamng --cores 1 --distribution center_close
fi

if [ $# -eq 1 ]
  then
    python3 preprocess_feasibility.py --scenario beamng --cores 1 --distribution center_close --data_path $1
fi
