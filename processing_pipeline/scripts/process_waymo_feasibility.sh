#!/bin/bash

if [ $# -eq 0 ]
  then
    python3 process_feasibility.py --scenario waymo --cores 1 --distribution center_full
fi

if [ $# -eq 1 ]
  then
    python3 process_feasibility.py --scenario waymo --cores 1 --distribution center_full --data_path $1
fi
