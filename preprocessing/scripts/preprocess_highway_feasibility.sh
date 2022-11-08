#!/bin/bash

if [ $# -eq 0 ]
  then
    python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution linear 
    python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution center_mid
    python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution center_close
fi

if [ $# -eq 1 ]
  then
    python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution linear --data_path $1
    python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution center_mid --data_path $1
    python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution center_close --data_path $1
fi
