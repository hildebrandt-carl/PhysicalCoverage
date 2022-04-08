#!/bin/bash

python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution linear
python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution center_mid
python3 preprocess_feasibility.py --scenario highway --cores 1 --distribution center_close