#!/bin/bash

# Used to generate unseen scenarios
python3 generate_unseen_tests_single.py --total_samples $1 --scenario highway --cores 100 