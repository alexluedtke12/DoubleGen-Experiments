#!/bin/bash

# Exit on any error
set -e

echo "Starting nuisance fitting process..."
python3 src/eval/preprocess.py

echo "Saving (estimated) counterfactual images..."
python3 src/eval/save_test_cfs.py

echo "Evaluating fid..."
python3 src/eval/fid.py