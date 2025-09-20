#!/bin/bash

# Exit on any error
set -e

echo "Starting nuisance fitting process..."
python3 src/train/fit_nuisances.py

# Array of weight types to iterate through
WEIGHT_TYPES=(
    "aipw_rightProp_rightOut"
    "no_wgt"
    "ipw_right"
    "gcomp_right"
    "aipw_rightProp_wrongOut"
    "aipw_wrongProp_rightOut"
    "ipw_wrong"
    "aipw_wrongProp_wrongOut"
    "gcomp_wrong"
)


# Training parameters
NUM_EPOCHS=500
LR_DECAY_EVERY=250

echo "Starting training processes..."
for weight_type in "${WEIGHT_TYPES[@]}"; do
    echo "Training with weight type: $weight_type"
    accelerate launch --multi_gpu --num_processes 3 --num_cpu_threads_per_process 3 --num_machines 1 --mixed_precision fp16 src/train/train.py \
        --which_wgt "$weight_type" \
        --num_epochs $NUM_EPOCHS \
        --lr_decay_every $LR_DECAY_EVERY
    
    echo "Completed training for weight type: $weight_type"
    echo "----------------------------------------"
done

echo "All training processes completed successfully!"