#!/bin/bash

# Exit on any error
set -e

echo "Starting nuisance fitting process..."
# python3 src/fit_nuisances.py

# Array of weight types to iterate through
WEIGHT_TYPES=(
    "aipw_rightProp_rightOut"
)

# Training parameters
NUM_EPOCHS=100
LR_DECAY_EVERY=250
# run more epochs for aipw_rightProp_rightOut - extra epochs only used for displayed images in Fig. 1. The performance comparisons are apples-to-apples, comparing the performance of all methods at the same number of epochs (run_nuisance_experiment.sh)
NUM_EPOCHS_AIPW_RIGHT=2000

echo "Starting training processes..."
for weight_type in "${WEIGHT_TYPES[@]}"; do
    echo "Training with weight type: $weight_type"
    
    accelerate launch --multi_gpu --num_processes 3 --num_cpu_threads_per_process 3 --num_machines 1 --mixed_precision fp16 src/train.py \
        --which_wgt "$weight_type" \
        --num_epochs $NUM_EPOCHS_AIPW_RIGHT \
        --lr_decay_every $LR_DECAY_EVERY \
        --save_individual_images_at $NUM_EPOCHS \
        --large_net 1
    
    echo "Completed training for weight type: $weight_type"
    echo "----------------------------------------"
done

echo "All training processes completed successfully!"