#!/bin/bash

# Step 06: Compute AIPW Weights with Multiple Variants
# This script computes 6 different weight columns:
# - gcomp_right: Standard k-NN based G-computation weight
# - gcomp_wrong: Uniform weighting across all eligible matches
# - aipw_ipw_right_out_right: ipw_right + augmentation from k-NN matching
# - aipw_ipw_right_out_wrong: ipw_right + augmentation from uniform weighting
# - aipw_ipw_wrong_out_right: ipw_wrong + augmentation from k-NN matching
# - aipw_ipw_wrong_out_wrong: ipw_wrong + augmentation from uniform weighting

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/log_management.sh"

# Setup logging
LOG_FILE=$(setup_step_logging "Step06_findNeighbors_multi" "Step 06: Find Nearest Neighbors with Multiple Weights")

# Check if log management was loaded
if [ -z "$LOG_FILE" ]; then
    echo "Error: Failed to setup logging. Exiting."
    exit 1
fi

echo "Starting Step 06: Compute AIPW Weights with Multiple Variants"
echo "This will process both training sets with cross-fitting design"
echo "Output will include 6 weight columns per dataset (2 gcomp + 4 AIPW)"

# Set up Python environment
cd "$SCRIPT_DIR"
source reviews/bin/activate
export JAVA_HOME=$SCRIPT_DIR/jdk-11.0.2
export PATH=$JAVA_HOME/bin:$PATH

# Process train1 with matches from train2
echo ""
echo "Processing training_set_1 with matches from training_set_2..."
echo "--------------------------------------------------------------"
python Step06_findNeighbors/find_nearest_neighbors_multi_weights.py \
    --training_set train1 \
    --matching_set train2 \
    --k 200 \
    --memory 400g \
    --cores 40 \
    --gpus 4

EXIT_CODE_1=$?

if [ $EXIT_CODE_1 -ne 0 ]; then
    echo "Error: Failed to process train1->train2 (exit code: $EXIT_CODE_1)"
    end_log "$LOG_FILE" $EXIT_CODE_1
    exit $EXIT_CODE_1
fi

# Process train2 with matches from train1
echo ""
echo "Processing training_set_2 with matches from training_set_1..."
echo "--------------------------------------------------------------"
python Step06_findNeighbors/find_nearest_neighbors_multi_weights.py \
    --training_set train2 \
    --matching_set train1 \
    --k 200 \
    --memory 400g \
    --cores 40 \
    --gpus 4

EXIT_CODE_2=$?

if [ $EXIT_CODE_2 -ne 0 ]; then
    echo "Error: Failed to process train2->train1 (exit code: $EXIT_CODE_2)"
    end_log "$LOG_FILE" $EXIT_CODE_2
    exit $EXIT_CODE_2
fi

echo ""
echo "Step 06 completed successfully!"
echo "Output saved to: ~/data_directory/amazon_reviews_2023_augmented_multi/"
echo "Each dataset now has 6 weight columns:"
echo "  - gcomp_right, gcomp_wrong"
echo "  - aipw_ipw_right_out_right, aipw_ipw_right_out_wrong"
echo "  - aipw_ipw_wrong_out_right, aipw_ipw_wrong_out_wrong"

# End logging
end_log "$LOG_FILE" 0