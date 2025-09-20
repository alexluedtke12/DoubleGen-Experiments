#!/bin/bash

# Step 02: Logistic Regression on Amazon Reviews - Complete Pipeline
# This script:
# 1. Trains logistic regression to predict verified_purchase using baseline features
# 2. Adds predictions to all three split datasets

# Source the log management functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/log_management.sh"

# Setup logging (this will clean old logs and start a new one)
LOG_FILE=$(setup_step_logging "Step02_logisticRegression" "Step 02: Logistic Regression Analysis")

# Set up environment
export JAVA_HOME="$SCRIPT_DIR/jdk-11.0.2"
export PATH="$JAVA_HOME/bin:$PATH"
export SPARK_SUBMIT_OPTS="-Djava.io.tmpdir=~/data_directory/tmp"

# Activate virtual environment
source "$SCRIPT_DIR/reviews/bin/activate"

# Create temp directory if it doesn't exist
mkdir -p ~/data_directory/tmp

echo "Starting Step 02: Logistic Regression Analysis" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"
echo "This will:" | tee -a "$LOG_FILE"
echo "1. Train logistic regression on propensity set" | tee -a "$LOG_FILE"
echo "2. Add predictions to training_set_1, training_set_2, and test_set" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if model already exists
if [ -d "~/data_directory/logistic_regression_model" ]; then
    echo "Model already exists. Skipping training phase." | tee -a "$LOG_FILE"
else
    echo "Phase 1: Training model on propensity set..." | tee -a "$LOG_FILE"
    # Run training without timeout
    python "$SCRIPT_DIR/Step02_logisticRegression/logistic_regression_propensity_simple.py" 2>&1 | tee -a "$LOG_FILE"
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Model training failed!" | tee -a "$LOG_FILE"
        end_log "$LOG_FILE" $TRAIN_EXIT_CODE
        exit 1
    fi
fi

echo "" | tee -a "$LOG_FILE"
echo "Phase 2: Adding predictions to all split datasets..." | tee -a "$LOG_FILE"
# Run predictions without timeout
python "$SCRIPT_DIR/Step02_logisticRegression/add_predictions_to_splits.py" 2>&1 | tee -a "$LOG_FILE"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# End the log with appropriate status
end_log "$LOG_FILE" $EXIT_CODE

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Step 02 COMPLETED SUCCESSFULLY!"
    echo "=============================================="
    echo "Results:"
    echo "- Model coefficients: $SCRIPT_DIR/Step02_logisticRegression/logistic_regression_propensity_results.txt"
    echo "- Predictions added to:"
    echo "  - ~/data_directory/amazon_reviews_2023_splits/training_set_1/"
    echo "  - ~/data_directory/amazon_reviews_2023_splits/training_set_2/"
    echo "  - ~/data_directory/amazon_reviews_2023_splits/test_set/"
    echo "- Full log: $LOG_FILE"
else
    echo ""
    echo "ERROR: Prediction generation failed!"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi