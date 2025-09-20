#!/bin/bash

# Step05: Fit LightGBM models with inverse propensity weights (RIGHT and WRONG)
# Purpose: Train separate LightGBM models on train1 and train2 to predict verified_purchase_synthetic
#          Creates both ipw_right (all features) and ipw_wrong (no category features)

# Source the log management functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/log_management.sh"

# Setup logging (this will clean old logs and start a new one)
LOG_FILE=$(setup_step_logging "Step05_fitLightGBM" "Step 05: Fit LightGBM with Inverse Propensity (RIGHT and WRONG)")

# Set up environment
# Activate the reviews virtual environment
source "$SCRIPT_DIR/reviews/bin/activate"

# Set Java environment for PySpark
export JAVA_HOME="$SCRIPT_DIR/jdk-11.0.2"
export PATH="$JAVA_HOME/bin:$PATH"

# Set Spark environment
export SPARK_HOME=$VIRTUAL_ENV/lib/python3.9/site-packages/pyspark
export PYSPARK_PYTHON=$VIRTUAL_ENV/bin/python
export PYSPARK_DRIVER_PYTHON=$VIRTUAL_ENV/bin/python

# Run the inverse propensity fitting script for both RIGHT and WRONG models
echo "Running inverse propensity LightGBM fitting (RIGHT and WRONG models)..." | tee -a "$LOG_FILE"
python "$SCRIPT_DIR/Step05_fitLightGBM/fit_inverse_propensity_both_models.py" 2>&1 | tee -a "$LOG_FILE"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# End the log with appropriate status
end_log "$LOG_FILE" $EXIT_CODE

# Check if the script ran successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo "Step 05 completed successfully!"
else
    echo "Step 05 failed. Check the logs for details: $LOG_FILE"
    exit 1
fi