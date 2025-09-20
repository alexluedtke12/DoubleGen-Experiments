#!/bin/bash

# Step03: Make Synthetic Data
# This script creates synthetic verified purchase data by:
# 1. Creating a synthetic verified purchase variable based on propensity scores
# 2. For verified purchases that become synthetic non-verified, replacing their review characteristics
# 3. Removing all actual non-verified purchases

# Source the log management functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/log_management.sh"

# Setup logging (this will clean old logs and start a new one)
LOG_FILE=$(setup_step_logging "Step03_makeSynthetic" "Step 03: Generate Synthetic Data")

# Set up environment
export JAVA_HOME="$SCRIPT_DIR/jdk-11.0.2"
export PATH="$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$SCRIPT_DIR/reviews/bin/python"
export PYSPARK_DRIVER_PYTHON="$SCRIPT_DIR/reviews/bin/python"

# Activate virtual environment
source "$SCRIPT_DIR/reviews/bin/activate"

# Run the synthetic data generation script
echo "Starting synthetic data generation..." | tee -a "$LOG_FILE"
python "$SCRIPT_DIR/Step03_makeSynthetic/make_synthetic_data.py" 2>&1 | tee -a "$LOG_FILE"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# End the log with appropriate status
end_log "$LOG_FILE" $EXIT_CODE

if [ $EXIT_CODE -eq 0 ]; then
    echo "Synthetic data generation completed successfully!"
else
    echo "ERROR: Synthetic data generation failed!"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi