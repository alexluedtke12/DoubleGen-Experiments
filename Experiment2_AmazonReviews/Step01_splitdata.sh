#!/bin/bash

# Step 01: Split Amazon Reviews data into four sets for analysis

# Source the log management functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/log_management.sh"

# Setup logging (this will clean old logs and start a new one)
LOG_FILE=$(setup_step_logging "Step01_splitData" "Step 01: Splitting Data into Four Sets")

# Set up environment
export JAVA_HOME="$SCRIPT_DIR/jdk-11.0.2"
export PATH="$JAVA_HOME/bin:$PATH"

echo "Activating Python environment..." | tee -a "$LOG_FILE"
source "$SCRIPT_DIR/reviews/bin/activate"

echo "Running data splitting script..." | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the splitting script with output to both console and log file
python "$SCRIPT_DIR/Step01_splitData/split_data_into_sets.py" 2>&1 | tee -a "$LOG_FILE"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# End the log with appropriate status
end_log "$LOG_FILE" $EXIT_CODE

# Check if the script ran successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================================="
    echo "Step 01 completed: Data successfully split into four sets"
    echo "Splits saved to: ~/data_directory/amazon_reviews_2023_splits/"
    echo "=================================================="
else
    echo "=================================================="
    echo "ERROR: Data splitting failed!"
    echo "Check the log file for details: $LOG_FILE"
    echo "=================================================="
    exit 1
fi