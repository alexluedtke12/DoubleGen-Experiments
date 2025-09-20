#!/bin/bash
# Step04: Format reviews with star rating prefix
# This script formats each review as "N stars: review text"

# Source the log management functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/log_management.sh"

# Setup logging (this will clean old logs and start a new one)
LOG_FILE=$(setup_step_logging "Step04_reformatReviews" "Step 04: Format Reviews with Star Prefix")

# Set up environment
export JAVA_HOME="$SCRIPT_DIR/jdk-11.0.2"
export PATH="$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$SCRIPT_DIR/reviews/bin/python"
export PYSPARK_DRIVER_PYTHON="$SCRIPT_DIR/reviews/bin/python"

# Activate virtual environment
source "$SCRIPT_DIR/reviews/bin/activate"

# Run the review formatting script
echo "Starting review formatting..." | tee -a "$LOG_FILE"
python "$SCRIPT_DIR/Step04_reformatReviews/add_star_prefix.py" 2>&1 | tee -a "$LOG_FILE"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# End the log with appropriate status
end_log "$LOG_FILE" $EXIT_CODE

if [ $EXIT_CODE -eq 0 ]; then
    echo "Step04 completed successfully!"
else
    echo "ERROR: Review formatting failed!"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi