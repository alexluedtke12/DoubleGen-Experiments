#!/bin/bash

# Amazon Reviews 2023 Processing Script
# =====================================
# Purpose: Automate the processing of Amazon Reviews 2023 dataset into ML-ready features
# 
# This script:
# 1. Validates all prerequisites (Java, Python env, source data)
# 2. Sets up the PySpark environment with proper memory allocation
# 3. Runs the feature engineering pipeline
# 4. Creates features for LightGBM modeling (verified purchase prediction)
#
# Requirements:
# - Java 11 (OpenJDK) in jdk-11.0.2/
# - Python virtual environment in reviews/ with PySpark 3.5.0
# - Source data at ~/data_directory/amazon_reviews_2023/
# - At least 64GB RAM and 16 CPU cores
#
# Output: Parquet files at ~/data_directory/amazon_reviews_2023_reduced/
# Expected runtime: ~2 hours

set -e  # Exit immediately if any command fails

# Source the log management functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/log_management.sh"

echo "========================================"
echo "Amazon Reviews 2023 Processing Pipeline"
echo "========================================"
echo ""

# Configuration Variables
# =======================
# Get the directory where this script is located (handles symlinks correctly)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Java 11 installation path (required for PySpark)
JAVA_HOME="$SCRIPT_DIR/jdk-11.0.2"

# Python virtual environment containing PySpark and dependencies
PYTHON_ENV="$SCRIPT_DIR/reviews"

# Main processing script that performs feature engineering
PROCESSING_SCRIPT="$SCRIPT_DIR/Step00_makeData/process_reviews_to_parquet_enhanced.py"

# Output directory where processed parquet files will be saved
OUTPUT_DIR="~/data_directory/amazon_reviews_2023_reduced"

# Terminal Color Codes for Better Readability
# ============================================
RED='\033[0;31m'     # Red for errors
GREEN='\033[0;32m'   # Green for success messages
YELLOW='\033[1;33m'  # Yellow for warnings
NC='\033[0m'         # No Color (reset to default)

# Helper Functions for Formatted Output
# =====================================
# Print success/info messages with timestamp
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Print error messages in red
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print warning messages in yellow
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Prerequisites Validation
# ========================
# This section ensures all required components are present before starting
# the processing job to avoid failures mid-execution

print_status "Checking prerequisites..."

# Verify Java 11 Installation
# Java 11 is required by PySpark 3.5.0 for compatibility
if [ ! -d "$JAVA_HOME" ]; then
    print_error "Java 11 not found at $JAVA_HOME"
    print_error "Please ensure jdk-11.0.2 directory exists"
    exit 1
fi

# Check if Python environment exists
if [ ! -d "$PYTHON_ENV" ]; then
    print_error "Python environment not found at $PYTHON_ENV"
    print_error "Please ensure reviews/ directory exists"
    exit 1
fi

# Check if processing script exists
if [ ! -f "$PROCESSING_SCRIPT" ]; then
    print_error "Processing script not found at $PROCESSING_SCRIPT"
    exit 1
fi

# Check if source data exists
SOURCE_DATA="~/data_directory/amazon_reviews_2023"
if [ ! -d "$SOURCE_DATA" ]; then
    print_error "Source data not found at $SOURCE_DATA"
    print_error "Please ensure Amazon Reviews 2023 dataset is available"
    exit 1
fi

print_status "All prerequisites checked ✓"
echo ""

# Setup environment
print_status "Setting up environment..."
export JAVA_HOME="$JAVA_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

# Verify Java version
java_version=$(java -version 2>&1 | head -n 1)
print_status "Java version: $java_version"

# Activate Python environment
source "$PYTHON_ENV/bin/activate"
python_version=$(python --version)
print_status "Python version: $python_version"
echo ""

# Check if output directory already exists
if [ -d "$OUTPUT_DIR" ]; then
    print_warning "Output directory already exists at $OUTPUT_DIR"
    print_status "Removing existing output directory..."
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
print_status "Creating output directory..."
mkdir -p "$OUTPUT_DIR"

# Display processing information
echo ""
echo "========================================"
echo "Processing Configuration:"
echo "========================================"
echo "Source data: $SOURCE_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Number of CPUs: 40"
echo "Driver memory: 400GB"
echo ""

# Automatically proceed with processing
print_status "Starting processing automatically..."

# Start processing
print_status "Starting PySpark processing job..."
echo ""

# Setup logging (this will clean old logs and start a new one)
LOG_FILE=$(setup_step_logging "Step00_makeData" "Amazon Reviews 2023 Processing Pipeline")
print_status "Logging to: $LOG_FILE"
echo ""

# Run the processing script
python "$PROCESSING_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# End the log with appropriate status
end_log "$LOG_FILE" $EXIT_CODE

# Check if processing completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    print_status "Processing completed successfully! ✓"
    
    # Display output statistics
    if [ -d "$OUTPUT_DIR/features.parquet" ]; then
        echo ""
        echo "========================================"
        echo "Output Statistics:"
        echo "========================================"
        num_files=$(ls -1 "$OUTPUT_DIR/features.parquet/"*.parquet 2>/dev/null | wc -l)
        total_size=$(du -sh "$OUTPUT_DIR/features.parquet" | cut -f1)
        echo "Number of parquet files: $num_files"
        echo "Total size: $total_size"
        echo "Output location: $OUTPUT_DIR/features.parquet/"
    fi
else
    print_error "Processing failed! Check the log file: $LOG_FILE"
    exit 1
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Your processed data is ready at: $OUTPUT_DIR/features.parquet/"
echo "2. See README.md for usage examples with PySpark and Pandas"
echo "3. The data contains 571M+ reviews with engineered features"
echo "4. Use the features for LightGBM modeling to predict verified_purchase"
echo ""
print_status "Done!"