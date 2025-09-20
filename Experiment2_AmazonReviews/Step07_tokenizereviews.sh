#!/bin/bash

# Step07: Tokenize reviews for model training

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Source log management functions
source "$SCRIPT_DIR/log_management.sh"

# Setup logging
LOG_FILE=$(setup_step_logging "Step07_tokenizeReviews" "Step 07: Tokenize Reviews for Model Training")

# Activate virtual environment
source "$SCRIPT_DIR/reviews/bin/activate"

# Set environment variables
export JAVA_HOME="$SCRIPT_DIR/jdk-11.0.2"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Set HuggingFace cache directory
export HF_HOME="~/data_directory/huggingface_cache"
export TRANSFORMERS_CACHE="~/data_directory/huggingface_cache"

# Load HuggingFace token from secure location
# The token should be stored in a persistent location on the cluster
# Priority order:
# 1. /home/your_username/.hf_token (persistent home directory)
# 2. /n/home/your_username/.hf_token (alternative home path if different)
# 3. /home/your_username/.config/huggingface/token (standard HF location)

if [ -f "/home/your_username/.hf_token" ]; then
    export HF_TOKEN=$(cat "/home/your_username/.hf_token")
    echo "HF_TOKEN loaded from /home/your_username/.hf_token" | tee -a "$LOG_FILE"
elif [ -f "/n/home/your_username/.hf_token" ]; then
    export HF_TOKEN=$(cat "/n/home/your_username/.hf_token")
    echo "HF_TOKEN loaded from /n/home/your_username/.hf_token" | tee -a "$LOG_FILE"
elif [ -f "/home/your_username/.config/huggingface/token" ]; then
    export HF_TOKEN=$(cat "/home/your_username/.config/huggingface/token")
    echo "HF_TOKEN loaded from /home/your_username/.config/huggingface/token" | tee -a "$LOG_FILE"
else
    echo "ERROR: HuggingFace token not found!" | tee -a "$LOG_FILE"
    echo "Please create one of the following files:" | tee -a "$LOG_FILE"
    echo "  1. /home/your_username/.hf_token (recommended - just the token on a single line)" | tee -a "$LOG_FILE"
    echo "  2. /home/your_username/.config/huggingface/token (HF CLI standard location)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "To create the token file, run:" | tee -a "$LOG_FILE"
    echo "  echo 'your_hf_token_here' > /home/your_username/.hf_token" | tee -a "$LOG_FILE"
    echo "  chmod 600 /home/your_username/.hf_token  # Make it readable only by you" | tee -a "$LOG_FILE"
    exit 1
fi

# Verify token is not empty
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is empty!" | tee -a "$LOG_FILE"
    exit 1
fi

# Log system information
echo "System Information:" | tee -a "$LOG_FILE"
echo "Python version: $(python --version)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the tokenization script
echo "Starting review tokenization..." | tee -a "$LOG_FILE"
echo "This will tokenize reviews once for all 8 weight variant models" | tee -a "$LOG_FILE"
echo "MAX_LENGTH: 192 tokens (covers 96-97% of reviews)" | tee -a "$LOG_FILE"

python -u "$SCRIPT_DIR/Step07_tokenizeReviews/tokenize_reviews.py" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Step07 completed successfully!" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "Step07 failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi

# End logging
end_log "$LOG_FILE" $EXIT_CODE

exit $EXIT_CODE