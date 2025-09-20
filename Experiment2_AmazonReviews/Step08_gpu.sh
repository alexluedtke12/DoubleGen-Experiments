#!/bin/bash

# Load environment
source /home/your_username/cf_reviews/reviews/bin/activate
export JAVA_HOME=/home/your_username/cf_reviews/jdk-11.0.2
export PATH=$JAVA_HOME/bin:$PATH

# Load HuggingFace token
export HF_TOKEN=$(cat ~/.hf_token)

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Change to script directory
cd /home/your_username/cf_reviews/Step08_finetuneModel

# Initialize queue with all models (overwrites if exists)
echo "Initializing model queue..."
cat > models_to_train.txt << EOF
naive
aipw_ipw_right_out_right
ipw_right
gcomp_right
aipw_ipw_right_out_wrong
aipw_ipw_wrong_out_right
ipw_wrong
gcomp_wrong
aipw_ipw_wrong_out_wrong
EOF

echo "Queue initialized with 9 models"
echo "Starting training loop..."
echo "========================================="

# Process models until queue is empty
while true; do
    echo ""
    echo "Checking queue at $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run wrapper to process next model
    python train_single_model_wrapper.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 1 ]; then
        echo "Queue empty - all models processed!"
        break
    elif [ $EXIT_CODE -eq 2 ]; then
        echo "Model training failed - continuing with next model"
    else
        echo "Model completed successfully"
    fi
    
    # Small delay between models
    sleep 5
done

echo ""
echo "========================================="
echo "All models have been processed"
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Models completed:"
if [ -f models_completed.txt ]; then
    cat models_completed.txt
else
    echo "(no completions logged)"
fi