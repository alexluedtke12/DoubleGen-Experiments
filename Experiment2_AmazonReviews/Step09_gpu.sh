#!/bin/bash

# Step 09: Compute Metrics for Generated Samples
# This script evaluates all generated models using perplexity, MAUVE, and Wasserstein distance
# Production configuration: 100K test samples, all 10K generated samples

echo "========================================="
echo "Step 09: Compute Metrics for Generated Samples"
echo "========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# Load environment
source /home/your_username/cf_reviews/reviews/bin/activate
export JAVA_HOME=/home/your_username/cf_reviews/jdk-11.0.2
export PATH=$JAVA_HOME/bin:$PATH

# Install/upgrade required packages
echo "Checking/installing required packages..."
pip install --quiet --upgrade mauve-text evaluate peft scipy

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Change to project directory
cd /home/your_username/cf_reviews

# Run metrics computation
echo "Running metrics computation..."
echo "Configuration: 100K test samples, all 10K generated samples"
python Step09_metrics/compute_metrics.py

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Metrics computation completed successfully!"
    echo "Results saved to: Step09_metrics/metrics_results/"
else
    echo "❌ Metrics computation failed with exit code $EXIT_CODE"
fi
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

exit $EXIT_CODE