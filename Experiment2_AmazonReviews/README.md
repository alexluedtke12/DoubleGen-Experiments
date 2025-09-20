# Amazon Reviews Experiment

## Quick Start

Run the complete pipeline with these commands from the root directory:

```bash
# Step 0: Process raw Amazon reviews data
nohup ./Step00_makedata.sh > Step00_makeData/logs/step00.log 2>&1 &

# Step 1: Split data into train/test sets
nohup ./Step01_splitdata.sh > Step01_splitData/logs/step01.log 2>&1 &

# Step 2: Train logistic regression for propensity scores
nohup ./Step02_logisticregression.sh > Step02_logisticRegression/logs/step02.log 2>&1 &

# Step 3: Generate synthetic verified purchases
nohup ./Step03_makesynthetic.sh > Step03_makeSynthetic/logs/step03.log 2>&1 &

# Step 4: Format reviews with rating prefix
nohup ./Step04_reformatreviews.sh > Step04_reformatReviews/logs/step04.log 2>&1 &

# Step 5: Fit LightGBM for inverse propensity weights
nohup ./Step05_fitlightgbm.sh > Step05_fitLightGBM/logs/step05.log 2>&1 &

# Step 6: Compute AIPW weights using k-NN
nohup ./Step06_findneighbors_multi.sh > Step06_findNeighbors/logs/step06.log 2>&1 &

# Step 7: Tokenize reviews for model training
nohup ./Step07_tokenizereviews.sh > Step07_tokenizeReviews/logs/step07.log 2>&1 &

# Step 8: Finetune language models (GPU required)
sbatch Step08_gpu.sbatch

# Step 9: Compute evaluation metrics (GPU required)
sbatch Step09_gpu.sbatch
```

## Installation

1. Install Java JDK 11 in the current directory:
   ```bash
   wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz
   tar -xzf openjdk-11+28_linux-x64_bin.tar.gz
   mv jdk-11 jdk-11.0.2
   rm openjdk-11+28_linux-x64_bin.tar.gz
   ```

2. Create Python virtual environment:
   ```bash
   python3.9 -m venv reviews
   source reviews/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. For GPU support with FAISS, also install:
   ```bash
   pip install faiss-gpu
   ```

## Dataset

Download the Amazon Reviews 2023 dataset using Hugging Face CLI:
```bash
mkdir -p data_directory
huggingface-cli download --repo-type dataset \
                         --resume-download \
                         --local-dir data_directory/amazon_reviews_2023 \
                         --local-dir-use-symlinks False \
                         McAuley-Lab/Amazon-Reviews-2023
```

Note: This dataset is required for Step 0 and should be placed in the appropriate directory as specified in your configuration.
