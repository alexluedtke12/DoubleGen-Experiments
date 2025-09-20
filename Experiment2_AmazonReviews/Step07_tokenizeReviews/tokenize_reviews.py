#!/usr/bin/env python3
"""
Tokenize reviews for all training sets to avoid redundant tokenization.

This script:
1. Loads training data with all weight columns
2. Filters to synthetic verified purchases only
3. Tokenizes reviews once using Llama-3.2-1B tokenizer
4. Saves tokenized data in HuggingFace Dataset format

This eliminates the need to tokenize 8 times (once per weight variant).
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer
import json
import gc

# Configuration
# Get HF token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set!")
    print("Please set it using: export HF_TOKEN='your_token_here'")
    print("Or load it from a secure file before running this script.")
    sys.exit(1)

MODEL_ID = "meta-llama/Llama-3.2-1B"
MAX_LENGTH = 192  # Updated from 128 to cover 96-97% of reviews

# Paths
BASE_DIR = Path("/n/scratch/users/a/your_username")
AIPW_DIR = BASE_DIR / "amazon_reviews_2023_augmented_multi"
OUTPUT_DIR = BASE_DIR / "tokenized_reviews"
CACHE_DIR = BASE_DIR / "huggingface_cache"

# All weight columns to preserve
WEIGHT_COLUMNS = [
    "ipw_right", "ipw_wrong",
    "gcomp_right", "gcomp_wrong",
    "aipw_ipw_right_out_right", "aipw_ipw_right_out_wrong",
    "aipw_ipw_wrong_out_right", "aipw_ipw_wrong_out_wrong"
]


def load_training_data(split: str) -> pd.DataFrame:
    """Load training data with all weight columns."""
    if split == "1":
        data_path = AIPW_DIR / "train1_with_aug_from_train2"
    elif split == "2":
        data_path = AIPW_DIR / "train2_with_aug_from_train1"
    else:
        raise ValueError(f"Invalid split: {split}. Must be '1' or '2'")
    
    print(f"\nLoading training set {split} from {data_path}")
    
    # Read all parquet files
    parquet_files = sorted(list(data_path.glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Read and combine
    dfs = []
    for i, pf in enumerate(parquet_files, 1):
        print(f"  Reading file {i}/{len(parquet_files)}: {pf.name}")
        df = pd.read_parquet(pf)
        dfs.append(df)
        
        # Memory management
        if i % 10 == 0:
            gc.collect()
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data):,} total records")
    
    # Check for required columns
    missing_cols = []
    for col in WEIGHT_COLUMNS + ['verified_purchase_synthetic', 'title', 'text', 'rating']:
        if col not in data.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return data


def filter_and_prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter to synthetic verified purchases and prepare for tokenization."""
    print(f"\nFiltering data...")
    print(f"  Initial records: {len(data):,}")
    
    # Filter to synthetic verified purchases only
    data = data[data['verified_purchase_synthetic'] == 1].copy()
    print(f"  After filtering to synthetic verified purchases: {len(data):,}")
    
    # Drop any records with missing review text
    initial_len = len(data)
    data = data.dropna(subset=['title', 'text', 'rating'])
    if len(data) < initial_len:
        print(f"  Dropped {initial_len - len(data):,} records with missing text fields")
    
    # Keep only necessary columns
    columns_to_keep = ['title', 'text', 'rating'] + WEIGHT_COLUMNS
    data = data[columns_to_keep]
    
    # Show weight statistics
    print(f"\nWeight column statistics:")
    for col in WEIGHT_COLUMNS:
        if col in data.columns:
            non_zero = (data[col] > 0).sum()
            mean_val = data[col].mean()
            print(f"  {col}: {non_zero:,} non-zero ({100*non_zero/len(data):.1f}%), mean={mean_val:.4f}")
    
    return data


def tokenize_data(data: pd.DataFrame, tokenizer, split: str) -> Dataset:
    """Tokenize reviews and create HuggingFace Dataset."""
    print(f"\nTokenizing {len(data):,} reviews...")
    
    # Use text as-is from Step04 (already formatted as "N stars: review text")
    print("  Using pre-formatted text from Step04...")
    start_time = time.time()
    texts = data['text'].tolist()  # Text is already in "N stars: review text" format
    print(f"  Text extraction took {time.time() - start_time:.1f} seconds")
    
    # Tokenize in batches for efficiency
    print(f"  Tokenizing with MAX_LENGTH={MAX_LENGTH}...")
    start_time = time.time()
    
    batch_size = 1000
    all_encodings = {'input_ids': [], 'attention_mask': []}
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        all_encodings['input_ids'].extend(batch_encodings['input_ids'])
        all_encodings['attention_mask'].extend(batch_encodings['attention_mask'])
        
        if (i + batch_size) % 10000 == 0:
            print(f"    Tokenized {min(i + batch_size, len(texts)):,}/{len(texts):,} reviews...")
    
    tokenization_time = time.time() - start_time
    print(f"  Tokenization took {tokenization_time:.1f} seconds ({len(texts)/tokenization_time:.0f} reviews/sec)")
    
    # Create dataset dictionary with tokenized data and weights
    print("  Creating HuggingFace Dataset...")
    dataset_dict = {
        'input_ids': all_encodings['input_ids'],
        'attention_mask': all_encodings['attention_mask'],
    }
    
    # Add all weight columns
    for col in WEIGHT_COLUMNS:
        dataset_dict[col] = data[col].tolist()
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    print(f"  Created dataset with {len(dataset):,} samples")
    
    # Show sample token lengths
    sample_lengths = [len([t for t in ids if t != tokenizer.pad_token_id]) 
                     for ids in all_encodings['input_ids'][:1000]]
    if sample_lengths:
        import numpy as np
        print(f"\n  Token length statistics (first 1000 samples):")
        print(f"    Mean: {np.mean(sample_lengths):.1f}")
        print(f"    Median: {np.median(sample_lengths):.1f}")
        print(f"    Max: {np.max(sample_lengths)}")
        print(f"    95th percentile: {np.percentile(sample_lengths, 95):.1f}")
    
    return dataset


def save_dataset(dataset: Dataset, split: str):
    """Save tokenized dataset to disk."""
    output_path = OUTPUT_DIR / f"train{split}_tokenized"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving tokenized dataset to {output_path}")
    start_time = time.time()
    
    # Save in HuggingFace format (memory-mapped for efficiency)
    dataset.save_to_disk(str(output_path))
    
    save_time = time.time() - start_time
    print(f"  Saved in {save_time:.1f} seconds")
    
    # Save metadata
    metadata = {
        "model_id": MODEL_ID,
        "max_length": MAX_LENGTH,
        "num_samples": len(dataset),
        "weight_columns": WEIGHT_COLUMNS,
        "tokenization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": split
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata to {metadata_path}")
    
    return output_path


def main():
    """Main tokenization pipeline."""
    print("="*80)
    print("TOKENIZING REVIEWS FOR MODEL TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_ID}")
    print(f"Max length: {MAX_LENGTH} tokens")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer once
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Pad token: {tokenizer.pad_token}")
    
    # Process both training splits
    overall_start = time.time()
    results = {}
    
    for split in ["1", "2"]:
        print(f"\n{'='*60}")
        print(f"PROCESSING TRAINING SET {split}")
        print('='*60)
        
        split_start = time.time()
        
        try:
            # Load data
            data = load_training_data(split)
            
            # Filter and prepare
            data = filter_and_prepare_data(data)
            
            # Tokenize
            dataset = tokenize_data(data, tokenizer, split)
            
            # Save
            output_path = save_dataset(dataset, split)
            
            split_time = time.time() - split_start
            results[f"train{split}"] = {
                "status": "success",
                "num_samples": len(dataset),
                "duration_seconds": split_time,
                "output_path": str(output_path)
            }
            
            print(f"\n✓ Training set {split} completed in {split_time/60:.1f} minutes")
            
            # Clean up memory
            del data
            del dataset
            gc.collect()
            
        except Exception as e:
            print(f"\n✗ ERROR processing training set {split}: {str(e)}")
            results[f"train{split}"] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Save overall summary
    overall_time = time.time() - overall_start
    summary = {
        "tokenization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": MODEL_ID,
        "max_length": MAX_LENGTH,
        "total_duration_minutes": overall_time / 60,
        "results": results
    }
    
    summary_path = OUTPUT_DIR / "tokenization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("TOKENIZATION COMPLETE")
    print("="*80)
    print(f"Total duration: {overall_time/60:.1f} minutes")
    print(f"Summary saved to: {summary_path}")
    
    # Print results
    for split, result in results.items():
        if result["status"] == "success":
            print(f"  {split}: ✓ {result['num_samples']:,} samples")
        else:
            print(f"  {split}: ✗ Failed")


if __name__ == "__main__":
    main()