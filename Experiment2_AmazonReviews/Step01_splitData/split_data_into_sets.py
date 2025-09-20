#!/usr/bin/env python3
"""
Split Amazon Reviews 2023 dataset into four sets for analysis.

PURPOSE:
This script creates four non-overlapping random subsets of the Amazon Reviews dataset
for use in causal inference and machine learning experiments. The splits enable
proper train/test separation and propensity score modeling.

SPLITS CREATED:
1. Propensity set (10%): Used to train a logistic regression model that predicts
   the probability of a review being a verified purchase based on product features.
   This propensity score can be used for:
   - Inverse propensity weighting (IPW)
   - Propensity score matching
   - Doubly robust estimation

2. Training set 1 (30%): First training dataset for model development
3. Training set 2 (30%): Second training dataset for model development
   (These can be used separately for different models or combined for larger training sets)

4. Test set (30%): Held-out test set for final unbiased model evaluation

SPLITTING METHOD:
- Each record is assigned a random number between 0 and 1 using PySpark's rand() function
- Records are assigned to sets based on the random number:
  * [0.0, 0.1): Propensity set (10%)
  * [0.1, 0.4): Training set 1 (30%)
  * [0.4, 0.7): Training set 2 (30%)
  * [0.7, 1.0]: Test set (30%)
- Fixed random seed (42) ensures reproducibility
- No stratification is performed - splits are uniformly random
- Each instance appears in exactly one set

INPUT:
- Reads from: ~/data_directory/amazon_reviews_2023_reduced/features.parquet/
- Expected size: ~571M records

OUTPUT:
- ~/data_directory/amazon_reviews_2023_splits/propensity_set/
- ~/data_directory/amazon_reviews_2023_splits/training_set_1/
- ~/data_directory/amazon_reviews_2023_splits/training_set_2/
- ~/data_directory/amazon_reviews_2023_splits/test_set/

RUNTIME: Approximately 5-7 minutes with 40 CPUs and 400GB RAM in local mode
"""

import os
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    """
    Main function that performs the data splitting.
    
    Process:
    1. Initialize Spark with optimized settings for large-scale data processing
    2. Read the full parquet dataset
    3. Add random numbers to each record for splitting
    4. Create four DataFrames based on random number ranges
    5. Write each split to its own parquet directory
    6. Verify counts and proportions
    7. Generate summary report
    """
    print(f"Starting data splitting at {datetime.now()}")
    
    # Initialize Spark session with local mode (40 CPUs, 400GB RAM)
    spark = SparkSession.builder \
        .appName("SplitDataIntoSets") \
        .config("spark.master", "local[40]") \
        .config("spark.driver.memory", "400g") \
        .config("spark.executor.memory", "400g") \
        .config("spark.driver.maxResultSize", "160g") \
        .config("spark.sql.shuffle.partitions", "160") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .getOrCreate()
    
    # Note on Spark configuration:
    # - local[40]: Use 40 CPU cores in local mode
    # - 400GB driver/executor memory (shared in local mode)
    # - 160GB max result size (40% of driver memory)
    # - 160 shuffle partitions (4x cores) for optimal parallelism
    # - Adaptive query execution optimizes performance dynamically
    # - Kryo serialization is faster than default Java serialization
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Read the parquet data
    print("Reading data from parquet files...")
    input_path = "~/data_directory/amazon_reviews_2023_reduced/features.parquet/"
    df = spark.read.parquet(input_path)
    
    # Get total count - this triggers a full scan but is necessary for verification
    print("Counting total records...")
    total_count = df.count()
    print(f"Total records: {total_count:,}")
    
    # Add random number for splitting
    print("\nAssigning random numbers for splitting...")
    # CRITICAL: Using a fixed seed (42) ensures reproducibility
    # Anyone running this script will get the exact same splits
    # The rand() function generates uniform random numbers in [0, 1)
    df_with_rand = df.withColumn("rand_num", F.rand(seed=42))
    
    # Define split boundaries
    # These boundaries create non-overlapping intervals that partition [0, 1):
    # Propensity: [0.0, 0.1) = 0.1 = 10%
    # Training 1: [0.1, 0.4) = 0.3 = 30%
    # Training 2: [0.4, 0.7) = 0.3 = 30%
    # Test:       [0.7, 1.0] = 0.3 = 30%
    # Total:                 = 1.0 = 100%
    
    print("\nCreating data splits...")
    
    # Propensity set (10%)
    print("Creating propensity set (10%)...")
    # Select records where rand_num < 0.1 (10% of uniform distribution)
    # drop("rand_num") removes the random column - we don't want it in output
    propensity_df = df_with_rand.filter(F.col("rand_num") < 0.1).drop("rand_num")
    
    # Training set 1 (30%)
    print("Creating training set 1 (30%)...")
    # Select records where 0.1 <= rand_num < 0.4 (30% of uniform distribution)
    training1_df = df_with_rand.filter(
        (F.col("rand_num") >= 0.1) & (F.col("rand_num") < 0.4)
    ).drop("rand_num")
    
    # Training set 2 (30%)
    print("Creating training set 2 (30%)...")
    # Select records where 0.4 <= rand_num < 0.7 (30% of uniform distribution)
    training2_df = df_with_rand.filter(
        (F.col("rand_num") >= 0.4) & (F.col("rand_num") < 0.7)
    ).drop("rand_num")
    
    # Test set (30%)
    print("Creating test set (30%)...")
    # Select records where rand_num >= 0.7 (30% of uniform distribution)
    # Note: This includes 1.0, though rand() technically never returns exactly 1.0
    test_df = df_with_rand.filter(F.col("rand_num") >= 0.7).drop("rand_num")
    
    # Define output paths
    output_base = "~/data_directory/amazon_reviews_2023_splits"
    propensity_path = f"{output_base}/propensity_set"
    training1_path = f"{output_base}/training_set_1"
    training2_path = f"{output_base}/training_set_2"
    test_path = f"{output_base}/test_set"
    
    # Write the splits to parquet files
    print("\nWriting splits to parquet files...")
    
    print(f"Writing propensity set to {propensity_path}...")
    start_time = time.time()
    propensity_df.write.mode("overwrite").parquet(propensity_path)
    propensity_time = time.time() - start_time
    propensity_count = spark.read.parquet(propensity_path).count()
    print(f"Propensity set written: {propensity_count:,} records in {propensity_time:.2f} seconds")
    
    print(f"\nWriting training set 1 to {training1_path}...")
    start_time = time.time()
    training1_df.write.mode("overwrite").parquet(training1_path)
    training1_time = time.time() - start_time
    training1_count = spark.read.parquet(training1_path).count()
    print(f"Training set 1 written: {training1_count:,} records in {training1_time:.2f} seconds")
    
    print(f"\nWriting training set 2 to {training2_path}...")
    start_time = time.time()
    training2_df.write.mode("overwrite").parquet(training2_path)
    training2_time = time.time() - start_time
    training2_count = spark.read.parquet(training2_path).count()
    print(f"Training set 2 written: {training2_count:,} records in {training2_time:.2f} seconds")
    
    print(f"\nWriting test set to {test_path}...")
    start_time = time.time()
    test_df.write.mode("overwrite").parquet(test_path)
    test_time = time.time() - start_time
    test_count = spark.read.parquet(test_path).count()
    print(f"Test set written: {test_count:,} records in {test_time:.2f} seconds")
    
    # Verify splits
    print("\n" + "="*80)
    print("SPLIT VERIFICATION")
    print("="*80)
    print(f"Original dataset: {total_count:,} records")
    print(f"Propensity set:   {propensity_count:,} records ({100*propensity_count/total_count:.2f}%)")
    print(f"Training set 1:   {training1_count:,} records ({100*training1_count/total_count:.2f}%)")
    print(f"Training set 2:   {training2_count:,} records ({100*training2_count/total_count:.2f}%)")
    print(f"Test set:         {test_count:,} records ({100*test_count/total_count:.2f}%)")
    print(f"Total in splits:  {propensity_count + training1_count + training2_count + test_count:,} records")
    print(f"Difference:       {total_count - (propensity_count + training1_count + training2_count + test_count)} records")
    
    # Save summary report
    report_path = "/home/your_username/cf_reviews/Step01_splitData/split_summary.txt"
    with open(report_path, 'w') as f:
        f.write("DATA SPLIT SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Split timestamp: {datetime.now()}\n")
        f.write(f"Input path: {input_path}\n")
        f.write(f"Output base: {output_base}\n\n")
        f.write("SPLIT STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Original dataset: {total_count:,} records\n\n")
        f.write(f"Propensity set:   {propensity_count:,} records ({100*propensity_count/total_count:.2f}%)\n")
        f.write(f"  Path: {propensity_path}\n")
        f.write(f"  Write time: {propensity_time:.2f} seconds\n\n")
        f.write(f"Training set 1:   {training1_count:,} records ({100*training1_count/total_count:.2f}%)\n")
        f.write(f"  Path: {training1_path}\n")
        f.write(f"  Write time: {training1_time:.2f} seconds\n\n")
        f.write(f"Training set 2:   {training2_count:,} records ({100*training2_count/total_count:.2f}%)\n")
        f.write(f"  Path: {training2_path}\n")
        f.write(f"  Write time: {training2_time:.2f} seconds\n\n")
        f.write(f"Test set:         {test_count:,} records ({100*test_count/total_count:.2f}%)\n")
        f.write(f"  Path: {test_path}\n")
        f.write(f"  Write time: {test_time:.2f} seconds\n\n")
        f.write(f"Total in splits:  {propensity_count + training1_count + training2_count + test_count:,} records\n")
        f.write(f"Verification:     All records accounted for (difference = {total_count - (propensity_count + training1_count + training2_count + test_count)})\n")
    
    print(f"\nSummary report saved to: {report_path}")
    
    # Stop Spark session
    spark.stop()
    print("\nData splitting complete!")

if __name__ == "__main__":
    main()