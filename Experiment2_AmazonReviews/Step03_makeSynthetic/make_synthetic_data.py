#!/usr/bin/env python3
"""
Step03: Generate Synthetic Verified Purchase Data

This script:
1. Loads the three splits (train1, train2, test) from Step02
2. For training sets (train1, train2):
   - Creates verified_purchase_synthetic based on transformed propensity scores
   - Transformation: Scales non-intercept coefficients by 2x in logit space
   - Adjusts intercept to control synthetic verified purchase rate
   - Formula: p_adjusted = expit(-4.5 + 2*logit(p_original))
   - For verified purchases that become synthetic non-verified, replaces their review characteristics (title, text, rating)
   - Removes all actual non-verified purchases
3. For test set:
   - Simply filters to keep only verified purchases (no synthetic generation)
   - Preserves original review characteristics
4. Saves the processed datasets
"""

import os
import sys
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, rand, when, lit, count, mean, stddev,
    min as spark_min, max as spark_max
)
from pyspark.sql.types import StringType, IntegerType, FloatType

def create_spark_session():
    """Create optimized Spark session with local mode (40 CPUs, 400GB RAM)
    
    Configuration for single machine with 40 cores and 400GB RAM:
    - Local mode with 40 CPU cores
    - 400GB memory (shared between driver and executors in local mode)
    - Shuffle partitions set to 4x cores (160)
    - Adaptive query execution for optimal performance
    """
    return SparkSession.builder \
        .appName("Step03_MakeSyntheticData") \
        .config("spark.master", "local[40]") \
        .config("spark.driver.memory", "400g") \
        .config("spark.executor.memory", "400g") \
        .config("spark.driver.maxResultSize", "160g") \
        .config("spark.sql.shuffle.partitions", "160") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()

def process_split(spark, split_name, input_path, output_path, seed=42):
    """Process a single split to create synthetic data (or filter for test set)"""
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Read the split data
    print(f"Reading data from {input_path}")
    df = spark.read.parquet(input_path)
    
    # Cache the dataframe for multiple operations
    df = df.cache()
    df.count()  # Trigger cache
    
    # Print initial statistics
    total_count = df.count()
    verified_count = df.filter(col("verified_purchase") == 1).count()
    non_verified_count = df.filter(col("verified_purchase") == 0).count()
    
    print(f"\nInitial statistics:")
    print(f"Total records: {total_count:,}")
    print(f"Verified purchases: {verified_count:,} ({verified_count/total_count*100:.2f}%)")
    print(f"Non-verified purchases: {non_verified_count:,} ({non_verified_count/total_count*100:.2f}%)")
    
    # Special handling for test set - just filter to verified purchases
    if split_name == "test_set":
        print("\nTest set: Filtering to verified purchases only (no synthetic generation)")
        final_df = df.filter(col("verified_purchase") == 1)
        
        # Final statistics
        final_count = final_df.count()
        print(f"\nFinal statistics for test set:")
        print(f"Total records (verified only): {final_count:,}")
        print(f"Data reduction: {total_count:,} -> {final_count:,} ({(1-final_count/total_count)*100:.2f}% removed)")
        
        # Save the dataset
        print(f"\nSaving test dataset to {output_path}")
        final_df.coalesce(200).write.mode("overwrite").parquet(output_path)
        
        # Cleanup
        df.unpersist()
        
        elapsed_time = time.time() - start_time
        print(f"\nCompleted {split_name} in {elapsed_time/60:.2f} minutes")
        
        return {
            'split_name': split_name,
            'original_count': total_count,
            'final_count': final_count,
            'synthetic_1_count': 0,  # Not applicable for test set
            'synthetic_0_count': 0,  # Not applicable for test set
            'processing_time': elapsed_time
        }
    
    # Normal synthetic processing for training sets
    # Step 1: Create verified_purchase_synthetic based on propensity
    print("\nStep 1: Creating verified_purchase_synthetic variable...")
    print("Using pre-transformed probabilities from Step02 (expit(-8.5 + 2*logit(p_raw)))")
    print("Note: Probabilities were already transformed with 2x coefficient scaling and -8.5 intercept in Step02")
    
    # The probabilities from Step02 have already been transformed using:
    # p_transformed = expit(-8.5 + 2*logit(p_raw))
    # So we can use them directly for synthetic data generation
    
    # Create synthetic variable based on the already-transformed probability
    df_with_synthetic = df.withColumn(
        "verified_purchase_synthetic",
        when(rand(seed=seed) < col("verified_purchase_prob"), 1).otherwise(0)
    )
    
    # Show probability statistics before dropping columns
    prob_stats = df_with_synthetic.select(
        mean("verified_purchase_prob").alias("mean_prob"),
        stddev("verified_purchase_prob").alias("std_prob"),
        spark_min("verified_purchase_prob").alias("min_prob"),
        spark_max("verified_purchase_prob").alias("max_prob")
    ).collect()[0]
    
    print(f"\nProbability statistics (already transformed):")
    print(f"Mean: {prob_stats['mean_prob']:.4f}, Std: {prob_stats['std_prob']:.4f}")
    print(f"Min: {prob_stats['min_prob']:.4f}, Max: {prob_stats['max_prob']:.4f}")
    
    # Count synthetic verified purchases
    synthetic_verified_count = df_with_synthetic.filter(col("verified_purchase_synthetic") == 1).count()
    print(f"\nSynthetic verified purchases: {synthetic_verified_count:,} ({synthetic_verified_count/total_count*100:.2f}%)")
    
    # Drop intermediate columns to keep data clean
    df_with_synthetic = df_with_synthetic.drop("prob_clipped", "logit_original_prob", "adjusted_prob")
    
    # Step 2: Process verified purchases (no longer need to prepare non-verified for sampling)
    print("\nStep 2: Processing verified purchases...")
    verified_df = df_with_synthetic.filter(col("verified_purchase") == 1)
    
    # Split verified purchases by synthetic status
    verified_synthetic_1 = verified_df.filter(col("verified_purchase_synthetic") == 1)
    verified_synthetic_0 = verified_df.filter(col("verified_purchase_synthetic") == 0)
    
    vs1_count = verified_synthetic_1.count()
    vs0_count = verified_synthetic_0.count()
    
    print(f"Verified purchases with synthetic=1 (unchanged): {vs1_count:,}")
    print(f"Verified purchases with synthetic=0 (need replacement): {vs0_count:,}")
    
    # For verified purchases with synthetic=0, we need to replace title, text, rating
    if vs0_count > 0:
        print("\nReplacing review characteristics with NA and -1 for synthetic=0 verified purchases...")
        
        # Simply replace the review content with NA and rating with -1
        # No need to sample from non-verified purchases anymore
        verified_synthetic_0_replaced = verified_synthetic_0 \
            .withColumn("title", lit("(missing)")) \
            .withColumn("text", lit("(missing)")) \
            .withColumn("rating", lit(-1).cast(IntegerType()))
        
        # Combine all verified purchases
        final_df = verified_synthetic_1.unionByName(verified_synthetic_0_replaced)
    else:
        # If no replacements needed, just use verified_synthetic_1
        final_df = verified_synthetic_1
    
    # Step 3: Final statistics
    print("\nFinal statistics:")
    final_count = final_df.count()
    final_synthetic_1 = final_df.filter(col("verified_purchase_synthetic") == 1).count()
    final_synthetic_0 = final_df.filter(col("verified_purchase_synthetic") == 0).count()
    
    print(f"Total records in synthetic dataset: {final_count:,}")
    print(f"Records with synthetic=1: {final_synthetic_1:,} ({final_synthetic_1/final_count*100:.2f}%)")
    print(f"Records with synthetic=0: {final_synthetic_0:,} ({final_synthetic_0/final_count*100:.2f}%)")
    print(f"Data reduction: {total_count:,} -> {final_count:,} ({(1-final_count/total_count)*100:.2f}% removed)")
    
    # Step 4: Save the synthetic dataset
    print(f"\nSaving synthetic dataset to {output_path}")
    final_df.coalesce(200).write.mode("overwrite").parquet(output_path)
    
    # Cleanup
    df.unpersist()
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted {split_name} in {elapsed_time/60:.2f} minutes")
    
    return {
        'split_name': split_name,
        'original_count': total_count,
        'final_count': final_count,
        'synthetic_1_count': final_synthetic_1,
        'synthetic_0_count': final_synthetic_0,
        'processing_time': elapsed_time
    }

def main():
    """Main execution function"""
    print(f"Starting synthetic data generation at {datetime.now()}")
    
    # Create Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Define paths
    # Step03 reads from Step02's output (which has probabilities added)
    input_base = "~/data_directory/amazon_reviews_2023_with_probs"
    output_base = "~/data_directory/amazon_reviews_2023_synthetic"
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Process each split
    splits = [
        ("training_set_1", f"{input_base}/training_set_1", f"{output_base}/training_set_1"),
        ("training_set_2", f"{input_base}/training_set_2", f"{output_base}/training_set_2"),
        ("test_set", f"{input_base}/test_set", f"{output_base}/test_set")
    ]
    
    results = []
    total_start_time = time.time()
    
    for split_name, input_path, output_path in splits:
        result = process_split(spark, split_name, input_path, output_path)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL SPLITS")
    print("="*60)
    
    total_original = sum(r['original_count'] for r in results)
    total_final = sum(r['final_count'] for r in results)
    total_synthetic_1 = sum(r['synthetic_1_count'] for r in results)
    total_synthetic_0 = sum(r['synthetic_0_count'] for r in results)
    
    for r in results:
        print(f"\n{r['split_name']}:")
        print(f"  Original: {r['original_count']:,}")
        print(f"  Final: {r['final_count']:,} ({r['final_count']/r['original_count']*100:.2f}%)")
        print(f"  Synthetic=1: {r['synthetic_1_count']:,}")
        print(f"  Synthetic=0: {r['synthetic_0_count']:,}")
        print(f"  Time: {r['processing_time']/60:.2f} minutes")
    
    print(f"\nTOTAL ACROSS ALL SPLITS:")
    print(f"  Original: {total_original:,}")
    print(f"  Final: {total_final:,} ({total_final/total_original*100:.2f}%)")
    print(f"  Synthetic=1: {total_synthetic_1:,} ({total_synthetic_1/total_final*100:.2f}%)")
    print(f"  Synthetic=0: {total_synthetic_0:,} ({total_synthetic_0/total_final*100:.2f}%)")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal processing time: {total_elapsed/60:.2f} minutes")
    
    # Save summary to file
    summary_path = "synthetic_data_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Synthetic Data Generation Summary\n")
        f.write(f"Generated at: {datetime.now()}\n")
        f.write(f"Total processing time: {total_elapsed/60:.2f} minutes\n\n")
        
        for r in results:
            f.write(f"{r['split_name']}:\n")
            f.write(f"  Original count: {r['original_count']:,}\n")
            f.write(f"  Final count: {r['final_count']:,}\n")
            f.write(f"  Synthetic=1: {r['synthetic_1_count']:,}\n")
            f.write(f"  Synthetic=0: {r['synthetic_0_count']:,}\n")
            f.write(f"  Processing time: {r['processing_time']/60:.2f} minutes\n\n")
        
        f.write(f"Total original: {total_original:,}\n")
        f.write(f"Total final: {total_final:,}\n")
        f.write(f"Total synthetic=1: {total_synthetic_1:,}\n")
        f.write(f"Total synthetic=0: {total_synthetic_0:,}\n")
    
    print(f"\nSummary saved to {summary_path}")
    
    # Stop Spark
    spark.stop()
    
    print(f"\nSynthetic data generation completed at {datetime.now()}")

if __name__ == "__main__":
    main()