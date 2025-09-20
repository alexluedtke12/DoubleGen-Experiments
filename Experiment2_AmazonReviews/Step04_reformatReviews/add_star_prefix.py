#!/usr/bin/env python3
"""
Step04: Format Reviews with Rating Prefix

This script:
1. Loads all datasets from Step03 (synthetic data)
2. Filters out reviews without ratings (if any)
3. Formats each review as: {N} stars: {review text} (or "1 star:" for singular)
4. Saves the updated datasets to a new location
"""

import os
import sys
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, concat, lit, isnan, isnull, count, when, repeat, expr
)

def create_spark_session():
    """Create optimized Spark session with local mode (40 CPUs, 400GB RAM)
    
    Configuration for single machine with 40 cores and 400GB RAM:
    - Local mode with 40 CPU cores
    - 400GB memory (shared between driver and executors in local mode)
    - Shuffle partitions set to 4x cores (160)
    - Adaptive query execution for optimal performance
    """
    return SparkSession.builder \
        .appName("Step04_FormatReviews") \
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

def process_dataset(spark, dataset_name, input_path, output_path):
    """Process a single dataset to add star prefix"""
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Read the dataset
    print(f"Reading data from {input_path}")
    df = spark.read.parquet(input_path)
    
    # Get initial count
    initial_count = df.count()
    print(f"Initial record count: {initial_count:,}")
    
    # Check for missing ratings
    missing_ratings = df.filter(
        isnull(col("rating")) | isnan(col("rating"))
    ).count()
    
    if missing_ratings > 0:
        print(f"WARNING: Found {missing_ratings:,} records with missing ratings ({missing_ratings/initial_count*100:.2f}%)")
        print("These records will be filtered out.")
        
        # Filter out records with missing ratings
        df = df.filter(
            ~isnull(col("rating")) & ~isnan(col("rating"))
        )
    else:
        print("All records have valid ratings.")
    
    # Create formatted text with rating and review
    # Format: "{N} stars: {text}" (or "1 star:" for singular)
    print("\nFormatting reviews with rating prefix...")
    df_with_prefix = df.withColumn(
        "text",
        concat(
            col("rating").cast("int").cast("string"),
            lit(" "),
            when(col("rating").cast("int") == 1, lit("star")).otherwise(lit("stars")),
            lit(": "),
            col("text")
        )
    )
    
    # Get final count
    final_count = df_with_prefix.count()
    print(f"\nFinal record count: {final_count:,}")
    
    if missing_ratings > 0:
        print(f"Records removed: {initial_count - final_count:,}")
    
    # Show some examples
    print("\nExamples of formatted reviews:")
    examples = df_with_prefix.select("rating", "text").limit(5).collect()
    for i, row in enumerate(examples, 1):
        text_preview = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
        print(f"  {i}. {text_preview}")
    
    # Get statistics on ratings distribution
    print("\nRating distribution:")
    rating_counts = df_with_prefix.groupBy("rating").count().orderBy("rating").collect()
    for row in rating_counts:
        pct = row['count'] / final_count * 100
        rating_int = int(row['rating'])
        star_text = f"{rating_int} star" if rating_int == 1 else f"{rating_int} stars"
        print(f"  {star_text}: {row['count']:,} ({pct:.2f}%)")
    
    # Save the dataset
    print(f"\nSaving dataset to {output_path}")
    df_with_prefix.coalesce(200).write.mode("overwrite").parquet(output_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted {dataset_name} in {elapsed_time/60:.2f} minutes")
    
    return {
        'dataset_name': dataset_name,
        'initial_count': initial_count,
        'final_count': final_count,
        'missing_ratings': missing_ratings,
        'processing_time': elapsed_time
    }

def main():
    """Main execution function"""
    print(f"Starting review formatting at {datetime.now()}")
    
    # Create Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Define paths
    input_base = "~/data_directory/amazon_reviews_2023_synthetic"
    output_base = "~/data_directory/amazon_reviews_2023_starred"
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Process each dataset
    datasets = [
        ("propensity_set", f"{input_base}/../amazon_reviews_2023_splits/propensity_set", 
         f"{output_base}/propensity_set"),
        ("training_set_1", f"{input_base}/training_set_1", f"{output_base}/training_set_1"),
        ("training_set_2", f"{input_base}/training_set_2", f"{output_base}/training_set_2"),
        ("test_set", f"{input_base}/test_set", f"{output_base}/test_set")
    ]
    
    results = []
    total_start_time = time.time()
    
    for dataset_name, input_path, output_path in datasets:
        # Check if input path exists
        if not os.path.exists(input_path):
            print(f"\nWARNING: Input path not found: {input_path}")
            print(f"Skipping {dataset_name}")
            continue
            
        result = process_dataset(spark, dataset_name, input_path, output_path)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL DATASETS")
    print("="*60)
    
    total_initial = sum(r['initial_count'] for r in results)
    total_final = sum(r['final_count'] for r in results)
    total_missing = sum(r['missing_ratings'] for r in results)
    
    for r in results:
        print(f"\n{r['dataset_name']}:")
        print(f"  Initial count: {r['initial_count']:,}")
        print(f"  Final count: {r['final_count']:,}")
        print(f"  Missing ratings: {r['missing_ratings']:,}")
        print(f"  Time: {r['processing_time']/60:.2f} minutes")
    
    print(f"\nTOTAL ACROSS ALL DATASETS:")
    print(f"  Initial records: {total_initial:,}")
    print(f"  Final records: {total_final:,}")
    print(f"  Total missing ratings: {total_missing:,}")
    if total_missing > 0:
        print(f"  Overall removal rate: {total_missing/total_initial*100:.2f}%")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal processing time: {total_elapsed/60:.2f} minutes")
    
    # Save summary to file
    summary_path = "review_formatting_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Review Formatting Summary\n")
        f.write(f"Generated at: {datetime.now()}\n")
        f.write(f"Total processing time: {total_elapsed/60:.2f} minutes\n\n")
        
        for r in results:
            f.write(f"{r['dataset_name']}:\n")
            f.write(f"  Initial count: {r['initial_count']:,}\n")
            f.write(f"  Final count: {r['final_count']:,}\n")
            f.write(f"  Missing ratings: {r['missing_ratings']:,}\n")
            f.write(f"  Processing time: {r['processing_time']/60:.2f} minutes\n\n")
        
        f.write(f"Total initial: {total_initial:,}\n")
        f.write(f"Total final: {total_final:,}\n")
        f.write(f"Total missing ratings: {total_missing:,}\n")
    
    print(f"\nSummary saved to {summary_path}")
    
    # Stop Spark
    spark.stop()
    
    print(f"\nReview formatting completed at {datetime.now()}")

if __name__ == "__main__":
    main()