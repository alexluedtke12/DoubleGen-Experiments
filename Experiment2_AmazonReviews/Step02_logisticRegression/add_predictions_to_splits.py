#!/usr/bin/env python3
"""
Add verified_purchase predictions to split datasets
This script:
1. Loads the trained logistic regression model
2. For each split (training_set_1, training_set_2, test_set):
   - Reads the original data
   - Generates predictions
   - Writes back to the original location with the new column
"""

import os
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel

def main():
    print(f"Starting prediction addition to splits at {datetime.now()}")
    
    # Initialize Spark session with local mode (40 CPUs, 400GB RAM)
    spark = SparkSession.builder \
        .appName("AddPredictionsToSplits") \
        .config("spark.master", "local[40]") \
        .config("spark.driver.memory", "400g") \
        .config("spark.executor.memory", "400g") \
        .config("spark.driver.maxResultSize", "160g") \
        .config("spark.sql.shuffle.partitions", "160") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Load the saved model
    model_path = "~/data_directory/logistic_regression_model"
    print(f"Loading model from: {model_path}")
    model = PipelineModel.load(model_path)
    print("Model loaded successfully!")
    
    # Define UDF for extracting probability
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.functions import udf
    
    def get_prob_1(prob_vector):
        return float(prob_vector.toArray()[1])
    
    get_prob_1_udf = udf(get_prob_1, returnType=DoubleType())
    
    # Process each dataset
    # Input and output paths
    input_base = "~/data_directory/amazon_reviews_2023_splits"
    output_base = "~/data_directory/amazon_reviews_2023_with_probs"
    
    sets_to_process = [
        ("training_set_1", f"{input_base}/training_set_1", f"{output_base}/training_set_1"),
        ("training_set_2", f"{input_base}/training_set_2", f"{output_base}/training_set_2"),
        ("test_set", f"{input_base}/test_set", f"{output_base}/test_set")
    ]
    
    overall_start = time.time()
    
    for set_name, input_path, output_path in sets_to_process:
        print(f"\n{'='*80}")
        print(f"Processing {set_name}")
        print('='*80)
        
        set_start_time = time.time()
        
        # Check if output already exists
        try:
            df_check = spark.read.parquet(output_path)
            if "verified_purchase_prob" in df_check.columns:
                print(f"  ✓ Predictions already exist at {output_path}, skipping...")
                continue
        except:
            pass
        
        # Read the dataset from input path
        print(f"Reading {set_name} from: {input_path}")
        df_set = spark.read.parquet(input_path)
        
        # Count records
        set_count = df_set.count()
        print(f"Records in {set_name}: {set_count:,}")
        
        # Generate predictions
        print("Generating predictions...")
        pred_start = time.time()
        df_with_pred = model.transform(df_set)
        
        # Extract probability
        df_with_pred = df_with_pred.withColumn(
            "verified_purchase_prob_raw",
            get_prob_1_udf(F.col("probability"))
        )
        
        # Apply transformation: expit(-8.5 + 2*logit(p_raw))
        # This scales coefficients by 2x and uses intercept of -8.5
        # This transformation is applied here so Step03 can use the transformed probabilities directly
        # log() in PySpark is natural logarithm
        # Clip probabilities to avoid log(0) or log(inf)
        df_with_pred = df_with_pred.withColumn(
            "verified_purchase_prob_clipped",
            F.when(F.col("verified_purchase_prob_raw") < 0.001, 0.001)
             .when(F.col("verified_purchase_prob_raw") > 0.999, 0.999)
             .otherwise(F.col("verified_purchase_prob_raw"))
        )
        
        # expit(-8.5 + 2*logit(p)) = 1/(1 + exp(-(-8.5 + 2*logit(p))))
        # = 1/(1 + exp(8.5 - 2*logit(p)))
        df_with_pred = df_with_pred.withColumn(
            "verified_purchase_prob",
            1.0 / (1.0 + F.exp(8.5 - 2.0 * F.log(F.col("verified_purchase_prob_clipped") / (1.0 - F.col("verified_purchase_prob_clipped")))))
        )
        
        # Show transformation statistics
        print("Checking probability transformation...")
        stats = df_with_pred.select(
            F.mean("verified_purchase_prob_raw").alias("mean_raw"),
            F.mean("verified_purchase_prob").alias("mean_transformed"),
            F.min("verified_purchase_prob_raw").alias("min_raw"),
            F.min("verified_purchase_prob").alias("min_transformed"),
            F.max("verified_purchase_prob_raw").alias("max_raw"),
            F.max("verified_purchase_prob").alias("max_transformed")
        ).collect()[0]
        
        print(f"  Raw probabilities: mean={stats['mean_raw']:.4f}, min={stats['min_raw']:.4f}, max={stats['max_raw']:.4f}")
        print(f"  Transformed probs: mean={stats['mean_transformed']:.4f}, min={stats['min_transformed']:.4f}, max={stats['max_transformed']:.4f}")
        
        pred_time = time.time() - pred_start
        print(f"Predictions generated in {pred_time:.2f} seconds")
        
        # Select all original columns plus the new prediction
        # Drop intermediate columns (raw and clipped probabilities)
        original_cols = df_set.columns
        df_final = df_with_pred.select(*original_cols, "verified_purchase_prob")
        
        # Write to output location
        print(f"Writing to output location: {output_path}")
        write_start = time.time()
        
        # Use coalesce instead of repartition for better performance
        df_final.coalesce(200).write.mode("overwrite").parquet(output_path)
        
        write_time = time.time() - write_start
        print(f"Data written in {write_time:.2f} seconds")
        
        total_time = time.time() - set_start_time
        print(f"{set_name} completed in {total_time:.2f} seconds")
        
        # Verify
        df_verify = spark.read.parquet(output_path)
        if "verified_purchase_prob" in df_verify.columns:
            print(f"  ✓ Verification passed: predictions added successfully")
            # Show sample
            print("  Sample predictions (after transformation):")
            df_verify.select("verified_purchase", "verified_purchase_prob").show(5, truncate=False)
            
            # Show mean probability
            mean_prob = df_verify.select(F.mean("verified_purchase_prob")).collect()[0][0]
            print(f"  Mean transformed probability: {mean_prob:.4f}")
        else:
            print(f"  ✗ ERROR: predictions not found in output!")
    
    overall_time = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"ALL PREDICTIONS COMPLETED")
    print(f"Total time: {overall_time:.2f} seconds ({overall_time/60:.1f} minutes)")
    print('='*80)
    
    # Final summary
    print("\nFinal verification:")
    for set_name, input_path, output_path in sets_to_process:
        df = spark.read.parquet(output_path)
        has_pred = "verified_purchase_prob" in df.columns
        count = df.count()
        status = "✓" if has_pred else "✗"
        print(f"  {status} {set_name}: {count:,} records, predictions: {has_pred}")
    
    # Stop Spark session
    spark.stop()
    print("\nProcess complete!")

if __name__ == "__main__":
    main()