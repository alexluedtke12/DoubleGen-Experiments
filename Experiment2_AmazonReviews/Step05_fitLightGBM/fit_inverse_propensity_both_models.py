#!/usr/bin/env python3
"""
Fit LightGBM models with inverse propensity weights on Amazon Reviews synthetic data
- Fits TWO models per training set: RIGHT (all features) and WRONG (no categories)
- Uses distributed approach with Spark for massive datasets
- Fits separate models on train1 and train2 to predict verified_purchase_synthetic
- Creates both ipw_right and ipw_wrong columns
- Uses binary classification with standard cross-entropy loss to estimate propensities
- Makes predictions on opposite training set and creates inverse propensity weights
"""

import os
import time
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType
from pyspark.ml.feature import VectorAssembler
import pandas as pd

def create_inverse_propensity_weights(df, predictions_col, weight_col_name, 
                                     synthetic_col='verified_purchase_synthetic',
                                     total_count=None, synthetic_count=None,
                                     truncation=1000):
    """
    Create inverse propensity weights by:
    1. Taking the predictions from the model (which are already 1/p)
    2. Clipping at minimum value of 1 (since 1/probability >= 1)
    3. First stabilization to target mean
    4. TRUNCATING at maximum value of 1000 (based on analysis of true propensities)
    5. Second stabilization to correct for truncation effect
    6. Setting to 0 where verified_purchase_synthetic is 0
    
    The key insight: synthetic records should have mean weight = total_count/synthetic_count
    so that when combined with zeros, the population mean = 1.0
    """
    # Step 1: Create initial inverse propensity weights (clipped at min=1, no max yet)
    df = df.withColumn(
        weight_col_name,
        F.when(
            F.col(synthetic_col) == 1,
            F.greatest(F.col(predictions_col), F.lit(1.0))
        ).otherwise(0.0)
    )
    
    # Calculate the target mean for synthetic records
    if total_count and synthetic_count:
        target_mean_for_synthetic = total_count / synthetic_count
        print(f"  Target mean for synthetic records: {target_mean_for_synthetic:.4f} (={total_count}/{synthetic_count})")
    else:
        print("  WARNING: Counts not provided, stabilization may be incorrect!")
        total_count = df.count()
        synthetic_count = df.filter(F.col(synthetic_col) == 1).count()
        target_mean_for_synthetic = total_count / synthetic_count if synthetic_count > 0 else 1.0
    
    # Step 2: First stabilization (before truncation)
    current_mean_nonzero = df.filter(F.col(weight_col_name) > 0).agg(
        F.mean(weight_col_name).alias('mean_weight')
    ).collect()[0]['mean_weight']
    
    print(f"  Pre-truncation mean of non-zero weights: {current_mean_nonzero:.4f}")
    
    scaling_factor_1 = target_mean_for_synthetic / current_mean_nonzero if current_mean_nonzero > 0 else 1.0
    print(f"  First scaling factor: {scaling_factor_1:.4f}")
    
    df = df.withColumn(
        weight_col_name,
        F.when(
            F.col(weight_col_name) > 0,
            F.col(weight_col_name) * scaling_factor_1
        ).otherwise(0.0)
    )
    
    # Step 3: Apply truncation at 1000
    df = df.withColumn(
        weight_col_name,
        F.when(
            F.col(weight_col_name) > truncation,
            F.lit(float(truncation))
        ).otherwise(F.col(weight_col_name))
    )
    
    # Count truncated weights
    truncated_count = df.filter(F.col(weight_col_name) == truncation).count()
    total_nonzero = df.filter(F.col(weight_col_name) > 0).count()
    
    if truncated_count > 0:
        print(f"  Truncated {truncated_count:,} weights at {truncation} ({100*truncated_count/total_nonzero:.2f}% of non-zero weights)")
    
    # Step 4: Second stabilization (after truncation to correct for truncation effect)
    current_mean_after_truncation = df.filter(F.col(weight_col_name) > 0).agg(
        F.mean(weight_col_name).alias('mean_weight')
    ).collect()[0]['mean_weight']
    
    print(f"  Post-truncation mean of non-zero weights: {current_mean_after_truncation:.4f}")
    
    scaling_factor_2 = target_mean_for_synthetic / current_mean_after_truncation if current_mean_after_truncation > 0 else 1.0
    print(f"  Second scaling factor (to correct truncation): {scaling_factor_2:.4f}")
    
    df = df.withColumn(
        weight_col_name,
        F.when(
            F.col(weight_col_name) > 0,
            F.col(weight_col_name) * scaling_factor_2
        ).otherwise(0.0)
    )
    
    return df


def main():
    print(f"Starting LightGBM inverse propensity fitting (RIGHT and WRONG models) at {datetime.now()}")
    
    # Initialize Spark session with local mode (40 CPUs, 400GB RAM)
    spark = SparkSession.builder \
        .appName("LightGBMInversePropensityBoth") \
        .config("spark.master", "local[40]") \
        .config("spark.driver.memory", "400g") \
        .config("spark.executor.memory", "400g") \
        .config("spark.driver.maxResultSize", "160g") \
        .config("spark.sql.shuffle.partitions", "160") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.3") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Process each training set
    training_sets = ['train1', 'train2']
    
    # Read first dataset to discover features
    print("Reading train1 to discover available features...")
    sample_df = spark.read.parquet(f"~/data_directory/amazon_reviews_2023_synthetic/training_set_1/")
    
    # Define baseline feature columns for RIGHT model (all features)
    baseline_features_right = [
        'title_length', 'description_length', 'image_count', 'video_count', 'details_length'
    ]
    
    # Dynamically discover category columns from the data
    category_cols = [col for col in sample_df.columns if col.startswith('category_')]
    baseline_features_right.extend(sorted(category_cols))  # Sort for consistency
    
    # Define features for WRONG model (no categories)
    baseline_features_wrong = [
        'title_length', 'description_length', 'image_count', 'video_count', 'details_length'
    ]
    
    print(f"Found {len(category_cols)} category columns")
    print(f"RIGHT model: Using {len(baseline_features_right)} baseline features (all)")
    print(f"WRONG model: Using {len(baseline_features_wrong)} baseline features (no categories)")
    
    results = []
    dir_mapping = {'train1': 'training_set_1', 'train2': 'training_set_2'}
    
    for train_set in training_sets:
        print(f"\n{'='*80}")
        print(f"Processing {train_set.upper()} - Fitting both RIGHT and WRONG models")
        print(f"{'='*80}")
        
        # Determine the opposite set for prediction
        opposite_set = 'train2' if train_set == 'train1' else 'train1'
        
        # Read training data from starred (Step04) output
        print(f"Reading {train_set} from starred data...")
        train_dir = dir_mapping[train_set]
        train_path = f"~/data_directory/amazon_reviews_2023_starred/{train_dir}/"
        df_train = spark.read.parquet(train_path)
        
        # Check if verified_purchase_synthetic exists
        if 'verified_purchase_synthetic' not in df_train.columns:
            print(f"ERROR: verified_purchase_synthetic not found in {train_set}")
            continue
            
        # Create binary classification target
        df_train = df_train.withColumn(
            'label',
            F.when(F.col('verified_purchase_synthetic') == 1, 1.0).otherwise(0.0)
        )
        
        # Get basic statistics
        train_count = df_train.count()
        synthetic_count = df_train.filter(F.col('verified_purchase_synthetic') == 1).count()
        print(f"{train_set} size: {train_count:,}")
        print(f"Synthetic verified purchases: {synthetic_count:,} ({100*synthetic_count/train_count:.2f}%)")
        
        # Read opposite set for prediction
        print(f"\nReading {opposite_set} for prediction from starred data...")
        opposite_dir = dir_mapping[opposite_set]
        opposite_path = f"~/data_directory/amazon_reviews_2023_starred/{opposite_dir}/"
        df_opposite = spark.read.parquet(opposite_path)
        
        # Train both models first
        print(f"\n--- Training RIGHT model with {len(baseline_features_right)} features ---")
        
        # Create feature vector for training
        assembler_train_right = VectorAssembler(
            inputCols=baseline_features_right,
            outputCol="features",
            handleInvalid="skip"
        )
        df_train_features_right = assembler_train_right.transform(df_train)
        
        # Configure and train RIGHT model
        from synapse.ml.lightgbm import LightGBMClassifier
        lgbm_right = LightGBMClassifier(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction_right",
            probabilityCol="probability_right",
            rawPredictionCol="raw_prediction_right",
            numIterations=100,
            learningRate=0.1,
            numLeaves=31,
            maxDepth=-1,
            minDataInLeaf=20,
            featureFraction=0.8,
            baggingFraction=0.8,
            baggingFreq=5,
            objective="binary",
            metric="binary_logloss",
            numThreads=40,
            verbosity=1,
            categoricalSlotIndexes=[],
            categoricalSlotNames=[]
        )
        
        print(f"Training RIGHT model on {train_set}...")
        start_time = time.time()
        model_right = lgbm_right.fit(df_train_features_right)
        training_time_right = time.time() - start_time
        print(f"RIGHT model training completed in {training_time_right:.2f} seconds")
        
        # Save RIGHT model
        model_right_path = f"~/data_directory/lightgbm_inverse_propensity_right_{train_set}"
        print(f"Saving RIGHT model to: {model_right_path}")
        model_right.write().overwrite().save(model_right_path)
        
        print(f"\n--- Training WRONG model with {len(baseline_features_wrong)} features ---")
        
        # Create feature vector for training WRONG model
        assembler_train_wrong = VectorAssembler(
            inputCols=baseline_features_wrong,
            outputCol="features",
            handleInvalid="skip"
        )
        df_train_features_wrong = assembler_train_wrong.transform(df_train)
        
        # Configure and train WRONG model
        lgbm_wrong = LightGBMClassifier(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction_wrong",
            probabilityCol="probability_wrong",
            rawPredictionCol="raw_prediction_wrong",
            numIterations=100,
            learningRate=0.1,
            numLeaves=31,
            maxDepth=-1,
            minDataInLeaf=20,
            featureFraction=0.8,
            baggingFraction=0.8,
            baggingFreq=5,
            objective="binary",
            metric="binary_logloss",
            numThreads=40,
            verbosity=1,
            categoricalSlotIndexes=[],
            categoricalSlotNames=[]
        )
        
        print(f"Training WRONG model on {train_set}...")
        start_time = time.time()
        model_wrong = lgbm_wrong.fit(df_train_features_wrong)
        training_time_wrong = time.time() - start_time
        print(f"WRONG model training completed in {training_time_wrong:.2f} seconds")
        
        # Save WRONG model
        model_wrong_path = f"~/data_directory/lightgbm_inverse_propensity_wrong_{train_set}"
        print(f"Saving WRONG model to: {model_wrong_path}")
        model_wrong.write().overwrite().save(model_wrong_path)
        
        # Now apply both models to the opposite set
        print(f"\nApplying both models to {opposite_set}...")
        
        # OPTIMIZATION: Only process synthetic records, set weights to 0 for others
        print(f"Filtering to synthetic records only for model predictions...")
        df_synthetic = df_opposite.filter(F.col('verified_purchase_synthetic') == 1)
        df_non_synthetic = df_opposite.filter(F.col('verified_purchase_synthetic') != 1)
        
        synthetic_count = df_synthetic.count()
        non_synthetic_count = df_non_synthetic.count()
        total_count = synthetic_count + non_synthetic_count
        print(f"Processing {synthetic_count:,} synthetic records for predictions")
        print(f"Setting weights to 0 for {non_synthetic_count:,} non-synthetic records")
        print(f"Total records: {total_count:,}")
        
        # Process synthetic records with both models
        df_to_predict = df_synthetic
        
        # Add RIGHT predictions
        assembler_right = VectorAssembler(
            inputCols=baseline_features_right,
            outputCol="features",  # Models expect "features" column
            handleInvalid="skip"
        )
        df_to_predict = assembler_right.transform(df_to_predict)
        
        # Apply RIGHT model predictions (already trained above)
        df_to_predict = model_right.transform(df_to_predict)
        
        # Rename features column to avoid conflict
        df_to_predict = df_to_predict.withColumnRenamed("features", "features_right")
        
        # Extract RIGHT propensity and compute inverse
        df_to_predict = df_to_predict.withColumn(
            'propensity_right',
            F.udf(lambda v: float(v[1]), FloatType())(F.col('probability_right'))
        ).withColumn(
            'inverse_prop_raw_right',
            F.lit(1.0) / F.col('propensity_right')
        )
        
        # Create RIGHT inverse propensity weights
        print("\nStabilizing RIGHT model weights...")
        df_to_predict = create_inverse_propensity_weights(
            df_to_predict,
            'inverse_prop_raw_right',
            'ipw_right',
            'verified_purchase_synthetic',
            total_count=total_count,
            synthetic_count=synthetic_count
        )
        
        # Add WRONG predictions
        assembler_wrong = VectorAssembler(
            inputCols=baseline_features_wrong,
            outputCol="features",  # Models expect "features" column
            handleInvalid="skip"
        )
        df_to_predict = assembler_wrong.transform(df_to_predict)
        
        # Apply WRONG model predictions (already trained above)
        df_to_predict = model_wrong.transform(df_to_predict)
        
        # Rename features column for clarity
        df_to_predict = df_to_predict.withColumnRenamed("features", "features_wrong")
        
        # Extract WRONG propensity and compute inverse
        df_to_predict = df_to_predict.withColumn(
            'propensity_wrong',
            F.udf(lambda v: float(v[1]), FloatType())(F.col('probability_wrong'))
        ).withColumn(
            'inverse_prop_raw_wrong',
            F.lit(1.0) / F.col('propensity_wrong')
        )
        
        # Create WRONG inverse propensity weights
        print("\nStabilizing WRONG model weights...")
        df_to_predict = create_inverse_propensity_weights(
            df_to_predict,
            'inverse_prop_raw_wrong',
            'ipw_wrong',
            'verified_purchase_synthetic',
            total_count=total_count,
            synthetic_count=synthetic_count
        )
        
        # Add zero weights to non-synthetic records
        df_non_synthetic = df_non_synthetic.withColumn('ipw_right', F.lit(0.0))
        df_non_synthetic = df_non_synthetic.withColumn('ipw_wrong', F.lit(0.0))
        
        # Combine back together
        df_final = df_to_predict.unionByName(df_non_synthetic, allowMissingColumns=True)
        
        # Calculate statistics for both weights
        print(f"\nCalculating weight statistics...")
        weight_stats_both = df_final.agg(
            # RIGHT weight stats
            F.sum(F.when(F.col('ipw_right') > 0, 1).otherwise(0)).alias('non_zero_right'),
            F.mean(F.when(F.col('ipw_right') > 0, F.col('ipw_right')).otherwise(None)).alias('mean_non_zero_right'),
            F.mean('ipw_right').alias('mean_population_right'),
            F.max('ipw_right').alias('max_right'),
            # WRONG weight stats
            F.sum(F.when(F.col('ipw_wrong') > 0, 1).otherwise(0)).alias('non_zero_wrong'),
            F.mean(F.when(F.col('ipw_wrong') > 0, F.col('ipw_wrong')).otherwise(None)).alias('mean_non_zero_wrong'),
            F.mean('ipw_wrong').alias('mean_population_wrong'),
            F.max('ipw_wrong').alias('max_wrong')
        ).collect()[0]
        
        print(f"\nRIGHT weight statistics:")
        print(f"  Non-zero weights: {weight_stats_both['non_zero_right']:,}")
        print(f"  Mean non-zero: {weight_stats_both['mean_non_zero_right']:.4f}")
        print(f"  Mean population (should be 1.0): {weight_stats_both['mean_population_right']:.4f}")
        print(f"  Max: {weight_stats_both['max_right']:.4f}")
        
        print(f"\nWRONG weight statistics:")
        print(f"  Non-zero weights: {weight_stats_both['non_zero_wrong']:,}")
        print(f"  Mean non-zero: {weight_stats_both['mean_non_zero_wrong']:.4f}")
        print(f"  Mean population (should be 1.0): {weight_stats_both['mean_population_wrong']:.4f}")
        print(f"  Max: {weight_stats_both['max_wrong']:.4f}")
        
        # Clean up intermediate columns (only from predicted records)
        columns_to_drop = [
            'features_right', 'features_wrong',
            'prediction_right', 'probability_right', 'raw_prediction_right',
            'prediction_wrong', 'probability_wrong', 'raw_prediction_wrong',
            'propensity_right', 'inverse_prop_raw_right',
            'propensity_wrong', 'inverse_prop_raw_wrong'
        ]
        # Drop columns that exist
        existing_cols = df_final.columns
        cols_to_drop_existing = [col for col in columns_to_drop if col in existing_cols]
        if cols_to_drop_existing:
            df_final = df_final.drop(*cols_to_drop_existing)
        
        # Drop the label column if it exists (not needed in output)
        if 'label' in df_final.columns:
            df_final = df_final.drop('label')
        
        # Save combined predictions with both weights
        output_path = f"~/data_directory/amazon_reviews_2023_inverse_propensity/{opposite_dir}/"
        print(f"Saving combined weights to: {output_path}")
        df_final.write.mode('overwrite').parquet(output_path)
        
        # Save results summary
        results.append({
            'train_set': train_set,
            'prediction_set': opposite_set,
            'training_count': train_count,
            'synthetic_count': synthetic_count,
            'training_time_right': training_time_right,
            'training_time_wrong': training_time_wrong,
            'weight_stats': weight_stats_both
        })
    
    # Save comprehensive results
    output_file = "/home/your_username/cf_reviews/Step05_fitLightGBM/inverse_propensity_results_both.txt"
    with open(output_file, 'w') as f:
        f.write("LIGHTGBM INVERSE PROPENSITY WEIGHT ESTIMATION RESULTS (RIGHT AND WRONG MODELS)\n")
        f.write("="*80 + "\n")
        f.write(f"\nAnalysis timestamp: {datetime.now()}\n")
        f.write(f"RIGHT model features: {len(baseline_features_right)} (all baseline features)\n")
        f.write(f"WRONG model features: {len(baseline_features_wrong)} (no category indicators)\n")
        f.write("\nNOTE: Using distributed SynapseML LightGBM with binary classification\n")
        f.write("Estimates propensity p(S=1|X) then computes inverse propensity weights as 1/p\n\n")
        
        for result in results:
            f.write(f"\nTRAINING SET: {result['train_set']}, PREDICTIONS ON: {result['prediction_set']}\n")
            f.write("-"*70 + "\n")
            f.write(f"Training size: {result['training_count']:,} records\n")
            f.write(f"Synthetic verified purchases: {result['synthetic_count']:,} ({100*result['synthetic_count']/result['training_count']:.2f}%)\n\n")
            
            # RIGHT model results
            f.write("RIGHT Model (all features):\n")
            f.write(f"  Training time: {result['training_time_right']:.2f} seconds\n")
            f.write(f"  Non-zero weights: {result['weight_stats']['non_zero_right']:,}\n")
            f.write(f"  Mean non-zero weight: {result['weight_stats']['mean_non_zero_right']:.4f}\n")
            f.write(f"  Mean population weight: {result['weight_stats']['mean_population_right']:.4f}\n")
            f.write(f"  Max weight: {result['weight_stats']['max_right']:.4f}\n")
            
            f.write("\nWRONG Model (no categories):\n")
            f.write(f"  Training time: {result['training_time_wrong']:.2f} seconds\n")
            f.write(f"  Non-zero weights: {result['weight_stats']['non_zero_wrong']:,}\n")
            f.write(f"  Mean non-zero weight: {result['weight_stats']['mean_non_zero_wrong']:.4f}\n")
            f.write(f"  Mean population weight: {result['weight_stats']['mean_population_wrong']:.4f}\n")
            f.write(f"  Max weight: {result['weight_stats']['max_wrong']:.4f}\n")
            f.write("\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Copy test set unchanged from Step04 to maintain complete dataset structure
    print("\nCopying test set from Step04 (no weights needed)...")
    test_input_path = "~/data_directory/amazon_reviews_2023_starred/test_set/"
    test_output_path = "~/data_directory/amazon_reviews_2023_inverse_propensity/test_set/"
    
    df_test = spark.read.parquet(test_input_path)
    df_test.write.mode('overwrite').parquet(test_output_path)
    print(f"Test set copied to: {test_output_path}")
    
    # Stop Spark session
    spark.stop()
    print("\nStep 05 completed successfully!")

if __name__ == "__main__":
    main()