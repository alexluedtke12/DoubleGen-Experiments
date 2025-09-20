#!/usr/bin/env python3
"""
Logistic Regression Analysis on Amazon Reviews 2023 Dataset - Simple Version
- Fit logistic regression on propensity set to predict verified_purchase
- Use only baseline features (no text, rating, or review length)
- Generate and save predictions separately
"""

import os
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def main():
    print(f"Starting logistic regression analysis at {datetime.now()}")
    
    # Initialize Spark session with local mode (40 CPUs, 400GB RAM)
    spark = SparkSession.builder \
        .appName("LogisticRegressionPropensitySimple") \
        .config("spark.master", "local[40]") \
        .config("spark.driver.memory", "400g") \
        .config("spark.executor.memory", "400g") \
        .config("spark.driver.maxResultSize", "160g") \
        .config("spark.sql.shuffle.partitions", "160") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Read propensity set first to discover available columns
    print("Reading propensity set to discover available features...")
    df_propensity = spark.read.parquet("~/data_directory/amazon_reviews_2023_splits/propensity_set/")
    
    # Define baseline feature columns (excluding text, rating, review length)
    # Start with non-category features
    baseline_features = [
        'title_length', 'description_length', 'image_count', 'video_count', 'details_length'
    ]
    
    # Dynamically discover category columns from the data
    category_cols = [col for col in df_propensity.columns if col.startswith('category_')]
    baseline_features.extend(sorted(category_cols))  # Sort for consistency
    
    print(f"Found {len(category_cols)} category columns")
    print(f"Using {len(baseline_features)} baseline features total")
    
    # ========================================
    # STEP 1: Train Model on Propensity Set
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: TRAINING MODEL ON PROPENSITY SET")
    print("="*80)
    
    # Convert boolean to double for logistic regression
    df_propensity = df_propensity.withColumn(
        "verified_purchase_num",
        F.when(F.col("verified_purchase"), 1.0).otherwise(0.0)
    )
    
    # Get count
    propensity_count = df_propensity.count()
    print(f"Propensity set size: {propensity_count:,}")
    
    # Create pipeline
    assembler = VectorAssembler(
        inputCols=baseline_features,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    logr = LogisticRegression(
        featuresCol="features",
        labelCol="verified_purchase_num",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
        tol=1e-6,
        fitIntercept=True,
        standardization=False
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, logr])
    
    # Train model
    print("Training logistic regression model...")
    start_time = time.time()
    model = pipeline.fit(df_propensity)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Extract and save coefficients
    logr_model = model.stages[-1]
    coef_data = [(baseline_features[i], float(logr_model.coefficients[i]), float(abs(logr_model.coefficients[i]))) 
                 for i in range(len(baseline_features))]
    coef_sorted = sorted(coef_data, key=lambda x: x[2], reverse=True)
    
    # Save results
    output_dir = "/home/your_username/cf_reviews/Step02_logisticRegression"
    output_file = os.path.join(output_dir, "logistic_regression_propensity_results.txt")
    
    with open(output_file, 'w') as f:
        f.write("LOGISTIC REGRESSION ANALYSIS RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"\nAnalysis timestamp: {datetime.now()}\n")
        f.write(f"Number of baseline features: {len(baseline_features)}\n")
        f.write(f"\nTraining data: Propensity set only\n")
        f.write(f"Training size: {propensity_count:,} records\n")
        f.write(f"Model intercept: {float(logr_model.intercept):.6f}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write("\nCOEFFICIENTS (sorted by absolute magnitude):\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Rank':<6} {'Feature':<40} {'Coefficient':>12} {'|Coef|':>10}\n")
        f.write("-"*70 + "\n")
        for idx, (feat, coef, abs_coef) in enumerate(coef_sorted):
            f.write(f"{idx+1:<6} {feat:<40} {coef:>12.6f} {abs_coef:>10.6f}\n")
    
    print(f"\nModel coefficients saved to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("MODEL TRAINING SUMMARY")
    print("="*80)
    print(f"Training set: Propensity set ({propensity_count:,} records)")
    print(f"Features used: {len(baseline_features)} baseline features only")
    print(f"\nTop 5 features by |coefficient|:")
    for i in range(min(5, len(coef_sorted))):
        feat, coef, _ = coef_sorted[i]
        print(f"  {i+1}. {feat}: {coef:.4f}")
    
    # Save the model
    model_path = "~/data_directory/logistic_regression_model"
    print(f"\nSaving model to: {model_path}")
    model.write().overwrite().save(model_path)
    print("Model saved successfully!")
    
    # ========================================
    # STEP 2: Generate Predictions (Separate Script)
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: PREDICTION GENERATION")
    print("="*80)
    print("To generate predictions for the three sets, run:")
    print("  python generate_predictions.py")
    print("This will load the saved model and process each dataset separately")
    print("to avoid memory issues.")
    
    # Stop Spark session
    spark.stop()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()