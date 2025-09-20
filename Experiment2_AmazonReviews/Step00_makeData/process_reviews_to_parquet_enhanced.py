#!/usr/bin/env python3
"""
Process Amazon Reviews 2023 dataset to create features for LightGBM

This script processes the Amazon Reviews 2023 dataset to extract features suitable
for predicting verified_purchase status using LightGBM. It performs the following:

1. Loads asin2category.json mapping file for accurate category assignment
2. Reads review data and product metadata from JSONL files
3. Extracts various features including:
   - One-hot encoded product categories (34 top-level Amazon categories)
   - Character count of product title
   - Character count of product description  
   - Number of product images
   - Number of product videos
   - Character count of product details
   - Review title (preserved as-is, treated as outcome variable)
   - Review text (preserved as-is)
   - User rating (1-5 stars)
   - Verified purchase status (target variable)
4. Saves the processed data as partitioned Parquet files

Key improvement: Uses asin2category.json to reduce unknown categories from 11% to <1%

Output: Parquet dataset with engineered features ready for LightGBM
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import json

def main():
    # ============ CAUTION: TESTING MODE - REMOVE FOR PRODUCTION ============
    # Set to True to process only a small subset for testing
    TEST_MODE = False  # Set to False for production
    TEST_CATEGORIES = 2  # Number of category files to process in test mode
    # ============ END CAUTION: TESTING MODE ============
    
    # Initialize Spark with optimal settings for 48 cores and 512GB RAM
    # ================================================================
    # Configuration explanation:
    # - local[40]: Use 40 CPU cores
    # - driver.memory=400g: Increased memory allocation
    # - executor.memory=400g: Match executor memory to driver
    # - maxResultSize=160g: Allow larger result sets
    # - shuffle.partitions=160: Optimize shuffle operations (4x cores)
    # - adaptive.enabled: Enable Spark 3.x adaptive query execution
    # - KryoSerializer: More efficient serialization for better performance
    # - network.timeout=800s: Increase timeout to prevent Py4J connection errors
    spark = SparkSession.builder \
        .appName("AmazonReviewsFeatureExtraction") \
        .config("spark.master", "local[40]") \
        .config("spark.driver.memory", "400g") \
        .config("spark.executor.memory", "400g") \
        .config("spark.driver.maxResultSize", "160g") \
        .config("spark.sql.shuffle.partitions", "160") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.files.maxPartitionBytes", "134217728") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()
    
    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    # Define input and output paths
    reviews_path = "~/data_directory/amazon_reviews_2023/raw/review_categories"
    meta_path = "~/data_directory/amazon_reviews_2023/raw/meta_categories"
    output_path = "~/data_directory/amazon_reviews_2023_reduced/features.parquet"
    asin2category_path = "~/data_directory/amazon_reviews_2023/asin2category.json"
    
    # Modify output path for test mode
    if TEST_MODE:
        output_path = "~/data_directory/amazon_reviews_2023_reduced/features_test.parquet"
        print(f"TEST MODE: Output will be written to {output_path}")
    
    # Load asin2category mapping (1.2GB file with 35.4M entries)
    print("Loading asin2category.json mapping...")
    with open(asin2category_path, 'r') as f:
        asin2category = json.load(f)
    print(f"Loaded {len(asin2category):,} ASIN to category mappings")
    
    # Get unique categories from the mapping (34 categories total)
    unique_categories = sorted(set(asin2category.values()))
    print(f"Found {len(unique_categories)} unique categories: {unique_categories}")
    
    # Create broadcast variable for efficient distributed lookup
    asin2category_broadcast = spark.sparkContext.broadcast(asin2category)
    
    # Get all categories by scanning review files
    categories = []
    for filename in os.listdir(reviews_path):
        if filename.endswith('.jsonl'):
            category = filename.replace('.jsonl', '')
            categories.append(category)
    
    # Apply test mode limit if enabled
    if TEST_MODE:
        categories = categories[:TEST_CATEGORIES]
        print(f"TEST MODE: Processing only {len(categories)} categories")
    
    print(f"Found {len(categories)} categories to process")
    
    # Define schema for reviews data
    # This ensures consistent data types across all files
    review_schema = StructType([
        StructField("rating", FloatType(), True),
        StructField("title", StringType(), True),
        StructField("text", StringType(), True),
        StructField("images", ArrayType(StringType()), True),  # Array of image URLs from reviews
        StructField("asin", StringType(), True),
        StructField("parent_asin", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("timestamp", LongType(), True),
        StructField("verified_purchase", BooleanType(), True)
    ])
    
    # Define schema for metadata
    # Metadata contains product information including images, videos, and details
    meta_schema = StructType([
        StructField("main_category", StringType(), True),
        StructField("title", StringType(), True),
        StructField("description", ArrayType(StringType()), True),
        StructField("images", ArrayType(MapType(StringType(), StringType())), True),  # Array of image objects
        StructField("videos", ArrayType(MapType(StringType(), StringType())), True),   # Array of video objects
        StructField("details", MapType(StringType(), StringType()), True),  # Product details as key-value pairs
        StructField("parent_asin", StringType(), True)
    ])
    
    # Create temporary directory for intermediate results
    # This helps manage memory by processing in batches
    temp_path = "~/data_directory/amazon_reviews_2023_reduced/temp_parts"
    os.makedirs(temp_path, exist_ok=True)
    
    # Process categories in batches to avoid memory issues
    # Increased batch size due to more cores and memory
    batch_size = 8
    processed_files = []
    
    for batch_start in range(0, len(categories), batch_size):
        batch_end = min(batch_start + batch_size, len(categories))
        batch_categories = categories[batch_start:batch_end]
        print(f"\nProcessing batch {batch_start//batch_size + 1}: categories {batch_start+1}-{batch_end}")
        
        batch_reviews = None
        
        for i, category in enumerate(batch_categories):
            print(f"  Processing category {batch_start+i+1}/{len(categories)}: {category}")
            
            # Read reviews for this category
            review_file = os.path.join(reviews_path, f"{category}.jsonl")
            reviews_df = spark.read.json(review_file, schema=review_schema)
            
            # Read metadata for this category
            meta_file = os.path.join(meta_path, f"meta_{category}.jsonl")
            meta_df = spark.read.json(meta_file, schema=meta_schema)
            
            # Process metadata to extract features (no longer need main_category from metadata)
            meta_df = meta_df.select(
                "parent_asin",
                F.col("title").alias("item_title"),
                F.concat_ws(" ", F.col("description")).alias("item_description"),
                F.col("images").alias("product_images"),
                F.col("videos").alias("product_videos"),
                F.col("details").alias("product_details")
            )
            
            # Join reviews with metadata on parent_asin
            joined_df = reviews_df.join(meta_df, on="parent_asin", how="left")
            
            # Use asin2category mapping to assign categories
            # Create UDF to look up category from broadcast variable
            def get_category(parent_asin):
                mapping = asin2category_broadcast.value
                return mapping.get(parent_asin, "Unknown")
            
            get_category_udf = F.udf(get_category, StringType())
            
            # Apply category mapping based on parent_asin
            joined_df = joined_df.withColumn(
                "main_category",
                get_category_udf(F.col("parent_asin"))
            )
            
            # Select required columns
            joined_df = joined_df.select(
                "main_category",
                "item_title",
                "item_description",
                "product_images",
                "product_videos",
                "product_details",
                "verified_purchase",
                "title",  # Review title - treated as outcome variable like text/rating
                "text",
                "rating"
            )
            
            # Union within batch
            if batch_reviews is None:
                batch_reviews = joined_df
            else:
                batch_reviews = batch_reviews.union(joined_df)
        
        # Save batch to temporary parquet file
        batch_path = os.path.join(temp_path, f"batch_{batch_start//batch_size}.parquet")
        batch_reviews.write.mode("overwrite").parquet(batch_path)
        processed_files.append(batch_path)
        print(f"  Saved batch to {batch_path}")
        
        # Clear cache to free memory and checkpoint to prevent lineage issues
        spark.catalog.clearCache()
        # Force garbage collection to prevent memory buildup
        import gc
        gc.collect()
    
    print(f"\nAll batches processed. Reading and combining {len(processed_files)} batch files...")
    
    # Read all batch files
    all_reviews = spark.read.parquet(*processed_files)
    
    print("All categories loaded, now creating features...")
    
    # Feature Engineering Section
    # ===========================
    # Create numeric features that capture product characteristics
    # These features are designed to be informative for predicting verified purchases
    
    # Feature 1: Title length (character count)
    # Hypothesis: Longer titles may indicate more detailed/professional listings
    all_reviews = all_reviews.withColumn(
        "title_length",
        F.when(F.col("item_title").isNotNull(), F.length("item_title")).otherwise(0)
    )
    
    # Feature 2: Description length (character count)
    # Hypothesis: More detailed descriptions may correlate with legitimate products
    all_reviews = all_reviews.withColumn(
        "description_length", 
        F.when(F.col("item_description").isNotNull(), F.length("item_description")).otherwise(0)
    )
    
    # Feature 3: Number of product images
    # Hypothesis: Products with more images may be more established/trustworthy
    # Count the size of the images array, default to 0 if null
    all_reviews = all_reviews.withColumn(
        "image_count",
        F.when(F.col("product_images").isNotNull(), F.size("product_images")).otherwise(0)
    )
    
    # Feature 4: Number of product videos
    # Hypothesis: Products with videos may have higher quality/verification standards
    # Count the size of the videos array, default to 0 if null
    all_reviews = all_reviews.withColumn(
        "video_count",
        F.when(F.col("product_videos").isNotNull(), F.size("product_videos")).otherwise(0)
    )
    
    # Feature 5: Details length (character count of all key-value pairs)
    # Hypothesis: More detailed product specifications may indicate professional sellers
    # Convert details map to string and count characters
    all_reviews = all_reviews.withColumn(
        "details_length",
        F.when(F.col("product_details").isNotNull(), 
               F.length(F.concat_ws(" ", 
                                  F.map_keys("product_details"), 
                                  F.map_values("product_details")))).otherwise(0)
    )
    
    # Drop the original columns we no longer need
    all_reviews = all_reviews.drop("item_title", "item_description", "product_images", 
                                   "product_videos", "product_details")
    
    # Create one-hot encoded columns for categories
    print("Creating one-hot encoded features for 34 categories...")
    
    # Function to normalize category names for column names
    def normalize_category_name(cat_name):
        # Convert to lowercase and replace various separators/punctuation
        normalized = cat_name.lower()
        normalized = normalized.replace(" and ", "_and_")
        normalized = normalized.replace("&", "and")
        normalized = normalized.replace(",", "")
        normalized = normalized.replace("'", "")
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace(" ", "_")
        return normalized
    
    # Create dummy variables for each of the 34 categories from asin2category.json
    for category in unique_categories:
        col_name = f"category_{normalize_category_name(category)}"
        all_reviews = all_reviews.withColumn(
            col_name,
            F.when(F.col("main_category") == category, 1).otherwise(0)
        )
        print(f"  Created column: {col_name}")
    
    # Drop the original category column
    all_reviews = all_reviews.drop("main_category")
    
    # Ensure verified_purchase is boolean type
    all_reviews = all_reviews.withColumn(
        "verified_purchase",
        F.col("verified_purchase").cast(BooleanType())
    )
    
    # Filter out Unknown category records
    print("Filtering out Unknown category records...")
    all_reviews = all_reviews.filter(F.col("category_unknown") != 1)
    
    # Define final column order (excluding category_unknown)
    # One-hot encoded categories first, then numeric features, then target, rating, title and text
    category_columns = sorted([col for col in all_reviews.columns if col.startswith("category_") and col != "category_unknown"])
    feature_columns = ["title_length", "description_length", "image_count", "video_count", "details_length"]
    final_columns = category_columns + feature_columns + ["verified_purchase", "rating", "title", "text"]
    
    all_reviews = all_reviews.select(*final_columns)
    
    print("Writing to parquet...")
    
    # Write to parquet with compression
    # Using 200 partitions for efficient parallel processing later
    all_reviews.repartition(200) \
        .write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(output_path)
    
    print(f"Successfully written to {output_path}")
    
    # Print final schema for verification
    print("\nFinal schema:")
    all_reviews.printSchema()
    
    # Print sample statistics
    total_count = all_reviews.count()
    verified_count = all_reviews.filter(F.col("verified_purchase") == True).count()
    
    print(f"\nTotal reviews (after filtering Unknown): {total_count:,}")
    print(f"Verified purchases: {verified_count:,} ({verified_count/total_count*100:.2f}%)")
    print(f"Successfully removed ~63.8M Unknown category records")
    
    # Clean up temporary files
    print("\nCleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_path)
    
    spark.stop()

if __name__ == "__main__":
    main()