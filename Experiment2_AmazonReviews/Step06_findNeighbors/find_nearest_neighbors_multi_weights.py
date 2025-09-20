#!/usr/bin/env python3
"""
GPU-accelerated version of find_nearest_neighbors for Amazon reviews data with multiple AIPW weight variants.

This script computes complete AIPW (Augmented Inverse Propensity Weighting) weights 
by combining IPW values with augmentation terms. It uses GPU acceleration for 
significantly faster computation on large datasets.

Weight Types Computed:
1. gcomp_right: Standard k-NN based G-computation weight
2. gcomp_wrong: Uniform weighting across all eligible matches
3. aipw_ipw_right_out_right: ipw_right + augmentation from k-NN matching
4. aipw_ipw_right_out_wrong: ipw_right + augmentation from uniform weighting
5. aipw_ipw_wrong_out_right: ipw_wrong + augmentation from k-NN matching
6. aipw_ipw_wrong_out_wrong: ipw_wrong + augmentation from uniform weighting

The "out_right" variants use the computationally intensive k-NN search.
The "out_wrong" variants use simple uniform weighting for much faster computation.

AIPW formula: aipw = ipw + augmentation_term
This creates doubly robust estimators that combine the benefits of both IPW and outcome modeling.
"""

import argparse
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import faiss
from collections import defaultdict
import os
import shutil

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    print("WARNING: CuPy not available. GPU augmentation weights disabled.", flush=True)
    CUPY_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description='Find nearest neighbors with multiple weight variants')
    parser.add_argument('--training_set', type=str, required=True, choices=['train1', 'train2'],
                        help='Which training set to process (train1 or train2)')
    parser.add_argument('--matching_set', type=str, required=True, choices=['train1', 'train2'],
                        help='Which set to use as matching candidates')
    parser.add_argument('--k', type=int, default=200,
                        help='Number of nearest neighbors to find')
    parser.add_argument('--memory', type=str, default='400g',
                        help='Spark executor memory')
    parser.add_argument('--cores', type=int, default=40,
                        help='Number of cores to use')
    parser.add_argument('--gpus', type=int, default=4,
                        help='Number of GPUs to use')
    return parser.parse_args()

def initialize_spark(memory, cores):
    """Initialize Spark session with appropriate settings."""
    spark = SparkSession.builder \
        .appName(f"FindNearestNeighborsMultiWeights") \
        .config("spark.driver.memory", memory) \
        .config("spark.executor.memory", memory) \
        .config("spark.driver.maxResultSize", "32g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", str(cores * 4)) \
        .config("spark.default.parallelism", str(cores * 2)) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
        .config("spark.sql.files.maxRecordsPerFile", "1000000") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.shuffle.service.enabled", "false") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.rpc.askTimeout", "600s") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def get_baseline_features():
    """Get list of baseline features (excluding categories)"""
    return [
        'title_length', 'description_length', 'image_count',
        'video_count', 'details_length'
    ]

def get_category_columns(df):
    """Get all category columns from the dataframe"""
    return [col for col in df.columns if col.startswith('category_')]

def compute_uniform_weights(train_df, match_df, ipw_col):
    """
    Compute uniform weights for the "out_wrong" strategy.
    
    Returns:
    - uniform_augmentation: (sum of (1-ipw) in training) / (count of synthetic=1 in matching)
    - uniform_gcomp: (count in training) / (count of synthetic=1 in matching)
    """
    # Get training statistics
    train_stats = train_df.agg(
        F.count("*").alias("train_count"),
        F.sum(F.lit(1.0) - F.col(ipw_col)).alias("sum_one_minus_ipw")
    ).collect()[0]
    
    train_count = train_stats["train_count"]
    sum_one_minus_ipw = float(train_stats["sum_one_minus_ipw"] or 0.0)
    
    # Get matching statistics (only synthetic=1)
    match_synthetic_count = match_df.filter(F.col("verified_purchase_synthetic") == 1).count()
    
    if match_synthetic_count == 0:
        return 0.0, 0.0
    
    uniform_augmentation = sum_one_minus_ipw / match_synthetic_count
    uniform_gcomp = float(train_count) / match_synthetic_count
    
    return uniform_augmentation, uniform_gcomp

def build_augmentation_weights_gpu_chunked(indices, train_ipw, match_ids, k_actual):
    """
    GPU-accelerated augmentation weight calculation using CuPy with chunking.
    This version processes data in chunks to avoid GPU OOM errors on large datasets.
    
    Returns:
        Tuple of two dictionaries:
        - augmentation_weights: mapping match_ids to augmentation weights (1/k * sum(1-ipw))
        - gcomp_right: mapping match_ids to neighbor counts/k (1/k * count)
    """
    print(f"    Using GPU for augmentation weight calculation (chunked)...", flush=True)
    start_time = time.time()
    
    n_match = len(match_ids)
    n_train = len(train_ipw)
    
    print(f"    Input sizes: {n_train:,} training samples, {n_match:,} matching samples, k={k_actual}", flush=True)
    
    # Initialize CPU arrays for accumulating results
    weights_cpu = np.zeros(n_match, dtype=np.float64)
    gcomp_right_cpu = np.zeros(n_match, dtype=np.float64)
    
    # Determine chunk size based on available GPU memory
    gpu_memory_gb = 40.0
    bytes_per_element = 4  # float32
    
    memory_per_sample = (k_actual + 2) * bytes_per_element
    fixed_memory = n_match * bytes_per_element
    fixed_memory_gb = fixed_memory / (1024**3)
    available_memory = (gpu_memory_gb * (1024**3)) - fixed_memory
    
    max_train_per_chunk = int(available_memory / memory_per_sample)
    chunk_size = min(max_train_per_chunk, 5000000)  # Max 5M training samples per chunk
    
    print(f"    Processing {n_train:,} training samples in chunks of {chunk_size:,}", flush=True)
    
    # Calculate contributions once on CPU
    contributions = (1 - train_ipw) / k_actual
    
    # Get GPU and initialize weights array
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    try:
        # Process in chunks
        num_chunks = (n_train + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_train)
            chunk_train_size = end_idx - start_idx
            
            if num_chunks > 5 or chunk_idx % 5 == 0:
                print(f"      Processing chunk {chunk_idx + 1}/{num_chunks} (samples {start_idx:,}-{end_idx:,})", flush=True)
            
            # Get chunk data
            chunk_indices = indices[start_idx:end_idx]
            chunk_contributions = contributions[start_idx:end_idx]
            
            # Transfer chunk to GPU
            indices_gpu = cp.asarray(chunk_indices, dtype=cp.int32)
            contributions_gpu = cp.asarray(chunk_contributions, dtype=cp.float32)
            
            # Flatten and repeat for this chunk
            flat_indices = indices_gpu.ravel()
            repeated_contributions = cp.repeat(contributions_gpu, k_actual)
            
            # Filter valid indices
            valid_mask = (flat_indices >= 0) & (flat_indices < n_match)
            valid_indices = flat_indices[valid_mask]
            valid_contributions = repeated_contributions[valid_mask]
            
            # Create temporary arrays for this chunk
            weights_gpu_chunk = cp.zeros(n_match, dtype=cp.float32)
            gcomp_right_gpu_chunk = cp.zeros(n_match, dtype=cp.float32)
            
            # Create array of 1/k_actual for gcomp_right calculation
            ones_over_k = cp.full(len(valid_indices), 1.0 / k_actual, dtype=cp.float32)
            
            # Use scatter_add for both calculations
            import cupyx
            cupyx.scatter_add(weights_gpu_chunk, valid_indices, valid_contributions)
            cupyx.scatter_add(gcomp_right_gpu_chunk, valid_indices, ones_over_k)
            
            # Accumulate to CPU arrays
            weights_cpu += cp.asnumpy(weights_gpu_chunk)
            gcomp_right_cpu += cp.asnumpy(gcomp_right_gpu_chunk)
            
            # Free GPU memory for this chunk
            del indices_gpu, contributions_gpu, flat_indices
            del repeated_contributions, valid_indices, valid_contributions
            del weights_gpu_chunk, gcomp_right_gpu_chunk, ones_over_k
            
            # Force memory cleanup
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
    
    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"    GPU out of memory! Error: {e}", flush=True)
        print(f"    Falling back to CPU implementation.", flush=True)
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return build_augmentation_weights_cpu_vectorized(indices, train_ipw, match_ids, k_actual)
    
    # Convert to dictionary format
    non_zero_indices = np.where(weights_cpu > 0)[0]
    augmentation_weights = {match_ids[idx]: float(weights_cpu[idx]) for idx in non_zero_indices}
    
    non_zero_gcomp = np.where(gcomp_right_cpu > 0)[0]
    gcomp_right = {match_ids[idx]: float(gcomp_right_cpu[idx]) for idx in non_zero_gcomp}
    
    elapsed = time.time() - start_time
    print(f"    GPU augmentation completed in {elapsed:.2f} seconds", flush=True)
    
    return augmentation_weights, gcomp_right

def build_augmentation_weights_cpu_vectorized(indices, train_ipw, match_ids, k_actual):
    """
    CPU vectorized fallback for augmentation weight calculation.
    
    Returns:
        Tuple of two dictionaries:
        - augmentation_weights: mapping match_ids to augmentation weights
        - gcomp_right: mapping match_ids to neighbor counts/k
    """
    print(f"    Using CPU vectorized implementation...", flush=True)
    start_time = time.time()
    
    n_match = len(match_ids)
    weights = np.zeros(n_match, dtype=np.float64)
    gcomp_right_counts = np.zeros(n_match, dtype=np.float64)
    
    # Calculate all contributions at once
    contributions = (1 - train_ipw) / k_actual
    
    # Process in chunks to be cache-friendly
    chunk_size = 100000
    for i in range(0, len(train_ipw), chunk_size):
        end_idx = min(i + chunk_size, len(train_ipw))
        
        # Get chunk of indices and contributions
        chunk_indices = indices[i:end_idx].ravel()
        chunk_contributions = np.repeat(contributions[i:end_idx], k_actual)
        
        # Filter valid indices
        valid_mask = (chunk_indices >= 0) & (chunk_indices < n_match)
        valid_indices = chunk_indices[valid_mask]
        valid_contributions = chunk_contributions[valid_mask]
        
        # Atomic add for weights
        np.add.at(weights, valid_indices, valid_contributions)
        
        # Atomic add for gcomp_right (count/k)
        ones_over_k = np.full(len(valid_indices), 1.0 / k_actual)
        np.add.at(gcomp_right_counts, valid_indices, ones_over_k)
    
    # Convert to dictionary
    non_zero_indices = np.where(weights > 0)[0]
    augmentation_weights = {match_ids[idx]: float(weights[idx]) for idx in non_zero_indices}
    
    non_zero_gcomp = np.where(gcomp_right_counts > 0)[0]
    gcomp_right = {match_ids[idx]: float(gcomp_right_counts[idx]) for idx in non_zero_gcomp}
    
    elapsed = time.time() - start_time
    print(f"    CPU augmentation completed in {elapsed:.2f} seconds", flush=True)
    
    return augmentation_weights, gcomp_right

def standardize_features(df, features):
    """
    Standardize features to mean 0, variance 1 within the category
    """
    for feature in features:
        # Calculate mean and stddev
        stats = df.select(
            F.mean(F.col(feature)).alias('mean'),
            F.stddev(F.col(feature)).alias('std')
        ).collect()[0]
        
        mean_val = stats['mean'] or 0.0
        std_val = stats['std'] or 1.0
        
        # Avoid division by zero
        if std_val == 0:
            std_val = 1.0
            
        # Standardize
        df = df.withColumn(
            f"{feature}_scaled",
            (F.col(feature) - mean_val) / std_val
        )
    
    return df

def create_gpu_index(d, n_vectors, use_flat=True, ngpus=1):
    """Create a GPU FAISS index"""
    if ngpus > 1:
        print(f"    Creating multi-GPU index with {ngpus} GPUs", flush=True)
        # Multi-GPU setup
        gpu_resources = []
        for i in range(ngpus):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory per GPU
            gpu_resources.append(res)
        
        # Create CPU index first
        if use_flat or n_vectors < 50000:
            cpu_index = faiss.IndexFlatL2(d)
        else:
            nlist = min(int(4 * np.sqrt(n_vectors)), 16384)
            cpu_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
        
        # Convert to multi-GPU
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # Shard the index across GPUs
        co.useFloat16 = True  # Use float16 for memory efficiency
        
        index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, cpu_index, co, range(ngpus))
        
        # Train if IVF
        if not use_flat and n_vectors >= 50000:
            print(f"    Training multi-GPU IVF index with nlist={nlist}", flush=True)
            # For multi-GPU, training happens after adding vectors
            index.nprobe = min(index.nlist // 4, 128)
    else:
        # Single GPU
        res = faiss.StandardGpuResources()
        res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory
        
        if use_flat or n_vectors < 50000:
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            config.useFloat16 = False  # Use full precision for flat index
            index = faiss.GpuIndexFlatL2(res, d, config)
        else:
            nlist = min(int(4 * np.sqrt(n_vectors)), 16384)
            config = faiss.GpuIndexIVFFlatConfig()
            config.device = 0
            config.useFloat16CoarseQuantizer = False
            config.useFloat16 = True
            config.interleavedLayout = True
            
            index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
            index.nprobe = min(nlist // 4, 128)
    
    return index

def process_category_knn_both(train_cat, match_cat, match_cat_filtered, category, baseline_features, 
                              k, cores, gpus):
    """
    Process a single category using k-NN search for out_right variants.
    Returns indices and BOTH IPW columns for augmentation weight calculation.
    This runs k-NN only once since neighbors are the same regardless of IPW column.
    """
    print(f"  Processing {category} with k-NN search...", flush=True)
    
    # Standardize features within category
    train_cat = standardize_features(train_cat, baseline_features)
    match_cat_filtered = standardize_features(match_cat_filtered, baseline_features)
    
    # Prepare features for FAISS
    scaled_features = [f"{col}_scaled" for col in baseline_features]
    
    # Collect training data - get BOTH IPW columns
    train_data = train_cat.select(
        "_unique_id",
        "ipw_right",
        "ipw_wrong",
        *scaled_features
    ).collect()
    
    if not train_data:
        return None, None, None, None
        
    train_ids = np.array([row["_unique_id"] for row in train_data])
    train_ipw_right = np.array([float(row["ipw_right"]) for row in train_data], dtype=np.float32)
    train_ipw_wrong = np.array([float(row["ipw_wrong"]) for row in train_data], dtype=np.float32)
    train_features = np.array([[row[col] for col in scaled_features] for row in train_data], dtype=np.float32)
    
    # Collect matching features
    match_data = match_cat_filtered.select(
        "_unique_id",
        *scaled_features
    ).collect()
    
    if not match_data:
        return None, None, None, None
        
    match_ids = np.array([row["_unique_id"] for row in match_data])
    match_features = np.array([[row[col] for col in scaled_features] for row in match_data], dtype=np.float32)
    
    d = match_features.shape[1]
    print(f"    Data collected. Train: {len(train_features):,}, Match: {len(match_features):,}", flush=True)
    
    # Use GPU for larger categories, CPU for small ones
    use_gpu_nn = len(match_features) >= 50000
    
    if not use_gpu_nn:
        print(f"    Using CPU for small category", flush=True)
        faiss.omp_set_num_threads(cores)
        if len(match_features) < 5000:
            index = faiss.IndexFlatL2(d)
        else:
            nlist = min(int(np.sqrt(len(match_features))), 100)
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.train(match_features)
            index.nprobe = min(index.nlist, 10)
        index.add(match_features)
    else:
        # Build GPU FAISS index
        print(f"    Using GPU for this category", flush=True)
        try:
            num_gpus = min(gpus, 4) if len(match_features) > 1000000 else 1
            index = create_gpu_index(d, len(match_features), use_flat=True, ngpus=num_gpus)
            
            print(f"    Adding vectors to GPU index...", flush=True)
            if num_gpus > 1:
                index.add(match_features)
            else:
                chunk_size = 100000
                for i in range(0, len(match_features), chunk_size):
                    end_idx = min(i + chunk_size, len(match_features))
                    index.add(match_features[i:end_idx])
                    
        except Exception as e:
            print(f"    GPU failed: {str(e)}, falling back to CPU", flush=True)
            faiss.omp_set_num_threads(cores)
            if len(match_features) < 5000:
                index = faiss.IndexFlatL2(d)
            else:
                nlist = min(int(np.sqrt(len(match_features))), 100)
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                index.train(match_features)
                index.nprobe = min(index.nlist, 10)
            index.add(match_features)
            use_gpu_nn = False
    
    # Search for nearest neighbors
    device_type = "GPU" if use_gpu_nn else "CPU"
    print(f"    Searching for {k} nearest neighbors on {device_type}...", flush=True)
    k_actual = min(k, len(match_features))
    
    # Process in batches
    batch_size = 500000 if use_gpu_nn else 50000
    all_indices = []
    
    for i in range(0, len(train_features), batch_size):
        end_idx = min(i + batch_size, len(train_features))
        batch_features = train_features[i:end_idx]
        
        distances, indices = index.search(batch_features, k_actual)
        all_indices.append(indices)
    
    # Concatenate all indices
    all_indices = np.vstack(all_indices) if len(all_indices) > 1 else all_indices[0]
    
    return all_indices, train_ipw_right, train_ipw_wrong, match_ids

def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"Finding Nearest Neighbors with Multiple Weight Variants")
    print(f"{'='*80}")
    print(f"Training set: {args.training_set}")
    print(f"Matching set: {args.matching_set}") 
    print(f"k (neighbors): {args.k}")
    print(f"Memory: {args.memory}")
    print(f"Cores: {args.cores}")
    print(f"GPUs: {args.gpus}")
    print(f"{'='*80}\n")
    
    # Initialize Spark
    spark = initialize_spark(args.memory, args.cores)
    
    # Define paths
    base_path = "~/data_directory/amazon_reviews_2023_inverse_propensity/"
    training_path = f"{base_path}training_set_{'1' if args.training_set == 'train1' else '2'}/"
    matching_path = f"{base_path}training_set_{'1' if args.matching_set == 'train1' else '2'}/"
    
    # Output path
    output_base = "~/data_directory/amazon_reviews_2023_augmented_multi/"
    output_path = f"{output_base}{args.training_set}_with_aug_from_{args.matching_set}/"
    temp_output_dir = f"{output_path}_temp"
    
    print(f"Training data path: {training_path}")
    print(f"Matching data path: {matching_path}")
    print(f"Output path: {output_path}\n")
    
    # Load data
    print("Loading training data...")
    train_df = spark.read.parquet(training_path)
    train_count = train_df.count()
    print(f"Training set size: {train_count:,}")
    
    print("\nLoading matching data...")
    match_df = spark.read.parquet(matching_path)
    match_count = match_df.count()
    print(f"Matching set size: {match_count:,}")
    
    # Filter matching set to synthetic=1 for nearest neighbor search
    match_df_filtered = match_df.filter(F.col("verified_purchase_synthetic") == 1)
    match_filtered_count = match_df_filtered.count()
    print(f"Matching set (synthetic=1): {match_filtered_count:,}")
    
    # Add unique IDs
    train_df = train_df.withColumn("_unique_id", F.monotonically_increasing_id())
    match_df = match_df.withColumn("_unique_id", F.monotonically_increasing_id())
    match_df_filtered = match_df_filtered.withColumn("_unique_id", F.monotonically_increasing_id())
    
    # Cache DataFrames
    train_df.cache()
    match_df.cache()
    match_df_filtered.cache()
    train_df.count()
    match_df.count()
    match_df_filtered.count()
    
    # Get features
    baseline_features = get_baseline_features()
    category_cols = get_category_columns(train_df)
    categories = []
    for col in category_cols:
        cat_name = col.replace('category_', '')
        if not cat_name:
            cat_name = "unknown"
        categories.append(cat_name)
    
    print(f"\nProcessing {len(categories)} categories...")
    print(f"Will compute 6 weight columns: gcomp_right, gcomp_wrong, and 4 AIPW variants\n")
    
    # Initialize weight dictionaries for all variants
    # We'll store augmentation weights temporarily but only output AIPW weights
    all_weights = {
        'gcomp_right': {},
        'gcomp_wrong': {},
        'aipw_ipw_right_out_right': {},
        'aipw_ipw_right_out_wrong': {},
        'aipw_ipw_wrong_out_right': {},
        'aipw_ipw_wrong_out_wrong': {}
    }
    
    # Also track augmentation weights temporarily for computation
    augmentation_weights = {
        'aug_ipw_right_out_right': {},
        'aug_ipw_right_out_wrong': {},
        'aug_ipw_wrong_out_right': {},
        'aug_ipw_wrong_out_wrong': {}
    }
    
    for idx, category in enumerate(categories):
        print(f"\n[{idx+1}/{len(categories)}] Processing category: {category}")
        category_start = time.time()
        
        # Filter by category
        if category == "unknown":
            category_col = "category_"
        else:
            category_col = f"category_{category}"
            
        train_cat = train_df.filter(F.col(category_col) == 1)
        match_cat = match_df.filter(F.col(category_col) == 1)
        match_cat_filtered = match_df_filtered.filter(F.col(category_col) == 1)
        
        # Check counts
        train_count = train_cat.count()
        match_count_full = match_cat.count()
        match_count_filtered = match_cat_filtered.count()
        
        print(f"  Train: {train_count:,}, Match (full): {match_count_full:,}, Match (synthetic=1): {match_count_filtered:,}")
        
        if train_count == 0 or match_count_filtered == 0:
            print(f"  Skipping - insufficient data")
            continue
        
        # Compute uniform weights for "out_wrong" variants
        print(f"  Computing uniform weights for out_wrong variants...")
        
        # For ipw_right
        uniform_aug_right, uniform_gcomp = compute_uniform_weights(train_cat, match_cat, "ipw_right")
        print(f"    ipw_right: uniform_aug={uniform_aug_right:.6f}, uniform_gcomp={uniform_gcomp:.6f}")
        
        # For ipw_wrong  
        uniform_aug_wrong, _ = compute_uniform_weights(train_cat, match_cat, "ipw_wrong")
        print(f"    ipw_wrong: uniform_aug={uniform_aug_wrong:.6f}")
        
        # Store uniform weights for all synthetic=1 matches in this category
        match_synthetic_ids = match_cat_filtered.select("_unique_id").collect()
        for row in match_synthetic_ids:
            uid = row["_unique_id"]
            all_weights['gcomp_wrong'][uid] = uniform_gcomp
            augmentation_weights['aug_ipw_right_out_wrong'][uid] = uniform_aug_right
            augmentation_weights['aug_ipw_wrong_out_wrong'][uid] = uniform_aug_wrong
        
        # Process k-NN ONCE for "out_right" variants (same neighbors for both IPW columns)
        print(f"  Computing k-NN search (once for all out_right variants)...")
        
        # Run k-NN search just once - neighbors are the same regardless of IPW column
        indices, ipw_right_vals, ipw_wrong_vals, match_ids = process_category_knn_both(
            train_cat, match_cat, match_cat_filtered, category, 
            baseline_features, args.k, args.cores, args.gpus
        )
        
        if indices is not None:
            # Build augmentation weights for ipw_right
            print(f"    Computing augmentation weights for ipw_right...")
            if CUPY_AVAILABLE and len(match_ids) > 10000:
                aug_right, gcomp_right = build_augmentation_weights_gpu_chunked(
                    indices, ipw_right_vals, match_ids, min(args.k, len(match_ids))
                )
            else:
                aug_right, gcomp_right = build_augmentation_weights_cpu_vectorized(
                    indices, ipw_right_vals, match_ids, min(args.k, len(match_ids))
                )
            
            # Store results
            augmentation_weights['aug_ipw_right_out_right'].update(aug_right)
            all_weights['gcomp_right'].update(gcomp_right)
            
            # Build augmentation weights for ipw_wrong (reusing same indices)
            print(f"    Computing augmentation weights for ipw_wrong...")
            if CUPY_AVAILABLE and len(match_ids) > 10000:
                aug_wrong, _ = build_augmentation_weights_gpu_chunked(
                    indices, ipw_wrong_vals, match_ids, min(args.k, len(match_ids))
                )
            else:
                aug_wrong, _ = build_augmentation_weights_cpu_vectorized(
                    indices, ipw_wrong_vals, match_ids, min(args.k, len(match_ids))
                )
            
            # Store results (gcomp_right already computed above)
            augmentation_weights['aug_ipw_wrong_out_right'].update(aug_wrong)
        
        category_time = time.time() - category_start
        print(f"  Category completed in {category_time:.2f} seconds")
    
    # Compute AIPW weights by combining IPW with augmentation weights
    print(f"\nComputing AIPW weights (IPW + augmentation)...")
    
    # First, we need to get the IPW values for each observation
    # Load just the IPW columns from the matching set
    ipw_df = match_df.select("_unique_id", "ipw_right", "ipw_wrong")
    ipw_data = ipw_df.collect()
    ipw_dict = {row["_unique_id"]: (float(row["ipw_right"]), float(row["ipw_wrong"])) for row in ipw_data}
    
    # Now compute AIPW weights by adding IPW to augmentation weights
    for uid in match_df.select("_unique_id").distinct().collect():
        uid_val = uid["_unique_id"]
        ipw_right, ipw_wrong = ipw_dict.get(uid_val, (0.0, 0.0))
        
        # AIPW = IPW + Augmentation
        # For out_right variants
        if uid_val in augmentation_weights['aug_ipw_right_out_right']:
            all_weights['aipw_ipw_right_out_right'][uid_val] = ipw_right + augmentation_weights['aug_ipw_right_out_right'][uid_val]
        else:
            all_weights['aipw_ipw_right_out_right'][uid_val] = ipw_right
            
        if uid_val in augmentation_weights['aug_ipw_wrong_out_right']:
            all_weights['aipw_ipw_wrong_out_right'][uid_val] = ipw_wrong + augmentation_weights['aug_ipw_wrong_out_right'][uid_val]
        else:
            all_weights['aipw_ipw_wrong_out_right'][uid_val] = ipw_wrong
        
        # For out_wrong variants
        if uid_val in augmentation_weights['aug_ipw_right_out_wrong']:
            all_weights['aipw_ipw_right_out_wrong'][uid_val] = ipw_right + augmentation_weights['aug_ipw_right_out_wrong'][uid_val]
        else:
            all_weights['aipw_ipw_right_out_wrong'][uid_val] = ipw_right
            
        if uid_val in augmentation_weights['aug_ipw_wrong_out_wrong']:
            all_weights['aipw_ipw_wrong_out_wrong'][uid_val] = ipw_wrong + augmentation_weights['aug_ipw_wrong_out_wrong'][uid_val]
        else:
            all_weights['aipw_ipw_wrong_out_wrong'][uid_val] = ipw_wrong
    
    # Create final DataFrame with AIPW and gcomp weights only
    print(f"\nCreating final output with AIPW and gcomp weight columns...")
    
    # Convert dictionaries to DataFrames using parallelized RDDs for better performance
    weight_dfs = []
    for weight_name, weight_dict in all_weights.items():
        if weight_dict:
            print(f"  Creating DataFrame for {weight_name} ({len(weight_dict):,} records)...")
            # Convert numpy int64 to Python int to avoid PySpark type issues
            weight_data = [(int(k), float(v)) for k, v in weight_dict.items()]
            # Parallelize the data creation across workers
            weight_rdd = spark.sparkContext.parallelize(weight_data, numSlices=400)
            weight_df = weight_rdd.toDF(["_unique_id", weight_name])
            weight_dfs.append(weight_df)
    
    # Join all weights to the match DataFrame
    # First combine all weight DataFrames into one to minimize joins
    print(f"  Combining weight DataFrames...")
    if weight_dfs:
        combined_weights_df = weight_dfs[0]
        for weight_df in weight_dfs[1:]:
            combined_weights_df = combined_weights_df.join(weight_df, on="_unique_id", how="outer")
        
        print(f"  Joining weights to match DataFrame...")
        result_df = match_df.join(combined_weights_df, on="_unique_id", how="left")
    else:
        result_df = match_df
    
    # Fill nulls with 0 for gcomp weights, but for AIPW weights we need the original IPW value
    result_df = result_df.withColumn("gcomp_right", F.coalesce(F.col("gcomp_right"), F.lit(0.0)))
    result_df = result_df.withColumn("gcomp_wrong", F.coalesce(F.col("gcomp_wrong"), F.lit(0.0)))
    
    # For AIPW columns, null means no augmentation was added, so it should be just the IPW value
    result_df = result_df.withColumn("aipw_ipw_right_out_right", 
                                     F.coalesce(F.col("aipw_ipw_right_out_right"), F.col("ipw_right")))
    result_df = result_df.withColumn("aipw_ipw_right_out_wrong", 
                                     F.coalesce(F.col("aipw_ipw_right_out_wrong"), F.col("ipw_right")))
    result_df = result_df.withColumn("aipw_ipw_wrong_out_right", 
                                     F.coalesce(F.col("aipw_ipw_wrong_out_right"), F.col("ipw_wrong")))
    result_df = result_df.withColumn("aipw_ipw_wrong_out_wrong", 
                                     F.coalesce(F.col("aipw_ipw_wrong_out_wrong"), F.col("ipw_wrong")))
    
    # Drop the temporary unique ID
    result_df = result_df.drop("_unique_id")
    
    # Save results
    print(f"Saving results to {output_path}...")
    # Repartition for better write parallelism
    result_df.repartition(200).write.mode("overwrite").parquet(output_path)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    stats = result_df.agg(
        *[F.mean(col).alias(f"mean_{col}") for col in all_weights.keys()],
        *[F.stddev(col).alias(f"std_{col}") for col in all_weights.keys()],
        *[F.sum(F.when(F.col(col) > 0, 1).otherwise(0)).alias(f"nonzero_{col}") for col in all_weights.keys()]
    ).collect()[0]
    
    for weight_name in all_weights.keys():
        print(f"\n{weight_name}:")
        print(f"  Mean: {stats[f'mean_{weight_name}']:.6f}")
        print(f"  Std: {stats[f'std_{weight_name}']:.6f}")
        print(f"  Non-zero count: {stats[f'nonzero_{weight_name}']:,}")
    
    print(f"\nProcessing complete!")
    spark.stop()

if __name__ == "__main__":
    main()