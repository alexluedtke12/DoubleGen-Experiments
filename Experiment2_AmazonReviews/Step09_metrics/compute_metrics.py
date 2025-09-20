#!/usr/bin/env python3
"""
Step09: Compute Metrics for Generated Samples

This script evaluates the quality of generated samples from Step08 using three metrics:
1. Perplexity - Using the trained models to evaluate test set
2. MAUVE - Distribution similarity between generated and test samples  
3. Wasserstein Distance - For star rating distributions

Author: Claude
Date: 2025-08-18
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import torch
from tqdm import tqdm
import scipy.stats
import pyarrow.parquet as pq
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from functools import partial

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ CAUTION: TESTING MODE - REMOVE FOR PRODUCTION ============
# Set TEST_MODE to False for production runs with larger sample sizes
TEST_MODE = False
if TEST_MODE:
    TEST_SAMPLES_FOR_PERPLEXITY = 1_000    # Quick test with 1K test samples
    TEST_SAMPLES_FOR_MAUVE = 1_000         # Quick test with 1K test samples
    GENERATED_SAMPLES_TO_USE = 1_000       # Use only 1K generated samples
    print("⚠️  RUNNING IN TEST MODE - Using 1K test samples and 1K generated samples")
else:
    TEST_SAMPLES_FOR_PERPLEXITY = 20_000   # Production: 20K test samples
    TEST_SAMPLES_FOR_MAUVE = 20_000        # Production: 20K test samples for MAUVE
    GENERATED_SAMPLES_TO_USE = None         # Use all generated samples (10K)
    print("🚀 RUNNING IN PRODUCTION MODE - Using full datasets")
# ============ END CAUTION: TESTING MODE ============

# Paths
BASE_DIR = Path("/n/scratch/users/a/your_username")
TEST_SET_DIR = BASE_DIR / "amazon_reviews_2023_starred" / "test_set"
MODELS_DIR = BASE_DIR / "models_finetuned_all_weights"
OUTPUT_DIR = Path("/home/your_username/cf_reviews/Step09_metrics/metrics_results")

# Model names to evaluate
MODEL_NAMES = [
    'naive', 'ipw_right', 'ipw_wrong', 'gcomp_right', 'gcomp_wrong',
    'aipw_ipw_right_out_right', 'aipw_ipw_right_out_wrong',
    'aipw_ipw_wrong_out_right', 'aipw_ipw_wrong_out_wrong'
]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")


def load_test_set_samples(num_samples_perplexity: int, num_samples_mauve: int) -> Tuple[List[str], List[str], pd.DataFrame, np.ndarray, Dict]:
    """
    Load and process test set samples.
    
    Returns:
        - test_reviews_perplexity: Sample for perplexity computation
        - test_reviews_mauve: Sample for MAUVE computation
        - test_ratings: Full rating array from entire test set
        - test_rating_pmf: PMF of ratings from entire test set
    """
    print("\n" + "="*80)
    print("LOADING TEST SET")
    print("="*80)
    
    # Get all parquet files in test set directory
    parquet_files = sorted(TEST_SET_DIR.glob("part-*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in test set")
    
    # Load data for sampling - keep all columns for detailed output
    all_dfs = []
    all_ratings = []
    
    print("Loading test set data...")
    for file in tqdm(parquet_files, desc="Reading files"):
        df = pq.read_table(file).to_pandas()
        all_dfs.append(df)
        all_ratings.extend(df['rating'].tolist())
    
    # Combine all dataframes
    full_df = pd.concat(all_dfs, ignore_index=True)
    all_reviews = full_df['text'].tolist()
    
    print(f"Total test samples loaded: {len(all_reviews):,}", flush=True)
    
    # Convert ratings to numpy array for efficient computation
    all_ratings = np.array(all_ratings)
    
    # Compute rating PMF from full test set
    rating_counts = np.bincount(all_ratings.astype(int), minlength=6)[1:6]  # Ratings 1-5
    test_rating_pmf = rating_counts / rating_counts.sum()
    
    print("\nTest set rating distribution:")
    for i, prob in enumerate(test_rating_pmf, 1):
        count = rating_counts[i-1]
        print(f"  {i} star{'s' if i != 1 else ''}: {prob:.4f} ({count:,} samples)")
    
    # Sample for perplexity and MAUVE
    random.seed(42)  # For reproducibility
    indices = list(range(len(all_reviews)))
    random.shuffle(indices)
    
    perplexity_indices = indices[:num_samples_perplexity]
    test_reviews_perplexity = [all_reviews[i] for i in perplexity_indices]
    test_reviews_mauve = [all_reviews[i] for i in indices[:num_samples_mauve]]
    
    # Get full dataframe for perplexity samples
    test_df_perplexity = full_df.iloc[perplexity_indices].reset_index(drop=True)
    
    print(f"\nSampled {len(test_reviews_perplexity):,} reviews for perplexity", flush=True)
    print(f"Sampled {len(test_reviews_mauve):,} reviews for MAUVE", flush=True)
    
    # Create rating distribution dict
    test_rating_dist = {
        str(i): float(prob) for i, prob in enumerate(test_rating_pmf, 1)
    }
    
    return test_reviews_perplexity, test_reviews_mauve, test_df_perplexity, all_ratings, test_rating_dist


def compute_rating_pmf(ratings: List[int]) -> np.ndarray:
    """Compute probability mass function for ratings 1-5."""
    counts = np.bincount(ratings, minlength=6)[1:6]  # Ignore 0, get 1-5
    return counts / counts.sum() if counts.sum() > 0 else counts


def extract_rating_from_text(text: str) -> Optional[int]:
    """Extract rating from generated text (first character should be 1-5)."""
    if text and text[0].isdigit():
        rating = int(text[0])
        if 1 <= rating <= 5:
            return rating
    return None


def load_generated_samples(model_name: str) -> Tuple[List[str], Dict]:
    """Load generated samples for a model."""
    sample_path = MODELS_DIR / model_name / "generated_samples_unconditional.json"
    
    if not sample_path.exists():
        print(f"⚠️  Generated samples not found for {model_name}")
        return [], {}
    
    with open(sample_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    
    # Extract ratings from samples
    ratings = []
    for sample in samples:
        rating = extract_rating_from_text(sample)
        if rating:
            ratings.append(rating)
    
    # Compute rating distribution
    rating_pmf = compute_rating_pmf(ratings)
    rating_dist = {str(i): float(prob) for i, prob in enumerate(rating_pmf, 1)}
    
    print(f"  Loaded {len(samples):,} generated samples")
    print(f"  Successfully extracted {len(ratings):,} ratings")
    
    return samples, rating_dist


def merge_lora_if_needed(model_name: str) -> Path:
    """
    Merge LoRA adapter with base model and save to merged_model directory.
    Always overwrites existing merged model.
    
    Returns:
        Path to the merged model directory
    """
    print(f"    Merging LoRA adapter for {model_name}...", flush=True)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import shutil
    
    # Paths
    lora_path = MODELS_DIR / model_name / "final_model"
    merged_path = MODELS_DIR / model_name / "merged_model"
    
    # Remove existing merged model if it exists
    if merged_path.exists():
        print(f"    Removing existing merged model at {merged_path}")
        shutil.rmtree(merged_path)
    
    # Load base model
    print(f"    Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load LoRA adapter
    print(f"    Loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    
    # Merge LoRA weights into base model
    print(f"    Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"    Saving merged model to {merged_path}")
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_path))
    
    # Copy tokenizer files to merged model directory
    print(f"    Copying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(merged_path))
    
    # Clean up to free memory
    del merged_model
    del model
    del base_model
    torch.cuda.empty_cache()
    
    print(f"    Merged model saved to {merged_path}", flush=True)
    return merged_path


def compute_perplexity_worker(gpu_id: int, model_name: str, merged_model_path: str, 
                              text_chunk: List[str], chunk_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Worker function to compute perplexity on a single GPU for a chunk of texts.
    
    Args:
        gpu_id: GPU device ID (0-3)
        model_name: Name of the model
        merged_model_path: Path to the merged model
        text_chunk: Chunk of texts to process
        chunk_id: ID of this chunk for logging
    
    Returns:
        Tuple of (individual_perplexities, token_lengths) as numpy arrays
    """
    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        
        import evaluate
        from transformers import AutoTokenizer
        
        print(f"    [GPU {gpu_id}] Processing chunk {chunk_id} with {len(text_chunk):,} reviews", flush=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(merged_model_path))
        
        # Tokenize text chunk to get token lengths
        token_lengths = []
        for review in text_chunk:
            tokens = tokenizer(review, truncation=True, max_length=192, return_tensors=None)
            token_lengths.append(len(tokens['input_ids']))
        token_lengths = np.array(token_lengths)
        
        # Load perplexity metric for this GPU
        perplexity_metric = evaluate.load("perplexity", module_type="metric")
        
        # Compute perplexity on this GPU
        batch_size = 8
        results = perplexity_metric.compute(
            model_id=str(merged_model_path),
            predictions=text_chunk,
            batch_size=batch_size,
            add_start_token=True,
            device="cuda"  # Uses the default CUDA device set by torch.cuda.set_device()
        )
        
        individual_perplexities = np.array(results['perplexities'])
        
        print(f"    [GPU {gpu_id}] Chunk {chunk_id} complete. Mean perplexity: {results['mean_perplexity']:.2f}", flush=True)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        return individual_perplexities, token_lengths
        
    except Exception as e:
        print(f"    [GPU {gpu_id}] Error in worker: {e}")
        import traceback
        traceback.print_exc()
        # Return empty arrays on error
        return np.array([]), np.array([])


def compute_perplexity(model_name: str, test_reviews: List[str]) -> Tuple[float, np.ndarray]:
    """
    Compute mean perplexity using the trained model on test reviews.
    Uses 4 GPUs in parallel for faster computation.
    
    Returns:
        - mean_perplexity: Arithmetic mean of all individual perplexities
        - individual_perplexities: Array of individual perplexity values
    """
    print(f"\n  Computing mean perplexity for {model_name} using 4 GPUs...", flush=True)
    print(f"    Using {len(test_reviews):,} test reviews", flush=True)
    
    try:
        # Check GPU availability
        num_gpus = torch.cuda.device_count()
        if num_gpus < 4:
            raise RuntimeError(f"Error: Only {num_gpus} GPUs available, but 4 are required")
        
        # Merge LoRA adapter with base model (only needs to be done once)
        merged_model_path = merge_lora_if_needed(model_name)
        
        # Split test reviews into 4 chunks for parallel processing
        chunk_size = len(test_reviews) // 4
        chunks = []
        for i in range(4):
            start_idx = i * chunk_size
            if i == 3:  # Last chunk gets any remaining samples
                chunks.append(test_reviews[start_idx:])
            else:
                chunks.append(test_reviews[start_idx:start_idx + chunk_size])
        
        print(f"    Split into chunks of sizes: {[len(c) for c in chunks]}")
        
        # Use multiprocessing to compute perplexity on each GPU in parallel
        print(f"    Launching 4 parallel workers on GPUs 0-3...")
        with mp.Pool(processes=4) as pool:
            # Create worker tasks
            tasks = []
            for gpu_id, (chunk_id, text_chunk) in enumerate(zip(range(4), chunks)):
                tasks.append((gpu_id, model_name, merged_model_path, text_chunk, chunk_id))
            
            # Run workers in parallel
            results = pool.starmap(compute_perplexity_worker, tasks)
        
        # Combine results from all GPUs
        all_perplexities = []
        all_token_lengths = []
        
        for i, (perplexities, token_lengths) in enumerate(results):
            if len(perplexities) > 0:
                all_perplexities.append(perplexities)
                all_token_lengths.append(token_lengths)
            else:
                print(f"    Warning: GPU {i} returned empty results")
        
        if not all_perplexities:
            print(f"    Error: No valid results from any GPU")
            return float('inf'), np.array([])
        
        # Concatenate all results
        individual_perplexities = np.concatenate(all_perplexities)
        token_lengths = np.concatenate(all_token_lengths)
        
        print(f"    Combined {len(individual_perplexities):,} perplexity values")
        print(f"    Average token length: {token_lengths.mean():.1f} (min: {token_lengths.min()}, max: {token_lengths.max()})")
        
        # Compute simple arithmetic mean perplexity
        mean_perplexity = np.mean(individual_perplexities)
        
        print(f"    Mean perplexity: {mean_perplexity:.2f}", flush=True)
        print(f"    Min perplexity: {individual_perplexities.min():.2f}", flush=True)
        print(f"    Max perplexity: {individual_perplexities.max():.2f}", flush=True)
        print(f"    Std perplexity: {individual_perplexities.std():.2f}", flush=True)
        
        return mean_perplexity, individual_perplexities
        
    except Exception as e:
        print(f"    Error computing perplexity: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), np.array([])



def compute_mauve(test_reviews: List[str], generated_samples: List[str]) -> Dict:
    """
    Compute MAUVE score between test and generated distributions.
    Returns dict with mauve score, frontier integral, and divergence curve.
    """
    print(f"\n  Computing MAUVE score...", flush=True)
    
    try:
        import mauve
        
        # Subset generated samples if in test mode
        if GENERATED_SAMPLES_TO_USE is not None:
            generated_samples = generated_samples[:GENERATED_SAMPLES_TO_USE]
        
        # Ensure equal number of samples
        min_samples = min(len(test_reviews), len(generated_samples))
        test_reviews = test_reviews[:min_samples]
        generated_samples = generated_samples[:min_samples]
        
        print(f"    Using {min_samples:,} samples for MAUVE (test: {len(test_reviews)}, gen: {len(generated_samples)})")
        
        # Compute MAUVE score
        out = mauve.compute_mauve(
            p_text=test_reviews,
            q_text=generated_samples,
            device_id=0 if torch.cuda.is_available() else -1,
            max_text_length=192,
            verbose=False,
            batch_size=32
        )
        
        # Extract all MAUVE outputs
        mauve_result = {
            'mauve': float(out.mauve),
            'frontier_integral': float(out.frontier_integral),
            'divergence_curve': out.divergence_curve.tolist() if hasattr(out.divergence_curve, 'tolist') else out.divergence_curve
        }
        
        print(f"    MAUVE score: {mauve_result['mauve']:.4f}")
        print(f"    Frontier integral: {mauve_result['frontier_integral']:.4f} (smaller is better)")
        
        return mauve_result
        
    except Exception as e:
        print(f"    Error computing MAUVE: {e}")
        return {'mauve': 0.0, 'frontier_integral': 0.0, 'divergence_curve': []}


def compute_wasserstein_distance(gen_pmf: np.ndarray, test_pmf: np.ndarray) -> float:
    """
    Compute Wasserstein distance between rating distributions.
    """
    # Support points (ratings 1-5)
    support = np.array([1, 2, 3, 4, 5])
    
    # Compute Wasserstein distance
    distance = scipy.stats.wasserstein_distance(
        support, support,
        u_weights=gen_pmf,
        v_weights=test_pmf
    )
    
    return distance


def evaluate_model(model_name: str, test_reviews_perplexity: List[str], 
                   test_reviews_mauve: List[str], test_df_perplexity: pd.DataFrame,
                   test_rating_dist: Dict, output_dir: Path) -> Dict:
    """
    Evaluate a single model on all metrics.
    """
    print(f"\n{'='*80}", flush=True)
    print(f"EVALUATING MODEL: {model_name}", flush=True)
    print(f"{'='*80}", flush=True)
    
    results = {
        'model_name': model_name,
        'perplexity': None,
        'individual_perplexities': None,
        'mauve': None,
        'mauve_frontier_integral': None,
        'mauve_divergence_curve': None,
        'wasserstein_distance': None,
        'rating_distribution': {},
        'num_generated_samples': 0,
        'num_generated_samples_used': 0
    }
    
    # Load generated samples
    generated_samples, gen_rating_dist = load_generated_samples(model_name)
    results['rating_distribution'] = gen_rating_dist
    results['num_generated_samples'] = len(generated_samples)
    
    if not generated_samples:
        print(f"  Skipping {model_name} - no generated samples found")
        return results
    
    # Subset generated samples if in test mode
    generated_samples_for_eval = generated_samples
    if GENERATED_SAMPLES_TO_USE is not None:
        generated_samples_for_eval = generated_samples[:GENERATED_SAMPLES_TO_USE]
        print(f"  Using {len(generated_samples_for_eval):,} of {len(generated_samples):,} generated samples (test mode)")
    
    results['num_generated_samples_used'] = len(generated_samples_for_eval)
    
    # Compute perplexity
    mean_perplexity, individual_perplexities = compute_perplexity(model_name, test_reviews_perplexity)
    results['perplexity'] = mean_perplexity
    results['individual_perplexities'] = individual_perplexities.tolist() if len(individual_perplexities) > 0 else []
    
    # Save detailed perplexity data
    if len(individual_perplexities) > 0:
        perplexity_details_dir = output_dir / "perplexity_details"
        perplexity_details_dir.mkdir(parents=True, exist_ok=True)
        
        # Add perplexity column to test dataframe
        test_df_with_perplexity = test_df_perplexity.copy()
        test_df_with_perplexity['perplexity'] = individual_perplexities
        
        # Save as parquet
        detail_path = perplexity_details_dir / f"{model_name}_perplexity_details.parquet"
        test_df_with_perplexity.to_parquet(detail_path, index=False)
        print(f"    Saved perplexity details to {detail_path}")
    
    # Compute MAUVE (function handles subsetting internally)
    mauve_result = compute_mauve(test_reviews_mauve, generated_samples)
    results['mauve'] = mauve_result.get('mauve', 0.0)
    results['mauve_frontier_integral'] = mauve_result.get('frontier_integral', 0.0)
    results['mauve_divergence_curve'] = mauve_result.get('divergence_curve', [])
    
    # Compute Wasserstein distance
    # For Wasserstein, always use ALL generated samples for accurate distribution
    gen_pmf = np.array([gen_rating_dist.get(str(i), 0.0) for i in range(1, 6)])
    test_pmf = np.array([test_rating_dist.get(str(i), 0.0) for i in range(1, 6)])
    wasserstein = compute_wasserstein_distance(gen_pmf, test_pmf)
    results['wasserstein_distance'] = wasserstein
    
    print(f"\n  Summary for {model_name}:", flush=True)
    print(f"    Mean Perplexity: {mean_perplexity:.2f}", flush=True)
    print(f"    MAUVE: {results['mauve']:.4f}", flush=True)
    print(f"    Frontier Integral: {results['mauve_frontier_integral']:.4f} (smaller is better)", flush=True)
    print(f"    Wasserstein Distance: {wasserstein:.4f}", flush=True)
    
    return results


def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("STEP 09: COMPUTE METRICS FOR GENERATED SAMPLES")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test set samples
    test_reviews_perplexity, test_reviews_mauve, test_df_perplexity, test_ratings, test_rating_dist = load_test_set_samples(
        TEST_SAMPLES_FOR_PERPLEXITY, TEST_SAMPLES_FOR_MAUVE
    )
    
    # Initialize results
    all_results = {
        'configuration': {
            'test_mode': TEST_MODE,
            'test_samples_perplexity': TEST_SAMPLES_FOR_PERPLEXITY,
            'test_samples_mauve': TEST_SAMPLES_FOR_MAUVE,
            'generated_samples_to_use': GENERATED_SAMPLES_TO_USE if TEST_MODE else 'all',
        },
        'test_set_stats': {
            'total_samples': len(test_ratings),
            'samples_used_for_perplexity': len(test_reviews_perplexity),
            'samples_used_for_mauve': len(test_reviews_mauve),
            'rating_distribution': test_rating_dist
        },
        'model_results': {},
        'timestamp': datetime.now().isoformat(),
    }
    
    # Evaluate each model
    for model_name in MODEL_NAMES:
        model_results = evaluate_model(
            model_name, 
            test_reviews_perplexity,
            test_reviews_mauve,
            test_df_perplexity,
            test_rating_dist,
            OUTPUT_DIR
        )
        all_results['model_results'][model_name] = model_results
        
        # Save intermediate results
        intermediate_path = OUTPUT_DIR / f"metrics_results_intermediate_{model_name}.json"
        with open(intermediate_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Calculate runtime
    runtime = time.time() - start_time
    all_results['runtime'] = str(timedelta(seconds=int(runtime)))
    
    # Save final results
    output_path = OUTPUT_DIR / "metrics_results_final.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("METRICS COMPUTATION COMPLETE")
    print("="*80)
    print(f"Total runtime: {timedelta(seconds=int(runtime))}")
    print(f"Results saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<30} {'Mean Perp.':>12} {'MAUVE':>10} {'Frontier':>10} {'Wasserstein':>12}")
    print("-" * 75)
    
    for model_name, results in all_results['model_results'].items():
        perp = results.get('perplexity', float('inf'))
        mauve = results.get('mauve', 0.0)
        frontier = results.get('mauve_frontier_integral', 0.0)
        wass = results.get('wasserstein_distance', float('inf'))
        
        if perp != float('inf'):
            print(f"{model_name:<30} {perp:>12.2f} {mauve:>10.4f} {frontier:>10.4f} {wass:>12.4f}")
        else:
            print(f"{model_name:<30} {'N/A':>12} {mauve:>10.4f} {frontier:>10.4f} {wass:>12.4f}")
    
    print("="*80)


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
