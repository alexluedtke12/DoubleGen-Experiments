#!/usr/bin/env python3

import os
import numpy as np
from datasets import load_dataset, load_from_disk
from PIL import Image
import pandas as pd
from torchvision import transforms

def resample_and_save_images():
    """
    Load the saved test set from HuggingFace, resample 10k rows with replacement
    based on AIPW weights, resize to 64x64, and save images as PNGs.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load the dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_from_disk("./CelebA-attrs-test")
    
    # Drop unnecessary columns to save memory
    print("Dropping mask and image_and_mask columns...")
    dataset = dataset.remove_columns(['mask', 'image_and_mask'])
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Extract weights directly from the dataset
    raw_weights = np.array(dataset['aipw_rightProp_rightOut'])

    # ---------- added diagnostics (fix #3) ----------
    n_neg  = np.sum(raw_weights < 0)
    n_nan  = np.sum(np.isnan(raw_weights))
    n_inf  = np.sum(~np.isfinite(raw_weights)) - n_nan  # ±inf only

    print(f"Weight quality check — negatives: {n_neg}, NaNs: {n_nan}, ±inf: {n_inf}")
    # -----------------------------------------------

    # Handle any NaN or negative weights
    weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.maximum(weights, 0)  # Ensure non-negative
    
    if np.sum(weights) == 0:
        print("Warning: All weights are zero, using uniform sampling")
        probabilities = np.ones(len(weights)) / len(weights)
    else:
        probabilities = weights / np.sum(weights)
    
    print(f"Weight statistics - Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
    
    # Sample 10k indices with replacement based on probabilities
    print("Sampling 10k rows with replacement...")
    sample_size = 10000
    sampled_indices = np.random.choice(
        len(dataset), 
        size=sample_size, 
        replace=True, 
        p=probabilities
    )
    
    # Set up the same image transformations as in the training script
    class SquareCenterCrop:
        def __call__(self, img):
            w, h = img.size
            size = min(w, h)
            return transforms.CenterCrop(size)(img)

    # Image preprocessing pipeline matching the training script
    image_preprocess = transforms.Compose([
        SquareCenterCrop(),  # Center crop to square
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 64x64
    ])
    
    # Create output directory
    output_dir = "test_cf_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the sampled images
    print(f"Saving resized 64x64 images to {output_dir}/...")
    
    for i, idx in enumerate(sampled_indices):
        # Convert numpy int64 to Python int for dataset indexing
        idx = int(idx)
        
        # Get the image (RGB format) directly from the dataset
        image = dataset[idx]['image']
        
        # Ensure it's a PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply the same preprocessing as in training (square crop + resize to 64x64)
        image = image_preprocess(image)
        
        # Save as PNG with zero-padded filename
        filename = f"image_{i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath, "PNG")
        
        # Print progress every 1000 images
        if (i + 1) % 1000 == 0:
            print(f"Saved {i + 1}/{sample_size} images")
    
    print(f"Successfully saved {sample_size} 64x64 images to {output_dir}/")
    
    # Save sampling metadata
    metadata = {
        'original_dataset_size': len(dataset),
        'sample_size': sample_size,
        'sampled_indices': sampled_indices.tolist(),
        'weight_column': 'aipw_rightProp_rightOut',
        'image_size': '64x64',
        'preprocessing': 'SquareCenterCrop + BICUBIC resize to 64x64',
        'weight_stats': {
            'min': float(weights.min()),
            'max': float(weights.max()),
            'mean': float(weights.mean()),
            'std': float(weights.std())
        }
    }
    
    import json
    with open(os.path.join(output_dir, 'sampling_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sampling metadata saved to {output_dir}/sampling_metadata.json")
    
    # Print frequency diagnostics
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    freq_df = pd.DataFrame({
        'original_index': unique_indices,
        'sample_count': counts,
        'weight': weights[unique_indices]
    })
    freq_df = freq_df.sort_values('sample_count', ascending=False)
    
    print("\n" + "="*60)
    print("SAMPLING FREQUENCY DIAGNOSTICS")
    print("="*60)
    print(f"Total unique images sampled: {len(unique_indices)} out of {len(dataset)} available")
    print(f"Most frequently sampled image appeared: {freq_df['sample_count'].max()} times")
    print(f"Least frequently sampled image appeared: {freq_df['sample_count'].min()} times")
    print(f"Average samples per unique image: {freq_df['sample_count'].mean():.2f}")
    
    print(f"\nTop 10 most frequently sampled images:")
    print("-" * 50)
    for i, row in freq_df.head(10).iterrows():
        print(f"Index {int(row['original_index']):6d}: sampled {int(row['sample_count']):3d} times (weight: {row['weight']:8.4f})")
    
    # Show distribution of sampling frequencies
    freq_counts = pd.Series(counts).value_counts().sort_index()
    print(f"\nDistribution of sampling frequencies:")
    print("-" * 40)
    for freq, count in freq_counts.items():
        print(f"{count:4d} images were sampled {freq:2d} time(s)")
    
    print("="*60)

if __name__ == "__main__":
    resample_and_save_images()