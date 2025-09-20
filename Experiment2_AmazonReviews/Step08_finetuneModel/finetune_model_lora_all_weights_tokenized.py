#!/usr/bin/env python3
"""
Finetune language model with LoRA using pre-tokenized data.

This script:
1. Loads pre-tokenized data from Step07
2. Trains 9 models with different weight columns (including naive baseline)
3. Saves 2-3 hours by avoiding redundant tokenization
"""

import os
import sys

# Force unbuffered output for real-time logging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # Line buffered
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)  # Line buffered

# Disable triton to avoid compilation issues - MUST be before torch import
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json
import gc
import random
import numpy as np

# Set random seeds for reproducibility
# Each process/GPU will use a different seed based on its rank
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Enable deterministic operations where possible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_from_disk, Dataset
import pandas as pd

# ============ CONFIGURATION ============

# HuggingFace token - load from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set!")
    print("Please set it using: export HF_TOKEN='your_token_here'")
    print("Or load it from a secure file before running this script.")
    sys.exit(1)

# All weight columns to iterate over
# Order: 1) naive 2) aipw both right 3) ipw right 4) gcomp right 
#        5-6) aipw one wrong 7) ipw wrong 8) gcomp wrong 9) aipw both wrong
WEIGHT_CONFIGS = {
    # 1. Naive weight (baseline)
    "naive": {"type": "naive", "description": "Uniform weight of 1 for all"},
    
    # 2. AIPW both right
    "aipw_ipw_right_out_right": {"type": "aipw", "description": "AIPW: right IPW + k-NN"},
    
    # 3. IPW right
    "ipw_right": {"type": "ipw", "description": "IPW with all features"},
    
    # 4. G-computation right
    "gcomp_right": {"type": "gcomp", "description": "G-comp with k-NN"},
    
    # 5-6. AIPW one wrong
    "aipw_ipw_right_out_wrong": {"type": "aipw", "description": "AIPW: right IPW + uniform"},
    "aipw_ipw_wrong_out_right": {"type": "aipw", "description": "AIPW: wrong IPW + k-NN"},
    
    # 7. IPW wrong
    "ipw_wrong": {"type": "ipw", "description": "IPW without categories"},
    
    # 8. G-computation wrong
    "gcomp_wrong": {"type": "gcomp", "description": "G-comp with uniform"},
    
    # 9. AIPW both wrong
    "aipw_ipw_wrong_out_wrong": {"type": "aipw", "description": "AIPW: wrong IPW + uniform"},
}

# Training configuration
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03

# Paths
BASE_DIR = Path("/n/scratch/users/a/your_username")
TOKENIZED_DIR = BASE_DIR / "tokenized_reviews"
OUTPUT_BASE_DIR = BASE_DIR / "models_finetuned_all_weights"
CACHE_DIR = BASE_DIR / "huggingface_cache"

# Model configuration
MODEL_ID = "meta-llama/Llama-3.2-1B"
MAX_LENGTH = 192  # Must match tokenization step

# Test mode settings
TEST_MODE = False  # Set to False for full training
TEST_SAMPLES = 1000 if TEST_MODE else None  # 1K samples for testing to test generation speed
TEST_EPOCHS = 1 if TEST_MODE else 1  # 1 epoch for both test and production


class WeightedDataCollator:
    """Custom data collator that preserves the weight field."""
    
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
    
    def __call__(self, features):
        # Extract weights before processing
        weights = torch.tensor([f.pop('weight', 1.0) for f in features], dtype=torch.float32)
        
        # Use standard collator for the rest
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        # Add weights back
        batch['weight'] = weights
        
        # For causal LM, labels are the same as input_ids
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
        
        return batch


class WeightedTrainer(Trainer):
    """Custom trainer that uses sample weights in loss computation."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract weights
        weights = inputs.pop("weight", None)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Compute per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['labels'][..., 1:].contiguous()
        
        # Reshape for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss = loss_fct(shift_logits, shift_labels)
        
        # Reshape back to [batch_size, seq_length]
        loss = loss.view(inputs['labels'].size(0), -1)
        
        # Average over sequence length
        loss = loss.mean(dim=1)
        
        # Apply sample weights if provided
        if weights is not None:
            weighted_loss = (loss * weights).sum() / weights.sum()
        else:
            weighted_loss = loss.mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss


def load_tokenized_data(split: str, weight_column: str, sample_size: Optional[int] = None) -> Dataset:
    """Load pre-tokenized data and prepare with specific weight column.
    
    Note: Data is already filtered to synthetic verified purchases in Step07,
    so no additional filtering by verified_purchase is needed.
    Split parameter is now ignored - always loads both train1 and train2.
    """
    
    # Load BOTH tokenized datasets
    train1_path = TOKENIZED_DIR / "train1_tokenized"
    train2_path = TOKENIZED_DIR / "train2_tokenized"
    
    print(f"\nLoading pre-tokenized training sets from both train1 and train2")
    print(f"(Already filtered to synthetic verified purchases)")
    
    if not train1_path.exists() or not train2_path.exists():
        raise FileNotFoundError(f"Tokenized data not found. Please run Step07 first.")
    
    # Load both datasets
    from datasets import concatenate_datasets
    dataset1 = load_from_disk(str(train1_path))
    dataset2 = load_from_disk(str(train2_path))
    print(f"Loaded {len(dataset1):,} samples from train1")
    print(f"Loaded {len(dataset2):,} samples from train2")
    
    # Combine the datasets
    dataset = concatenate_datasets([dataset1, dataset2])
    print(f"Combined into {len(dataset):,} total tokenized samples")
    
    # Handle special case of 'naive' weight
    if weight_column == 'naive':
        # For naive, use weight 1 for all samples
        print(f"Using naive weighting: weight=1 for all samples")
        print(f"Working with {len(dataset):,} samples (all synthetic verified purchases)")
    else:
        # Check if weight column exists for non-naive cases
        if weight_column not in dataset.column_names:
            available_weights = [col for col in dataset.column_names if 'ipw' in col or 'gcomp' in col or 'aipw' in col]
            raise ValueError(f"Weight column '{weight_column}' not found. Available: {available_weights}")
        
        # No filtering needed - data already filtered to synthetic verified purchases in Step07
        print(f"Using weight column: {weight_column}")
        print(f"Working with {len(dataset):,} samples (all synthetic verified purchases)")
    
    # Shuffle the combined dataset for good mixing between train1 and train2
    print("Shuffling combined dataset...")
    dataset = dataset.shuffle(seed=42)
    
    # Sample if requested (for testing)
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.select(range(sample_size))
        print(f"Sampled to {sample_size:,} samples for testing")
    
    # Set weight column - either 1 for naive or from the dataset
    if weight_column == 'naive':
        dataset = dataset.map(lambda x: {'weight': 1.0}, num_proc=4)
    else:
        dataset = dataset.map(lambda x: {'weight': x[weight_column]}, num_proc=4)
    
    # Remove other weight columns to save memory
    columns_to_remove = [col for col in dataset.column_names 
                        if col in WEIGHT_CONFIGS and col != 'weight']
    dataset = dataset.remove_columns(columns_to_remove)
    
    # Show weight statistics
    weights = dataset['weight']
    print(f"\nWeight statistics for {weight_column}:")
    print(f"  Mean: {sum(weights)/len(weights):.4f}")
    print(f"  Min: {min(weights):.4f}")
    print(f"  Max: {max(weights):.4f}")
    
    print(f"Final dataset has {len(dataset):,} samples with features: {dataset.column_names}")
    
    return dataset


def generate_samples(model, tokenizer, num_samples: int = 10000, max_new_tokens: int = 100, prompt: str = ""):
    """Generate sample reviews from the finetuned model.
    
    Args:
        model: The finetuned model
        tokenizer: The tokenizer
        num_samples: Number of samples to generate
        max_new_tokens: Maximum new tokens to generate (will be truncated to MAX_LENGTH total)
        prompt: The prompt to use (empty for unconditional, or "N stars: " for conditional)
    """
    model.eval()
    samples = []
    
    generation_type = "unconditional" if prompt == "" else f"conditional ({prompt.strip()})"
    print(f"\nGenerating {num_samples} {generation_type} sample reviews...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Set seed for this specific example (sequential generation starts at 0)
            seed = i
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Use the provided prompt
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            
            # Generate with max_length constraint (not max_new_tokens)
            # This ensures total length including prompt doesn't exceed MAX_LENGTH
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,  # Use max_length instead of max_new_tokens
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append(generated_text)
            
            if (i + 1) % 1000 == 0:  # Print progress every 1000 samples
                print(f"  Generated {i+1}/{num_samples} samples...")
    
    return samples


def train_single_model(weight_column: str, weight_info: Dict, split: str = "1"):
    """Train a single model with specified weight column using pre-tokenized data."""
    
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training with {weight_column}")
    print(f"Type: {weight_info['type']}, Description: {weight_info['description']}")
    print("="*80)
    
    # Reset random seeds for each model to ensure reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Create output directory for this weight variant
    output_dir = OUTPUT_BASE_DIR / weight_column
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize variables that need cleanup
    model = None
    trainer = None
    
    try:
        # Load tokenizer (needed for generation and padding)
        print(f"\nLoading tokenizer from {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print(f"Loading model {MODEL_ID} with bfloat16...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            token=HF_TOKEN
        )
    
        # Configure LoRA
        if "llama" in MODEL_ID.lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
    
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
        # Enable gradient checkpointing
        model.enable_input_require_grads()
    
        # Ensure model is in training mode
        model.train()
    
        # Load pre-tokenized data with specific weight column
        train_dataset = load_tokenized_data(split, weight_column, sample_size=TEST_SAMPLES)
    
        # Training arguments
        # Calculate logging steps based on dataset size to reduce log spam
        # Log approximately 1000 times per epoch
        total_steps = len(train_dataset) // (24 * torch.cuda.device_count())  # batch_size * num_gpus
        log_interval = max(1, total_steps // 1000)  # Log ~1000 times per epoch, minimum 1 step
        
        training_args = TrainingArguments(
            output_dir=output_dir / "checkpoints",
            per_device_train_batch_size=24,  # Increased from 16 to 24
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=TEST_EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=log_interval,  # Dynamic logging interval
            logging_first_step=True,  # Log first step for debugging
            save_strategy="epoch",
            eval_strategy="no",
            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
            report_to="none",
            disable_tqdm=False,  # Keep progress bar but it won't spam logs
            dataloader_num_workers=12,  # Increased from 0 to 12 for better data loading performance
            dataloader_persistent_workers=True,  # Keep workers alive between epochs
            dataloader_prefetch_factor=6,  # Prefetch 6 batches per worker
            remove_unused_columns=False,
            # Note: Trainer shuffles by default, no need for explicit parameter
        )
    
        # Create trainer
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=WeightedDataCollator(tokenizer),
        )
    
        # Train
        print(f"\nStarting training for {weight_column}...")
        start_time = datetime.now()
    
        trainer.train()
    
        duration = datetime.now() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Training completed in {duration}")
    
        # Save model
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving model to {output_dir}/final_model...")
        final_model_dir = output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    
        # Generate samples - both unconditional and conditional
        # Generate different number of samples based on test mode
        num_samples_to_generate = 40 if TEST_MODE else 10000  # 40 for test, 10K for production
        
        # Generate 6 different sample files
        prompts = {
            "unconditional": "",
            "1_star": "1 star: ",  # Note: singular for 1 star
            "2_stars": "2 stars: ",
            "3_stars": "3 stars: ",
            "4_stars": "4 stars: ",
            "5_stars": "5 stars: "
        }
        
        all_generated_samples = {}
        
        for prompt_name, prompt_text in prompts.items():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating {prompt_name} samples...")
            
            # Always use parallel generation for better consistency and speed
            print(f"Using parallel generation with 4 GPUs for {num_samples_to_generate} samples...")
            import subprocess
            
            temp_samples_file = output_dir / f"temp_generated_samples_{prompt_name}.json"
            
            # Call parallel generation script v2 with prompt support
            cmd = [
                sys.executable,
                "/home/your_username/cf_reviews/Step08_finetuneModel/generate_samples_parallel_v2.py",
                weight_column,
                str(temp_samples_file),
                str(num_samples_to_generate)
            ]
            # Add prompt as argument if not empty
            if prompt_text:
                cmd.append(prompt_text)
            
            result = subprocess.run(cmd, text=True)
            
            if result.returncode != 0:
                print(f"Warning: Parallel generation failed with return code {result.returncode}")
                print("Falling back to sequential generation...")
                samples = generate_samples(model, tokenizer, num_samples=num_samples_to_generate, prompt=prompt_text)
            else:
                # Load the generated samples
                with open(temp_samples_file, 'r') as f:
                    samples = json.load(f)
                temp_samples_file.unlink()  # Clean up temp file
            
            # Save samples for this prompt type
            samples_file = output_dir / f"generated_samples_{prompt_name}.json"
            with open(samples_file, 'w') as f:
                json.dump({
                    "weight_column": weight_column,
                    "weight_type": weight_info['type'],
                    "description": weight_info['description'],
                    "prompt_type": prompt_name,
                    "prompt": prompt_text,
                    "training_duration": str(duration),
                    "num_samples": len(samples),
                    "samples": samples
                }, f, indent=2)
            
            all_generated_samples[prompt_name] = samples
            print(f"✓ Saved {len(samples)} {prompt_name} samples to {samples_file}")
    
        # Save training summary
        summary_file = output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "weight_column": weight_column,
                "weight_type": weight_info['type'],
                "description": weight_info['description'],
                "model_id": MODEL_ID,
                "max_length": MAX_LENGTH,
                "learning_rate": LEARNING_RATE,
                "num_epochs": TEST_EPOCHS,
                "training_duration": str(duration),
                "dataset_size": len(train_dataset),
                "test_mode": TEST_MODE,
                "using_pretokenized": True
            }, f, indent=2)
    
        print(f"✓ Saved training summary to {summary_file}")
        
        return duration
        
    finally:
        # Always clear GPU memory, even if training fails
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Cleaning up GPU memory...")
        if model is not None:
            del model
        if trainer is not None:
            del trainer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU memory cleared")


def main(model_name=None):
    """Main training function. If model_name provided, train single model, otherwise train all.
    
    Args:
        model_name: Optional name of specific model to train (e.g., 'naive', 'ipw_right')
    """
    
    print("\n" + "="*80)
    print("TRAINING MODELS WITH PRE-TOKENIZED DATA")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test mode: {TEST_MODE}")
    print(f"Samples per model: {TEST_SAMPLES if TEST_MODE else 'Full dataset'}")
    print(f"Epochs: {TEST_EPOCHS}")
    
    # Determine which models to train
    if model_name:
        if model_name not in WEIGHT_CONFIGS:
            print(f"ERROR: Unknown model '{model_name}'")
            print(f"Available models: {', '.join(WEIGHT_CONFIGS.keys())}")
            return 1
        models_to_train = {model_name: WEIGHT_CONFIGS[model_name]}
        print(f"Training single model: {model_name}")
    else:
        models_to_train = WEIGHT_CONFIGS
        print(f"Number of weight variants: {len(WEIGHT_CONFIGS)}")
    print(f"Tokenized data directory: {TOKENIZED_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    # Create base output directory
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track results
    all_results = {}
    total_start = datetime.now()
    
    # Train model for each weight variant
    for idx, (weight_column, weight_info) in enumerate(models_to_train.items(), 1):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting model {idx}/{len(models_to_train)}: {weight_column}")
        
        try:
            duration = train_single_model(weight_column, weight_info, split="1")
            all_results[weight_column] = {
                "status": "success",
                "duration": str(duration),
                "type": weight_info['type'],
                "description": weight_info['description']
            }
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Model {idx}/{len(models_to_train)} completed successfully")
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ ERROR training model {idx}/{len(models_to_train)} ({weight_column}): {str(e)}")
            all_results[weight_column] = {
                "status": "failed",
                "error": str(e),
                "type": weight_info['type'],
                "description": weight_info['description']
            }
            continue
    
    # Total duration
    total_duration = datetime.now() - total_start
    
    # Save overall summary
    summary_file = OUTPUT_BASE_DIR / "all_models_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "test_mode": TEST_MODE,
            "total_duration": str(total_duration),
            "num_models_trained": sum(1 for r in all_results.values() if r['status'] == 'success'),
            "num_models_failed": sum(1 for r in all_results.values() if r['status'] == 'failed'),
            "using_pretokenized": True,
            "model_results": all_results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE")
    print("="*80)
    print(f"Total duration: {total_duration}")
    print(f"Successfully trained: {sum(1 for r in all_results.values() if r['status'] == 'success')}/{len(WEIGHT_CONFIGS)}")
    print(f"Results saved to: {summary_file}")
    
    # Print summary table
    print("\nSummary by weight type:")
    for weight_type in ['naive', 'ipw', 'gcomp', 'aipw']:
        type_results = {k: v for k, v in all_results.items() if v['type'] == weight_type}
        success_count = sum(1 for r in type_results.values() if r['status'] == 'success')
        print(f"  {weight_type}: {success_count}/{len(type_results)} successful")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train language models with different weight variants')
    parser.add_argument('--model', type=str, help='Train specific model instead of all')
    args = parser.parse_args()
    
    # Run main with optional model name
    result = main(model_name=args.model)
    sys.exit(result if result else 0)