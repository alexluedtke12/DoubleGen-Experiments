#!/usr/bin/env python3
"""
Parallel sample generation using 4 GPUs.
Each GPU generates num_samples/4 samples independently with fixed seeds.
"""

import os
import sys
import torch
from torch.multiprocessing import Process, Queue
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import random
import numpy as np

# Get HF token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set!")
    print("Please set it using: export HF_TOKEN='your_token_here'")
    print("Or load it from a secure file before running this script.")
    sys.exit(1)

def generate_on_gpu(gpu_id, model_path, num_samples, queue, start_index, max_new_tokens=100):
    """Generate samples on a specific GPU with index-based seeds."""
    
    # Set the device
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    print(f"GPU {gpu_id}: Loading model (will use seeds {start_index} to {start_index + num_samples - 1})...", flush=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        token=HF_TOKEN,
        cache_dir="~/data_directory/huggingface_cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        cache_dir="~/data_directory/huggingface_cache",
        token=HF_TOKEN
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)
    model.eval()
    
    print(f"GPU {gpu_id}: Generating {num_samples} samples...", flush=True)
    samples = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Set seed for this specific example
            seed = start_index + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Empty prompt for unconditional generation
            inputs = tokenizer("", return_tensors="pt", padding=True).to(device)
            
            # Generate with pure random sampling (faster)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,  # Pure random sampling
                top_p=1.0,        # No nucleus sampling cutoff
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append(generated_text)
            
            if (i + 1) % 250 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: Generated {i+1}/{num_samples} samples...", flush=True)
    
    print(f"GPU {gpu_id}: Done! Generated {len(samples)} samples", flush=True)
    queue.put(samples)

def generate_samples_parallel(model_path, total_samples=10000, num_gpus=4):
    """Generate samples in parallel across multiple GPUs."""
    
    start_time = time.time()
    samples_per_gpu = total_samples // num_gpus
    
    # Create queue for collecting results
    queue = Queue()
    processes = []
    
    # Start a process for each GPU
    for gpu_id in range(num_gpus):
        # Calculate start index for this GPU
        # GPU 0: 0, GPU 1: 2500, GPU 2: 5000, GPU 3: 7500 (for 10k total)
        start_index = gpu_id * samples_per_gpu
        
        # Last GPU might get extra samples if total_samples not divisible by num_gpus
        num_samples = samples_per_gpu + (total_samples % num_gpus if gpu_id == num_gpus - 1 else 0)
        
        p = Process(target=generate_on_gpu, args=(gpu_id, model_path, num_samples, queue, start_index))
        p.start()
        processes.append(p)
    
    # Collect results
    all_samples = []
    for _ in range(num_gpus):
        samples = queue.get()
        all_samples.extend(samples)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    duration = time.time() - start_time
    print(f"\nGenerated {len(all_samples)} total samples in {duration:.1f} seconds")
    print(f"Speed: {len(all_samples) / duration:.1f} samples/second")
    
    return all_samples

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python generate_samples_parallel.py <model_name> <output_file>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    output_file = sys.argv[2]
    
    model_path = f"~/data_directory/models_finetuned_all_weights/{model_name}/final_model"
    
    print(f"Generating 10000 samples for {model_name} using 4 GPUs...")
    samples = generate_samples_parallel(model_path, total_samples=10000, num_gpus=4)
    
    # Save samples
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_file}")