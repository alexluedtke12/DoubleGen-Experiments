# Experiment 1: Smiling Faces (CelebA)

This directory contains the core implementation for the counterfactual generation experiment using CelebA dataset with smiling attribute.

## Directory Structure

- `src/` - Source code for training and evaluation
  - `train/` - Training scripts including nuisance fitting
  - `eval/` - Evaluation scripts for preprocessing, CF generation, and FID computation
- `CelebA-attrs-model/` - Model checkpoints directory (filled in during training)
- `run_nuisance_experiment.sh` - Main training pipeline
- `eval_performance.sh` - Evaluation pipeline
- `environment.yml` - Conda environment specification

## Running the Experiment

1. Set up the environment:
   ```bash
   conda env create -f environment.yml
   conda activate torch-cuda
   ```

2. Install additional dependencies and download model:
   ```bash
   pip install mediapipe
   pip install git+https://github.com/facebookresearch/segment-anything.git
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

3. Run the training pipeline:
   ```bash
   ./run_nuisance_experiment.sh
   ```

4. Evaluate performance:
   ```bash
   ./eval_performance.sh
   ```

## Notes

1. Model checkpoint files (*.pth) are not included due to size. The directory structure is preserved for reference.

2. Dataset references have been updated to use local paths (`./CelebA-attrs`). You'll need to either:
   - Place the CelebA dataset in the local directory
   - Update the paths to point to your HuggingFace datasets
