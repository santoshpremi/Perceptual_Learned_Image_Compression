# Perceptual Learned Image Compression 

## Project Scope

- **Target paper**: `HFLIC: Human Friendly Perceptual Learned Image Compression with Reinforced Transform` (Ning et al., 2023)
- **Objective**: Reproduce and extend the HFLIC pipeline to optimise for perceptual quality on benchmark image datasets.
- **Codebase**: Official HFLIC implementation mirrored at the repository root (config, datasets, models, losses, modules, utils, weights).

## Repository Layout

- `config/`, `datasets/`, `loss/`, `models/`, `modules/`, `utils/`, `weights/`, `train_gan.py`, `test_5.py`: Core HFLIC implementation as provided in the reference repository ([beiluo97/HFLIC](https://github.com/beiluo97/HFLIC/tree/main)).
- `configs/`: Dataset paths, filelists, and practicum-specific configuration templates.
- `scripts/`: Helper scripts for environment setup, dataset preparation, training, and evaluation.
- `literature/`: Source papers and bibliography references required for the practicum.
- `data/`: Placeholder directory for datasets once downloaded (ignored by git).
- `outputs/`: Directory for experiment artefacts (ignored by git).

## Environment Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# choose the wheel that matches your system (cu121, cu118, cpu)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> Alternatively run `bash scripts/setup_environment.sh --cuda 12.1` to automate the steps above.

## Dataset Preparation

- **Open Images V7 (training/finetuning)**: 
  - Download image IDs from [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)
  - Create an image list file with format: `train/<image_id>` (one per line)
  - Run: `bash scripts/download_open_images.sh --image-list <list_file> /absolute/path/to/data`
  - Or use: `bash scripts/download_open_images.sh --split train --max-images 1000 /absolute/path/to/data` for a subset
  - Images will be downloaded to `data/open_images/train/`, `data/open_images/validation/`, etc.
- **Kodak PhotoCD (evaluation)**: `bash scripts/download_kodak.sh /absolute/path/to/data`
  - Downloads and verifies the 24 Kodak images under `data/kodak/`.

## Training & Evaluation

**Important**: HFLIC requires two-stage training as recommended by the paper authors.

### Complete Workflow

#### 1. Download Datasets

**Download Open Images V7 (training dataset):**

**Option A: Download with image list file (recommended)**
```bash
# First, download image IDs from Open Images website and create a list file
# Format: train/<image_id> (one per line)
bash scripts/download_open_images.sh --image-list <path/to/image_list.txt> /absolute/path/to/data
```

**Option B: Download a subset for testing**
```bash
# Download first 1000 training images
bash scripts/download_open_images.sh --split train --max-images 1000 /absolute/path/to/data

# Download validation images
bash scripts/download_open_images.sh --split validation /absolute/path/to/data
```

**Download Kodak dataset (for evaluation):**
```bash
bash scripts/download_kodak.sh /absolute/path/to/data
```

#### 2. Stage 1: Pre-training without GAN (Required first)

```bash
bash scripts/train_stage1.sh /absolute/path/to/data \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0
```

**With additional options:**
```bash
bash scripts/train_stage1.sh /absolute/path/to/data \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --gpu_id 0 \
  --train-split train \
  --eval-split kodak
```

**Loss components**: Charbonnier (λ=2e-6) + LPIPS (λ=1) + Style (λ=1e2) + Rate/BPP (λ=0.3)  
**Output checkpoint**: `./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar`

#### 3. Stage 2: Finetuning with GAN (After Stage 1 completes)

```bash
bash scripts/train_hflic.sh /absolute/path/to/data \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0
```

**Or use a specific epoch checkpoint:**
```bash
bash scripts/train_hflic.sh /absolute/path/to/data \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_050.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0
```

**Loss components**: Charbonnier (λ=2e-6) + LPIPS (λ=1) + Style (λ=1e2) + GAN (λ=1) + Rate/BPP (λ=0.3)  
**Output checkpoint**: `./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar`

#### 4. Evaluation

**Evaluate on Kodak dataset:**
```bash
bash scripts/eval_hflic.sh /absolute/path/to/data/kodak \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split kodak \
  --test-batch-size 1 \
  --gpu_id 0
```

**Evaluate on Open Images validation:**
```bash
bash scripts/eval_hflic.sh /absolute/path/to/data/open_images \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split validation \
  --test-batch-size 1 \
  --gpu_id 0
```

### Quick Start Example

```bash
# Set your data directory
DATA_DIR="/absolute/path/to/data"

# 1. Download datasets
bash scripts/download_kodak.sh ${DATA_DIR}
bash scripts/download_open_images.sh --split train --max-images 5000 ${DATA_DIR}
bash scripts/download_open_images.sh --split validation ${DATA_DIR}

# 2. Stage 1: Pre-training (100 epochs)
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0

# 3. Stage 2: GAN finetuning (100 epochs)
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0

# 4. Evaluation on Kodak
bash scripts/eval_hflic.sh ${DATA_DIR}/kodak \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split kodak \
  --gpu_id 0
```

Modify `config/config_5group.py` or the CLI options in `utils/args.py` to adjust lambda settings, batch size, GPU selection, and dataset splits as needed.

### Training Workflow Summary

```
Stage 1 (Pre-training): train.py
├── Loss: Charbonnier + LPIPS + Style + Rate (BPP)
├── No discriminator
└── Output: checkpoint_best_loss.pth.tar

Stage 2 (GAN finetuning): train_gan.py  
├── Load: checkpoint from Stage 1
├── Loss: Charbonnier + LPIPS + Style + GAN + Rate (BPP)
├── With discriminator
└── Output: Final model with improved perceptual quality
```

## Current Status

- [x] Mirror HFLIC implementation at repository root.
- [x] Provide dataset preparation scripts and configuration templates.
- [x] Supply environment setup and training/evaluation helpers.
- [ ] Run finetuning/benchmark experiments and collect results.
- [ ] Document findings and prepare practicum deliverables.
