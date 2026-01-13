# Quick Download Guide

## Commands for Downloading Open Images Dataset

### Download 1000 Images (Recommended for Quick Setup)

```bash
# Set your data directory
export DATA_DIR="/path/to/your/data"

# Training: 1000 images
bash scripts/download_open_images.sh --split train --max-images 1000 ${DATA_DIR}

# Validation: 1000 images  
bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR}

# Test: 1000 images (for future evaluation)
bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR}
```

### Download FULL Dataset (Complete Setup)

```bash
# Set your data directory
export DATA_DIR="/path/to/your/data"

# Training: ALL images (millions - takes hours/days)
bash scripts/download_open_images.sh --split train --full ${DATA_DIR}

# Validation: ALL images
bash scripts/download_open_images.sh --split validation --full ${DATA_DIR}

# Test: ALL images
bash scripts/download_open_images.sh --split test --full ${DATA_DIR}
```

### Download Evaluation Datasets

```bash
# Download Kodak (24 images, standard benchmark)
bash scripts/download_kodak.sh ${DATA_DIR}

# Download CLIC2022 validation set (used in HFLIC paper)
bash scripts/download_clic2022.sh ${DATA_DIR}
```

## Complete One-Line Setup

### Quick Setup (1000 images each)
```bash
DATA_DIR="/path/to/your/data" && \
bash scripts/download_open_images.sh --split train --max-images 1000 ${DATA_DIR} && \
bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR} && \
bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR} && \
bash scripts/download_kodak.sh ${DATA_DIR} && \
bash scripts/download_clic2022.sh ${DATA_DIR}
```

### Full Setup (All images)
```bash
DATA_DIR="/path/to/your/data" && \
bash scripts/download_open_images.sh --split train --full ${DATA_DIR} && \
bash scripts/download_open_images.sh --split validation --full ${DATA_DIR} && \
bash scripts/download_open_images.sh --split test --full ${DATA_DIR} && \
bash scripts/download_kodak.sh ${DATA_DIR} && \
bash scripts/download_clic2022.sh ${DATA_DIR}
```

## Usage in Training

### Using Kodak for Evaluation (Standard Benchmark)
```bash
bash scripts/train_stage1.sh ${DATA_DIR} \
  --train-split train \
  --eval-split kodak \
  --experiment hflic_stage1
```

### Using CLIC2022 for Evaluation (Matches HFLIC Paper)
```bash
bash scripts/train_stage1.sh ${DATA_DIR} \
  --train-split train \
  --eval-split clic2022 \
  --experiment hflic_stage1
```

### Using Open Images Test for Evaluation
```bash
bash scripts/train_stage1.sh ${DATA_DIR} \
  --train-split train \
  --eval-split test \
  --experiment hflic_stage1
```

## Notes

- **No manual .txt files needed** - The script automatically fetches image IDs from S3
- **Full download**: Takes hours/days and requires hundreds of GB disk space
- **1000 images**: Good for quick testing and development
- **Default**: If you don't specify `--max-images` or `--full`:
  - Train: 5000 images
  - Validation: 1000 images  
  - Test: 100 images
