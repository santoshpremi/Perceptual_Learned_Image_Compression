# Open Images Dataset Download Commands

This document provides commands for downloading Open Images V7 dataset for training, validation, and testing.

## Quick Reference

### Download 1000 Images (Quick Setup)

```bash
# Set your data directory
export DATA_DIR="/path/to/your/data"

# Download 1000 training images
bash scripts/download_open_images.sh --split train --max-images 1000 ${DATA_DIR}

# Download 1000 validation images
bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR}

# Download 1000 test images
bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR}
```

### Download Full Dataset (Complete Setup)

```bash
# Set your data directory
export DATA_DIR="/path/to/your/data"

# Download FULL training dataset (all available images - millions)
bash scripts/download_open_images.sh --split train --full ${DATA_DIR}

# Download FULL validation dataset (all available images)
bash scripts/download_open_images.sh --split validation --full ${DATA_DIR}

# Download FULL test dataset (all available images)
bash scripts/download_open_images.sh --split test --full ${DATA_DIR}
```

## Complete Setup Script

Save this as `download_all_datasets.sh`:

```bash
#!/bin/bash
set -e

# Configuration
DATA_DIR="/path/to/your/data"
NUM_IMAGES=1000  # Change to "full" for complete dataset

echo "=========================================="
echo "Downloading Open Images Dataset"
echo "=========================================="

if [ "${NUM_IMAGES}" = "full" ]; then
    echo "Downloading FULL datasets..."
    
    echo "Downloading training split (this will take a long time)..."
    bash scripts/download_open_images.sh --split train --full ${DATA_DIR}
    
    echo "Downloading validation split..."
    bash scripts/download_open_images.sh --split validation --full ${DATA_DIR}
    
    echo "Downloading test split..."
    bash scripts/download_open_images.sh --split test --full ${DATA_DIR}
else
    echo "Downloading ${NUM_IMAGES} images per split..."
    
    echo "Downloading ${NUM_IMAGES} training images..."
    bash scripts/download_open_images.sh --split train --max-images ${NUM_IMAGES} ${DATA_DIR}
    
    echo "Downloading ${NUM_IMAGES} validation images..."
    bash scripts/download_open_images.sh --split validation --max-images ${NUM_IMAGES} ${DATA_DIR}
    
    echo "Downloading ${NUM_IMAGES} test images..."
    bash scripts/download_open_images.sh --split test --max-images ${NUM_IMAGES} ${DATA_DIR}
fi

echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Training images: ${DATA_DIR}/open_images/train/"
echo "Validation images: ${DATA_DIR}/open_images/validation/"
echo "Test images: ${DATA_DIR}/open_images/test/"
```

Make it executable and run:
```bash
chmod +x download_all_datasets.sh
./download_all_datasets.sh
```

## Usage in Training

### Using Kodak for Evaluation (Standard Benchmark)

```bash
# Training uses Open Images train split
# Evaluation uses Kodak dataset

bash scripts/train_stage1.sh ${DATA_DIR} \
  --train-split train \
  --eval-split kodak \
  --experiment hflic_stage1
```

### Using CLIC2022 for Evaluation (Matches HFLIC Paper)

```bash
# Training uses Open Images train split
# Evaluation uses CLIC2022 validation set

bash scripts/train_stage1.sh ${DATA_DIR} \
  --train-split train \
  --eval-split clic2022 \
  --experiment hflic_stage1
```

### Using Open Images Test for Evaluation

```bash
# Training uses Open Images train split
# Evaluation uses Open Images test split

bash scripts/train_stage1.sh ${DATA_DIR} \
  --train-split train \
  --eval-split test \
  --experiment hflic_stage1
```

## Notes

- **Full dataset download**: The full training dataset contains millions of images and will take a very long time to download. Use `--max-images` for faster setup.
- **Default behavior**: If you don't specify `--max-images` or `--full`:
  - Training: Downloads 5000 images by default
  - Validation: Downloads 1000 images by default
  - Test: Downloads 100 images by default
- **Disk space**: Full dataset requires significant disk space (hundreds of GB).
- **Network**: Full downloads require stable internet connection and may take hours/days.

## Recommended Setup for Quick Start

```bash
# Quick setup with 1000 images per split
export DATA_DIR="/path/to/your/data"

bash scripts/download_open_images.sh --split train --max-images 1000 ${DATA_DIR}
bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR}
bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR}

# Download Kodak for evaluation
bash scripts/download_kodak.sh ${DATA_DIR}
```

