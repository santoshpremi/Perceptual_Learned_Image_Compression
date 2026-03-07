#!/bin/bash
# Phase 2 BASELINE Training: Original HFLIC with GAN
# Loss = 3e-4 × Charbonnier + 2.0 × LPIPS + Style + GAN + BPP
# NO MSE, NO VSI - Original HFLIC for fair comparison

echo "=========================================="
echo "Phase 2 BASELINE: Original HFLIC with GAN"
echo "=========================================="

# Get absolute path of data directory
DATA_DIR=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")

# Set default training and evaluation roots
TRAIN_ROOT="${DATA_DIR}/open_images"
EVAL_ROOT="${DATA_DIR}"

echo "Training root: $TRAIN_ROOT"
echo "Evaluation root: $EVAL_ROOT"
echo ""

# Run training with baseline script
python train_gan_baseline.py \
    --train_root "$TRAIN_ROOT" \
    --train_split train \
    --eval_root "$EVAL_ROOT" \
    --eval_split kodak \
    "${@:2}"
