# Step-by-Step Guide to Run the Project

Complete guide to set up and run the Perceptual Learned Image Compression project from scratch.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 50GB free disk space (for 1000 images setup)
- Stable internet connection

---

## Understanding Training, Validation, and Evaluation

### Key Concepts

**1. Training Phase** (During Each Epoch)
- **When**: Happens during the training loop, before validation
- **Data**: Open Images train split (`--train-split train`)
- **Purpose**: Learn compression by updating model weights
- **Process**: Forward pass → Compute loss → Backward pass → Update weights
- **Mode**: `model.train()` (enables gradients)

**2. Validation Phase** (During Training, After Each Epoch)
- **When**: Happens after each epoch completes
- **Data**: Selected via `--eval-split` flag (Kodak, validation, or test)
- **Purpose**: Monitor progress, detect overfitting, select best model
- **Process**: Forward pass only → Compute metrics → Save best checkpoint
- **Mode**: `model.eval()` (disables gradients, no weight updates)

**3. Evaluation Phase** (After Training Completes)
- **When**: Separate script run after training finishes
- **Data**: Selected via `--split` flag in evaluation script
- **Purpose**: Final performance assessment, generate results
- **Process**: Load best model → Compress/decompress → Compute final metrics
- **Mode**: `model.eval()` (no training)

### Data Options Summary

| Phase | Flag | Options | Recommended | Purpose |
|-------|------|---------|-------------|---------|
| **Training** | `--train-split` | `train` | `train` | Learn from large dataset |
| **Validation** | `--eval-split` | `kodak`, `clic2022`, `validation`, `test` | `kodak` | Fast validation, standard benchmark |
| **Evaluation** | `--split` | `kodak`, `clic2022`, `test`, `validation` | `kodak` | Final assessment, paper results |

### Recommended Data Strategy

**For Training:**
- Use Open Images train split (large, diverse dataset)
- Command: `--train-split train`

**For Validation (During Training):**
- Use Kodak dataset (fast, standard benchmark) - Command: `--eval-split kodak`
- Use CLIC2022 validation set (matches HFLIC paper) - Command: `--eval-split clic2022`
- Alternative: Open Images validation if you want larger validation set

**For Evaluation (After Training):**
- Use Kodak for standard benchmarking (recommended) - Command: `--split kodak`
- Use CLIC2022 for HFLIC paper comparison - Command: `--split clic2022`
- Alternative: Open Images test for comprehensive evaluation

---

## Step 1: Clone and Navigate to Project

```bash
# If cloning from GitHub
git clone <your-repo-url>
cd Perceptual-Learned-Image-Compression

# Or if you already have the project
cd /path/to/Perceptual-Learned-Image-Compression
```

---

## Step 2: Set Up Python Environment

### Option A: Automated Setup (Recommended)

```bash
# For CUDA 12.1
bash scripts/setup_environment.sh --cuda 12.1

# For CUDA 11.8
bash scripts/setup_environment.sh --cuda 11.8

# For CPU only
bash scripts/setup_environment.sh
```

### Option B: Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose based on your system)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio

# Install project dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 3: Set Data Directory

```bash
# Set your data directory (use absolute path)
export DATA_DIR="/absolute/path/to/your/data"

# Example:
# export DATA_DIR="/Users/yourname/data"
# OR
# export DATA_DIR="/home/yourname/data"
```

**Note:** Create the directory if it doesn't exist:
```bash
mkdir -p ${DATA_DIR}
```

---

## Step 4: Download Datasets

### Quick Setup (1000 images each - Recommended for Testing)

```bash
# Download 1000 training images
bash scripts/download_open_images.sh --split train --max-images 1000 ${DATA_DIR}

# Download 1000 validation images
bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR}

# Download 1000 test images (for future evaluation)
bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR}

# Download Kodak dataset (24 images for evaluation)
bash scripts/download_kodak.sh ${DATA_DIR}

# Download CLIC2022 validation set (used in HFLIC paper)
bash scripts/download_clic2022.sh ${DATA_DIR}
```

**Expected output:**
- Training images: `${DATA_DIR}/open_images/train/` (~1000 images)
- Validation images: `${DATA_DIR}/open_images/validation/` (~1000 images)
- Test images: `${DATA_DIR}/open_images/test/` (~1000 images)
- Kodak images: `${DATA_DIR}/kodak/` (24 images)
- CLIC2022 images: `${DATA_DIR}/clic2022/` (~102 validation images)

### Full Setup (All Images - For Production)

```bash
# WARNING: This will download millions of images and take hours/days
# Requires hundreds of GB disk space

bash scripts/download_open_images.sh --split train --full ${DATA_DIR}
bash scripts/download_open_images.sh --split validation --full ${DATA_DIR}
bash scripts/download_open_images.sh --split test --full ${DATA_DIR}
bash scripts/download_kodak.sh ${DATA_DIR}
```

**Verify downloads:**
```bash
# Check number of images downloaded
echo "Training images: $(find ${DATA_DIR}/open_images/train -name '*.jpg' | wc -l)"
echo "Validation images: $(find ${DATA_DIR}/open_images/validation -name '*.jpg' | wc -l)"
echo "Test images: $(find ${DATA_DIR}/open_images/test -name '*.jpg' | wc -l)"
echo "Kodak images: $(find ${DATA_DIR}/kodak -name '*.png' | wc -l)"
echo "CLIC2022 images: $(find ${DATA_DIR}/clic2022 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)"
```

---

## Step 5: Stage 1 Training (Pre-training without GAN)

This stage trains the compression model using perceptual losses (Charbonnier + LPIPS + Style + Rate).

### Understanding Training, Validation, and Evaluation

**During Training (Each Epoch):**
1. **Training Phase**: Model learns from Open Images train split (`--train-split train`)
   - Forward pass: Compress and decompress images
   - Compute loss: Charbonnier + LPIPS + Style + Rate (BPP)
   - Backward pass: Update model weights
   - Purpose: Learn compression from training data

2. **Validation Phase**: Model is evaluated on validation dataset (`--eval-split`)
   - Forward pass only (no weight updates)
   - Compute metrics: Loss, PSNR, MS-SSIM, BPP
   - Save reconstructed images
   - Save best checkpoint if validation loss improves
   - Purpose: Monitor progress and select best model

**After Training:**
3. **Evaluation Phase**: Final assessment using separate script (`test_5.py`)
   - Load best checkpoint
   - Compress and decompress test images
   - Compute final metrics
   - Save compressed bitstreams
   - Purpose: Final performance assessment

### Validation Data Options (`--eval-split`)

The `--eval-split` flag determines which dataset is used for **validation during training**:

- **`--eval-split kodak`** (Recommended - Default)
  - Uses Kodak PhotoCD dataset (24 images)
  - Standard benchmark for image compression
  - Small dataset = fast validation
  - Located at: `${DATA_DIR}/kodak/`
  - **Best for**: Quick validation, standard benchmarking

- **`--eval-split clic2022`**
  - Uses CLIC2022 validation set (~102 images)
  - Matches HFLIC paper evaluation setup
  - Located at: `${DATA_DIR}/clic2022/`
  - **Best for**: Comparing with HFLIC paper results

- **`--eval-split validation`**
  - Uses Open Images validation split
  - Larger dataset = slower validation
  - Located at: `${DATA_DIR}/open_images/validation/`
  - **Best for**: More comprehensive validation during training
  - **Requires**: Download validation images first

- **`--eval-split test`**
  - Uses Open Images test split
  - Located at: `${DATA_DIR}/open_images/test/`
  - **Best for**: Using test set for validation (not recommended for final evaluation)
  - **Requires**: Download test images first

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Basic command (uses default: eval-split kodak)
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0

# With validation on Kodak (recommended - standard benchmark)
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --gpu_id 0 \
  --train-split train \
  --eval-split kodak \
  --num-workers 4

# With validation on Open Images validation split
# (Requires: bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR})
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --gpu_id 0 \
  --train-split train \
  --eval-split validation \
  --num-workers 4
```

**What happens:**
- **Training**: Uses Open Images train split (`--train-split train`)
  - Iterates through all training images
  - Updates model weights based on loss
  - Logs training metrics every 100 steps

- **Validation**: Uses selected validation dataset (`--eval-split`)
  - Runs after each epoch completes
  - Computes validation loss, PSNR, MS-SSIM, BPP
  - Saves reconstructed images to `./experiments/hflic_stage1/val_images/`
  - Saves best checkpoint if validation loss improves
  - Best checkpoint: `./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar`

**Monitor training:**
```bash
# View training logs
tail -f ./experiments/hflic_stage1/train_hflic_stage1.log

# View TensorBoard (in another terminal)
tensorboard --logdir ./tb_logger/hflic_stage1
# Then open http://localhost:6006 in browser
```

**Expected duration:**
- 1000 images: ~2-4 hours (depending on GPU)
- Full dataset: Days

**Check progress:**
```bash
# List checkpoints
ls -lh ./experiments/hflic_stage1/checkpoints/

# Check latest log
tail -20 ./experiments/hflic_stage1/train_hflic_stage1.log
```

---

## Step 6: Stage 2 Training (GAN Finetuning)

**IMPORTANT:** Wait for Stage 1 to complete before starting Stage 2.

### Validation During Stage 2

Same validation options apply as Stage 1:
- **`--eval-split kodak`** (Recommended): Fast validation on standard benchmark
- **`--eval-split validation`**: Validation on Open Images validation split
- **`--eval-split test`**: Validation on Open Images test split

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Basic command (uses default: eval-split kodak)
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0

# With validation on Kodak (recommended)
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --gpu_id 0 \
  --train-split train \
  --eval-split kodak \
  --num-workers 4

# With validation on Open Images validation split
# (Requires: bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR})
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --gpu_id 0 \
  --train-split train \
  --eval-split validation \
  --num-workers 4
```

**What happens:**
- **Loads checkpoint from Stage 1**: Continues training from best Stage 1 model
- **Adds discriminator network**: GAN discriminator for adversarial training
- **Training**: Uses Open Images train split
  - Generator (compression model) + Discriminator training
  - Loss: Charbonnier + LPIPS + Style + **GAN** + Rate (BPP)
- **Validation**: Uses selected validation dataset (`--eval-split`)
  - Runs after each epoch
  - Computes validation metrics
  - Saves best checkpoint
- **Saves checkpoints in**: `./experiments/hflic_stage2/checkpoints/`
- **Best checkpoint**: `./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar`

**Monitor training:**
```bash
# View training logs
tail -f ./experiments/hflic_stage2/train_hflic_stage2.log

# View TensorBoard
tensorboard --logdir ./tb_logger/hflic_stage2
```

---

## Step 7: Evaluation (Final Assessment)

**Note:** Evaluation is different from validation:
- **Validation**: Happens during training (after each epoch) to monitor progress
- **Evaluation**: Happens after training completes to assess final performance

### Evaluation Data Options (`--split`)

The `--split` flag in the evaluation script determines which dataset to use for **final evaluation**:

- **`--split kodak`** (Recommended - Standard Benchmark)
  - Uses Kodak PhotoCD dataset (24 images)
  - Standard benchmark for image compression research
  - Allows comparison with published results
  - Located at: `${DATA_DIR}/kodak/`
  - **Best for**: Final evaluation, paper results, benchmarking

- **`--split test`**
  - Uses Open Images test split
  - Larger dataset for comprehensive evaluation
  - Located at: `${DATA_DIR}/open_images/test/`
  - **Best for**: Large-scale evaluation, production assessment
  - **Requires**: Download test images first

- **`--split validation`**
  - Uses Open Images validation split
  - Located at: `${DATA_DIR}/open_images/validation/`
  - **Best for**: Additional evaluation on validation set
  - **Note**: Usually validation is used during training, but can also be used for final eval

### Evaluate on Kodak Dataset (Recommended - Standard Benchmark)

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Evaluate on Kodak (standard benchmark)
bash scripts/eval_hflic.sh ${DATA_DIR}/kodak \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split kodak \
  --test-batch-size 1 \
  --gpu_id 0
```

**What happens:**
- Loads the best trained model from Stage 2
- Compresses all Kodak images (creates bitstreams)
- Decompresses images
- Computes final metrics: PSNR, MS-SSIM, BPP
- Saves compressed bitstreams and reconstructed images

**Output:**
- Results saved in `./coderesult/hflic_test/`
- Metrics logged in: `./coderesult/hflic_test/test_hflic_test.log`
- Compressed bitstreams: `./coderesult/hflic_test/codestream/*/`
- Reconstructed images: `./coderesult/hflic_test/codestream/*/*_rec.png`

### Evaluate on Open Images Test Split

```bash
# Requires: bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR}

bash scripts/eval_hflic.sh ${DATA_DIR}/open_images \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split test \
  --test-batch-size 1 \
  --gpu_id 0
```

**What happens:**
- Same process as Kodak evaluation
- Evaluates on larger test dataset
- Provides more comprehensive performance assessment

### Evaluate on Open Images Validation Split

```bash
# Requires: bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR}

bash scripts/eval_hflic.sh ${DATA_DIR}/open_images \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split validation \
  --test-batch-size 1 \
  --gpu_id 0
```

**View results:**
```bash
# Check evaluation log
cat ./coderesult/hflic_test/test_hflic_test.log

# View reconstructed images
ls ./coderesult/hflic_test/codestream/*/

# Check metrics summary
grep "Avg" ./coderesult/hflic_test/test_hflic_test.log
```

---

## Complete Workflow Script

Save this as `run_complete_pipeline.sh`:

```bash
#!/bin/bash
set -e

# Configuration
DATA_DIR="/absolute/path/to/your/data"
GPU_ID=0
EPOCHS=100
BATCH_SIZE=8
NUM_IMAGES=1000  # Change to "full" for complete dataset

echo "=========================================="
echo "Complete Pipeline Execution"
echo "=========================================="

# Step 1: Activate environment
echo "Step 1: Activating virtual environment..."
source .venv/bin/activate

# Step 2: Download datasets
echo ""
echo "Step 2: Downloading datasets..."
if [ "${NUM_IMAGES}" = "full" ]; then
    echo "Downloading FULL datasets (this will take a long time)..."
    bash scripts/download_open_images.sh --split train --full ${DATA_DIR}
    bash scripts/download_open_images.sh --split validation --full ${DATA_DIR}
    bash scripts/download_open_images.sh --split test --full ${DATA_DIR}
else
    echo "Downloading ${NUM_IMAGES} images per split..."
    bash scripts/download_open_images.sh --split train --max-images ${NUM_IMAGES} ${DATA_DIR}
    bash scripts/download_open_images.sh --split validation --max-images ${NUM_IMAGES} ${DATA_DIR}
    bash scripts/download_open_images.sh --split test --max-images ${NUM_IMAGES} ${DATA_DIR}
fi
bash scripts/download_kodak.sh ${DATA_DIR}

# Step 3: Stage 1 Training
echo ""
echo "Step 3: Stage 1 Training (Pre-training without GAN)..."
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --learning-rate 1e-4 \
  --gpu_id ${GPU_ID} \
  --train-split train \
  --eval-split kodak

# Step 4: Stage 2 Training
echo ""
echo "Step 4: Stage 2 Training (GAN Finetuning)..."
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --learning-rate 1e-4 \
  --gpu_id ${GPU_ID} \
  --train-split train \
  --eval-split kodak

# Step 5: Evaluation
echo ""
echo "Step 5: Evaluation on Kodak dataset..."
bash scripts/eval_hflic.sh ${DATA_DIR}/kodak \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split kodak \
  --test-batch-size 1 \
  --gpu_id ${GPU_ID}

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved in:"
echo "  - Training logs: ./experiments/hflic_stage1/ and ./experiments/hflic_stage2/"
echo "  - Evaluation results: ./coderesult/hflic_test/"
echo "  - Best checkpoint: ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar"
```

Make it executable and run:
```bash
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 4  # or 2, or 1
```

### Dataset Not Found
```bash
# Verify data directory
ls ${DATA_DIR}/open_images/train/
ls ${DATA_DIR}/kodak/

# Check paths in scripts
echo ${DATA_DIR}
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Training Stuck or Slow
```bash
# Check GPU usage
nvidia-smi

# Reduce number of workers
--num-workers 2  # instead of 4
```

### Resume Training from Checkpoint
```bash
# Stage 1 resume
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_050.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0

# Stage 2 resume
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage2/checkpoints/checkpoint_050.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0
```

---

## Quick Reference

```bash
# 1. Setup
source .venv/bin/activate
export DATA_DIR="/path/to/data"

# 2. Download (1000 images)
bash scripts/download_open_images.sh --split train --max-images 1000 ${DATA_DIR}
bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR}
bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR}
bash scripts/download_kodak.sh ${DATA_DIR}

# 3. Stage 1 (with validation on Kodak - recommended)
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --eval-split kodak \
  --gpu_id 0

# 4. Stage 2 (with validation on Kodak - recommended)
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --eval-split kodak \
  --gpu_id 0

# 5. Final Evaluation
# Evaluate on Kodak (standard benchmark)
bash scripts/eval_hflic.sh ${DATA_DIR}/kodak \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split kodak \
  --gpu_id 0

# Evaluate on CLIC2022 (matches HFLIC paper)
bash scripts/eval_hflic.sh ${DATA_DIR} \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split clic2022 \
  --gpu_id 0

# Alternative: Final Evaluation (on Open Images test split)
bash scripts/eval_hflic.sh ${DATA_DIR}/open_images \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split test \
  --gpu_id 0
```

### Data Usage Summary

| Phase | Data Used | Purpose | Flag |
|-------|-----------|---------|------|
| **Training** | Open Images train split | Learn compression | `--train-split train` |
| **Validation** (during training) | Kodak / CLIC2022 / Open Images validation | Monitor progress, select best model | `--eval-split kodak/clic2022/validation` |
| **Evaluation** (after training) | Kodak / CLIC2022 / Open Images test | Final performance assessment | `--split kodak/clic2022/test` |

**Recommended Setup:**
- **Training**: Open Images train split (large dataset)
- **Validation**: Kodak (fast, standard benchmark) or CLIC2022 (matches HFLIC paper)
- **Evaluation**: Kodak (standard benchmark), CLIC2022 (HFLIC paper comparison), or Open Images test (comprehensive)

---

## Expected File Structure After Running

```
Perceptual-Learned-Image-Compression/
├── data/
│   ├── kodak/                    # 24 Kodak images
│   └── open_images/
│       ├── train/                 # Training images
│       ├── validation/            # Validation images
│       └── test/                  # Test images
├── experiments/
│   ├── hflic_stage1/             # Stage 1 training outputs
│   │   ├── checkpoints/          # Model checkpoints
│   │   ├── val_images/           # Validation images
│   │   └── train_hflic_stage1.log
│   └── hflic_stage2/             # Stage 2 training outputs
│       ├── checkpoints/          # Final model checkpoints
│       ├── val_images/           # Validation images
│       └── train_hflic_stage2.log
├── coderesult/
│   └── hflic_test/               # Evaluation results
│       ├── codestream/           # Compressed images
│       └── test_hflic_test.log
└── tb_logger/                    # TensorBoard logs
    ├── hflic_stage1/
    └── hflic_stage2/
```

---

## Next Steps

After successful training and evaluation:
1. Analyze results in `./coderesult/hflic_test/`
2. Compare metrics (PSNR, MS-SSIM, BPP) with baseline methods
3. Visualize reconstructed images
4. Adjust hyperparameters in `config/config_5group.py` if needed
5. Train on full dataset for production use

