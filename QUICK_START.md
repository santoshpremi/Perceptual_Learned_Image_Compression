# Quick Start Guide

## Essential Commands (Copy & Paste)

### 1. Setup Environment
```bash
bash scripts/setup_environment.sh --cuda 12.1
source .venv/bin/activate
```

### 2. Set Data Directory
```bash
export DATA_DIR="/absolute/path/to/your/data"
mkdir -p ${DATA_DIR}
```

### 3. Download Datasets (1000 images each)
```bash
bash scripts/download_open_images.sh --split train --max-images 1000 ${DATA_DIR}
bash scripts/download_open_images.sh --split validation --max-images 1000 ${DATA_DIR}
bash scripts/download_open_images.sh --split test --max-images 1000 ${DATA_DIR}
bash scripts/download_kodak.sh ${DATA_DIR}
```

### 4. Stage 1 Training
```bash
bash scripts/train_stage1.sh ${DATA_DIR} \
  --experiment hflic_stage1 \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0
```

### 5. Stage 2 Training (After Stage 1 completes)
```bash
bash scripts/train_hflic.sh ${DATA_DIR} \
  --experiment hflic_stage2 \
  --checkpoint ./experiments/hflic_stage1/checkpoints/checkpoint_best_loss.pth.tar \
  --epochs 100 \
  --batch-size 8 \
  --gpu_id 0
```

### 6. Evaluation
```bash
bash scripts/eval_hflic.sh ${DATA_DIR}/kodak \
  ./experiments/hflic_stage2/checkpoints/checkpoint_best_loss.pth.tar \
  --split kodak \
  --gpu_id 0
```

## Monitor Training
```bash
# View logs
tail -f ./experiments/hflic_stage1/train_hflic_stage1.log

# TensorBoard
tensorboard --logdir ./tb_logger/hflic_stage1
```

## Check Results
```bash
# View evaluation results
cat ./coderesult/hflic_test/test_hflic_test.log

# List checkpoints
ls -lh ./experiments/hflic_stage2/checkpoints/
```

For detailed instructions, see `STEP_BY_STEP_GUIDE.md`
