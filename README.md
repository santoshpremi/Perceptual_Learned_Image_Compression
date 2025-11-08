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

- **Vimeo-90K (training/finetuning)**: `bash scripts/download_vimeo.sh /absolute/path/to/data`
  - Requests access from the dataset authors, prepares directory scaffold, and documents extraction steps.
  - Training uses the `vimeo_test_clean` split (centre frame `im4.png` per sequence).
- **Kodak PhotoCD (evaluation)**: `bash scripts/download_kodak.sh /absolute/path/to/data`
  - Downloads and verifies the 24 Kodak images under `data/kodak/`.

Update `configs/datasets.yaml` to point to the actual dataset locations and maintain train/val filelists under `configs/filelists/`.

## Training & Evaluation

1. **Training / Finetuning**
   ```
   bash scripts/train_hflic.sh /absolute/path/to/data --experiment hflic_vimeo --epochs 200
   ```
   The script infers `vimeo_test_clean/` for training frames and `kodak/` for evaluation, sets `PYTHONPATH` to the project root, and executes `train_gan.py` with any additional CLI flags.
2. **Evaluation (Kodak)**
   ```
   bash scripts/eval_hflic.sh /absolute/path/to/kodak outputs/hflic/checkpoints/checkpoint_200.pth.tar
   ```
   Computes PSNR/SSIM/LPIPS and saves reconstructions following the official workflow.
3. **Evaluation (Vimeo-90K test clean)**
   ```
   bash scripts/eval_hflic.sh /absolute/path/to/vimeo_test_clean outputs/hflic/checkpoints/checkpoint_200.pth.tar \
     --split vimeo_test_clean --list-file sep_testlist.txt --frame-index 4 --test-batch-size 1
   ```
   Uses the official `sep_testlist.txt` file and centre frame (`im4.png`) for each sequence.

Modify `config/config_5group.py` or the CLI options in `utils/args.py` to adjust lambda settings, batch size, GPU selection, and dataset splits as needed.

## Current Status

- [x] Mirror HFLIC implementation at repository root.
- [x] Provide dataset preparation scripts and configuration templates.
- [x] Supply environment setup and training/evaluation helpers.
- [ ] Run finetuning/benchmark experiments and collect results.
- [ ] Document findings and prepare practicum deliverables.
