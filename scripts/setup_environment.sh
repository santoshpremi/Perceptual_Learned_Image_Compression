#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a Python virtual environment for the practicum project.
# Usage:
#   ./scripts/setup_environment.sh [--cuda 12.1]
# Supported CUDA versions: 12.1, 11.8 (defaults to cpu wheels if omitted).

CUDA_VERSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

PYTHON_BIN="python3"
VENV_DIR=".venv"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment in ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel setuptools

TORCH_BASE="torch torchvision torchaudio"
case "${CUDA_VERSION}" in
  "12.1")
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    ;;
  "11.8")
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    ;;
  "")
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "Unsupported CUDA version: ${CUDA_VERSION}" >&2
    exit 1
    ;;
 esac

echo "Installing PyTorch from ${TORCH_INDEX}"
pip install ${TORCH_BASE} --index-url "${TORCH_INDEX}"

echo "Installing Python dependencies from requirements.txt"
pip install -r requirements.txt

cat <<'EOF'
------------------------------------------------------------
Environment setup complete.
Activate the environment with:

  source .venv/bin/activate

If you need different CUDA wheels, rerun with --cuda <version>.
------------------------------------------------------------
EOF

