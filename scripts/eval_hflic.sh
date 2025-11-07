#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 /absolute/path/to/kodak /path/to/checkpoint [additional args...]" >&2
  echo "Example: $0 /Users/me/data/kodak outputs/hflic/checkpoints/checkpoint_100.pth.tar" >&2
  exit 1
fi

DATASET_ROOT="$1"
CHECKPOINT_PATH="$2"
shift 2 || true

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
python test_5.py --dataset "${DATASET_ROOT}" --checkpoint "${CHECKPOINT_PATH}" --split kodak "$@"
