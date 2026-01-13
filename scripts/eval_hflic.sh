#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 /absolute/path/to/dataset /path/to/checkpoint [--split DATASET_NAME] [additional args...]" >&2
  echo "Example: $0 /Users/me/data/kodak outputs/hflic/checkpoints/checkpoint_100.pth.tar --split kodak" >&2
  echo "Example: $0 /Users/me/data outputs/hflic/checkpoints/checkpoint_100.pth.tar --split clic2022" >&2
  exit 1
fi

DATASET_ROOT="$1"
CHECKPOINT_PATH="$2"
shift 2 || true

# Default split to kodak for backward compatibility
SPLIT="kodak"
REMAINING_ARGS=""

# Parse --split argument if provided
while [[ $# -gt 0 ]]; do
  case $1 in
    --split)
      SPLIT="$2"
      shift 2
      ;;
    *)
      # Pass through other arguments
      REMAINING_ARGS="${REMAINING_ARGS} $1"
      shift
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

# Handle dataset path based on split type
# For backward compatibility: original code passed full path to dataset directory
# e.g., /path/to/data/kodak with --split kodak
if [ "${SPLIT}" = "kodak" ] || [ "${SPLIT}" = "clic2022" ]; then
  # Normalize path (remove trailing slash)
  DATASET_ROOT_NORM="${DATASET_ROOT%/}"
  # Check if DATASET_ROOT ends with the split name (exact match, not substring)
  # e.g., /path/to/data/kodak ends with /kodak
  if [[ "${DATASET_ROOT_NORM}" == */"${SPLIT}" ]]; then
    # DATASET_ROOT is already the full path to the dataset directory
    # Pass it as-is (Python code will handle extracting parent directory)
    DATASET_PATH="${DATASET_ROOT_NORM}"
  else
    # DATASET_ROOT is the parent directory, construct full path
    DATASET_PATH="${DATASET_ROOT_NORM}/${SPLIT}"
  fi
  python test_5.py --dataset "${DATASET_PATH}" --checkpoint "${CHECKPOINT_PATH}" --split "${SPLIT}" ${REMAINING_ARGS}
else
  # For Open Images splits, pass the parent directory as-is
  python test_5.py --dataset "${DATASET_ROOT}" --checkpoint "${CHECKPOINT_PATH}" --split "${SPLIT}" ${REMAINING_ARGS}
fi
