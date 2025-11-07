#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}"

BASE_ROOT=""
TRAIN_ROOT=""
EVAL_ROOT=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF' >&2
Usage: train_hflic.sh [base_data_dir] [options]

If base_data_dir is provided, the script assumes the training frames are under
  base_data_dir/vimeo_test_clean
and evaluation images live under
  base_data_dir (containing the kodak/ folder).

You can override the defaults explicitly with --train-root and --eval-root.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-root)
      TRAIN_ROOT="$2"
      shift 2
      ;;
    --eval-root)
      EVAL_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      if [ -z "${BASE_ROOT}" ]; then
        BASE_ROOT="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [ -z "${TRAIN_ROOT}" ]; then
  if [ -n "${BASE_ROOT}" ]; then
    TRAIN_ROOT="${BASE_ROOT%/}/vimeo_test_clean"
  else
    TRAIN_ROOT="${PROJECT_ROOT}/data/vimeo_test_clean"
  fi
fi

if [ -z "${EVAL_ROOT}" ]; then
  if [ -n "${BASE_ROOT}" ]; then
    EVAL_ROOT="${BASE_ROOT%/}"
  else
    EVAL_ROOT="${PROJECT_ROOT}/data"
  fi
fi

if [ ! -d "${TRAIN_ROOT}" ]; then
  echo "Training root not found: ${TRAIN_ROOT}" >&2
  exit 1
fi

if [ ! -d "${EVAL_ROOT}" ]; then
  echo "Evaluation root not found: ${EVAL_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

python train_gan.py \
  --train-root "${TRAIN_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  "${EXTRA_ARGS[@]}"
