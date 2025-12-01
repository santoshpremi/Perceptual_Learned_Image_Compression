#!/usr/bin/env bash
set -euo pipefail

# Automated downloader for Open Images V7 dataset
# Usage:
#   ./download_open_images.sh [--split train|validation|test] [--max-images N] /path/to/data

SPLIT="train"
MAX_IMAGES=""
IMAGE_LIST_FILE=""

usage() {
  cat <<'EOF' >&2
Usage: download_open_images.sh [options] /path/to/data

Options:
  --split SPLIT        Dataset split: train, validation, or test (default: train)
  --max-images N       Maximum number of images to download (optional)
  --image-list FILE    Path to file containing image IDs to download (one per line, format: split/image_id)
  -h, --help           Show this help message

Examples:
  # Download all training images
  ./download_open_images.sh /path/to/data
  
  # Download first 1000 validation images
  ./download_open_images.sh --split validation --max-images 1000 /path/to/data
  
  # Download specific images from a list file
  ./download_open_images.sh --image-list image_ids.txt /path/to/data
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --max-images)
      MAX_IMAGES="$2"
      shift 2
      ;;
    --image-list)
      IMAGE_LIST_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ${#ARGS[@]} -ne 1 ]; then
  usage
  exit 1
fi

TARGET_ROOT="${ARGS[0]}"
DOWNLOAD_DIR="${TARGET_ROOT%/}/open_images"
mkdir -p "${DOWNLOAD_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADER_SCRIPT="${SCRIPT_DIR}/downloader.py"

if [ ! -f "${DOWNLOADER_SCRIPT}" ]; then
  echo "Error: downloader.py not found at ${DOWNLOADER_SCRIPT}" >&2
  echo "Please ensure the Open Images downloader script is present." >&2
  exit 1
fi

# Create image list file if needed
if [ -n "${IMAGE_LIST_FILE}" ]; then
  if [ ! -f "${IMAGE_LIST_FILE}" ]; then
    echo "Error: Image list file not found: ${IMAGE_LIST_FILE}" >&2
    exit 1
  fi
  LIST_FILE="${IMAGE_LIST_FILE}"
else
  # Generate image list from annotations (simplified - you may need to download annotations first)
  echo "Note: For full dataset download, you may need to:"
  echo "1. Download image IDs from: https://storage.googleapis.com/openimages/web/download_v7.html"
  echo "2. Create a list file with format: ${SPLIT}/<image_id>"
  echo "3. Use --image-list option with that file"
  echo ""
  echo "For now, creating a placeholder list file..."
  
  LIST_FILE="${DOWNLOAD_DIR}/image_list_${SPLIT}.txt"
  if [ ! -f "${LIST_FILE}" ]; then
    echo "# Placeholder list file for ${SPLIT} split" > "${LIST_FILE}"
    echo "# Add image IDs in format: ${SPLIT}/<image_id>" >> "${LIST_FILE}"
    echo "# One per line" >> "${LIST_FILE}"
    echo ""
    echo "Please download the image IDs from Open Images website and populate this file."
    echo "Then run this script again with --image-list ${LIST_FILE}"
    exit 1
  fi
fi

# Limit number of images if specified
if [ -n "${MAX_IMAGES}" ]; then
  TEMP_LIST="${LIST_FILE}.limited"
  head -n "${MAX_IMAGES}" "${LIST_FILE}" > "${TEMP_LIST}"
  LIST_FILE="${TEMP_LIST}"
fi

echo "Downloading Open Images ${SPLIT} split to ${DOWNLOAD_DIR}"
echo "Using image list: ${LIST_FILE}"

python3 "${DOWNLOADER_SCRIPT}" "${LIST_FILE}" \
  --download_folder="${DOWNLOAD_DIR}/${SPLIT}" \
  --num_processes=5

# Clean up temporary list if created
if [ -n "${MAX_IMAGES}" ] && [ -f "${TEMP_LIST}" ]; then
  rm -f "${TEMP_LIST}"
fi

echo "Open Images ${SPLIT} split downloaded to ${DOWNLOAD_DIR}/${SPLIT}"



