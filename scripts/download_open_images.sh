#!/usr/bin/env bash
set -euo pipefail

# Automated downloader for Open Images V7 dataset
# Usage:
#   ./download_open_images.sh [--split train|validation|test] [--max-images N] /path/to/data

SPLIT="train"
MAX_IMAGES=""
IMAGE_LIST_FILE=""
DOWNLOAD_FULL=false
DEFAULT_MAX_IMAGES=5000  # Default number of images to download if --max-images not specified

usage() {
  cat <<'EOF' >&2
Usage: download_open_images.sh [options] /path/to/data

Options:
  --split SPLIT        Dataset split: train, validation, or test (default: train)
  --max-images N       Maximum number of images to download (default: 5000 for train, all for validation/test)
  --full               Download full dataset (all available images) - overrides --max-images
  --image-list FILE    Path to file containing image IDs to download (one per line, format: split/image_id)
  -h, --help           Show this help message

Examples:
  # Download default number of training images (5000)
  ./download_open_images.sh /path/to/data
  
  # Download first 1000 training images
  ./download_open_images.sh --split train --max-images 1000 /path/to/data
  
  # Download full training dataset (all images)
  ./download_open_images.sh --split train --full /path/to/data
  
  # Download first 1000 validation images
  ./download_open_images.sh --split validation --max-images 1000 /path/to/data
  
  # Download full validation dataset
  ./download_open_images.sh --split validation --full /path/to/data
  
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
    --full)
      DOWNLOAD_FULL=true
      shift
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

# Function to generate image IDs automatically
download_image_ids() {
  local split="$1"
  local output_file="$2"
  local max_images="${3:-}"
  
  echo "Generating image IDs for ${split} split..."
  
  # Use the helper Python script to get image IDs
  # This script tries to fetch from S3 first, then falls back to sample IDs
  GET_IDS_SCRIPT="${SCRIPT_DIR}/get_open_images_ids.py"
  
  if [ ! -f "${GET_IDS_SCRIPT}" ]; then
    echo "Error: get_open_images_ids.py not found at ${GET_IDS_SCRIPT}" >&2
    exit 1
  fi
  
  # Generate image list using Python helper
  if [ "${DOWNLOAD_FULL}" = "true" ]; then
    # Download full dataset - pass 0 to indicate "all"
    python3 "${GET_IDS_SCRIPT}" "${split}" "${output_file}" "0"
  elif [ -n "${max_images}" ] && [ "${max_images}" != "" ]; then
    python3 "${GET_IDS_SCRIPT}" "${split}" "${output_file}" "${max_images}"
  else
    # For train, use default max; for validation/test, get all available
    if [ "${split}" = "train" ]; then
      python3 "${GET_IDS_SCRIPT}" "${split}" "${output_file}" "${DEFAULT_MAX_IMAGES}"
    else
      # For validation/test, try to get a reasonable number (1000 for validation, 100 for test)
      local default_val_test=1000
      if [ "${split}" = "test" ]; then
        default_val_test=100
      fi
      python3 "${GET_IDS_SCRIPT}" "${split}" "${output_file}" "${default_val_test}"
    fi
  fi
  
  if [ $? -ne 0 ] || [ ! -s "${output_file}" ]; then
    echo "Error: Failed to generate image IDs." >&2
    exit 1
  fi
  
  local num_images=$(wc -l < "${output_file}" | tr -d ' ')
  echo "Generated ${num_images} image IDs for ${split} split."
}

# Create image list file if needed
if [ -n "${IMAGE_LIST_FILE}" ]; then
  if [ ! -f "${IMAGE_LIST_FILE}" ]; then
    echo "Error: Image list file not found: ${IMAGE_LIST_FILE}" >&2
    exit 1
  fi
  LIST_FILE="${IMAGE_LIST_FILE}"
  echo "Using provided image list file: ${LIST_FILE}"
else
  # Automatically download image IDs from Open Images metadata
  LIST_FILE="${DOWNLOAD_DIR}/image_list_${SPLIT}.txt"
  
  # Check if list file already exists and has content
  if [ -f "${LIST_FILE}" ] && [ -s "${LIST_FILE}" ]; then
    echo "Image list file already exists: ${LIST_FILE}"
    echo "Skipping metadata download. To re-download, delete this file first."
  else
    # Download image IDs automatically
    download_image_ids "${SPLIT}" "${LIST_FILE}" "${MAX_IMAGES}"
  fi
fi

# Limit number of images if specified (only if not already limited during download and not --full)
if [ -n "${MAX_IMAGES}" ] && [ -z "${IMAGE_LIST_FILE}" ] && [ "${DOWNLOAD_FULL}" != "true" ]; then
  # Already limited during download, but double-check
  TEMP_LIST="${LIST_FILE}.limited"
  head -n "${MAX_IMAGES}" "${LIST_FILE}" > "${TEMP_LIST}"
  LIST_FILE="${TEMP_LIST}"
fi

# Filter out comment lines and empty lines from list file
TEMP_CLEAN_LIST="${LIST_FILE}.clean"
grep -v '^#' "${LIST_FILE}" | grep -v '^$' | grep -E "^${SPLIT}/" > "${TEMP_CLEAN_LIST}" || {
  echo "Warning: No valid image IDs found in list file. Checking format..." >&2
  # If no matches, try without split prefix check
  grep -v '^#' "${LIST_FILE}" | grep -v '^$' > "${TEMP_CLEAN_LIST}" || {
    echo "Error: Image list file is empty or invalid." >&2
    exit 1
  }
}
LIST_FILE="${TEMP_CLEAN_LIST}"

NUM_IMAGES=$(wc -l < "${LIST_FILE}" | tr -d ' ')
if [ "${NUM_IMAGES}" -eq 0 ]; then
  echo "Error: No valid image IDs found in list file." >&2
  exit 1
fi

echo "=========================================="
echo "Downloading Open Images ${SPLIT} split"
echo "=========================================="
echo "Target directory: ${DOWNLOAD_DIR}/${SPLIT}"
echo "Number of images: ${NUM_IMAGES}"
echo "Using image list: ${LIST_FILE}"
echo ""

# Check if boto3 is available
python3 -c "import boto3" 2>/dev/null || {
  echo "Installing boto3 (required for downloading from S3)..."
  pip install boto3 --quiet
}

python3 "${DOWNLOADER_SCRIPT}" "${LIST_FILE}" \
  --download_folder="${DOWNLOAD_DIR}/${SPLIT}" \
  --num_processes=5

# Clean up temporary files
rm -f "${TEMP_CLEAN_LIST}"
if [ -n "${MAX_IMAGES}" ] && [ -f "${LIST_FILE}.limited" ]; then
  rm -f "${LIST_FILE}.limited"
fi

# Verify download
DOWNLOADED_COUNT=$(find "${DOWNLOAD_DIR}/${SPLIT}" -type f \( -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo "Downloaded ${DOWNLOADED_COUNT} images to ${DOWNLOAD_DIR}/${SPLIT}"
echo "Open Images ${SPLIT} split is ready for use."



