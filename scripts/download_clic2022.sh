#!/usr/bin/env bash
set -euo pipefail

# Automated downloader for the CLIC2022 validation dataset.
# CLIC2022 is used for evaluation/validation in the HFLIC paper.
# Usage:
#   ./download_clic2022.sh /path/to/data

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/data" >&2
  exit 1
fi

TARGET_ROOT="$1"
TARGET_DIR="${TARGET_ROOT%/}/clic2022"
mkdir -p "${TARGET_DIR}"

# CLIC2022 validation set download URL
# Note: Update this URL if the official source changes
# CLIC datasets are typically hosted at compression.cc or similar
BASE_URL="https://data.vision.ee.ethz.ch/cvl/clic2022/validation"

echo "Downloading CLIC2022 validation dataset..."
echo "Target directory: ${TARGET_DIR}"

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
else
    echo "Error: Neither wget nor curl found. Please install one of them." >&2
    exit 1
fi

# Download validation images
# CLIC2022 validation set typically contains ~102 images
# The exact file list may vary - this is a template that can be updated
echo "Note: CLIC2022 validation set download"
echo "Please check https://compression.cc/ for the official download link"
echo "or download manually from: ${BASE_URL}"
echo ""
echo "If you have the images, place them in: ${TARGET_DIR}"
echo ""
echo "For automated download, you may need to:"
echo "1. Visit https://compression.cc/"
echo "2. Download CLIC2022 validation set"
echo "3. Extract to ${TARGET_DIR}"

# Alternative: If you have a file list, you can download individual images
# Uncomment and modify the following section if you have the image list:
#
# if [ -f "clic2022_validation_list.txt" ]; then
#     while IFS= read -r filename; do
#         url="${BASE_URL}/${filename}"
#         destination="${TARGET_DIR}/${filename}"
#         
#         if [ -f "${destination}" ]; then
#             echo "Skipping existing ${filename}"
#             continue
#         fi
#         
#         echo "Downloading ${filename}"
#         if [ "${DOWNLOAD_CMD}" = "wget" ]; then
#             wget "${url}" -O "${destination}"
#         else
#             curl -L "${url}" -o "${destination}"
#         fi
#     done < clic2022_validation_list.txt
# fi

# Check if directory has images
IMAGE_COUNT=$(find "${TARGET_DIR}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')

if [ "${IMAGE_COUNT}" -gt 0 ]; then
    echo "CLIC2022 dataset ready at ${TARGET_DIR} (${IMAGE_COUNT} images found)"
else
    echo "CLIC2022 directory created at ${TARGET_DIR}"
    echo "Please download the validation images manually and place them in this directory"
    echo "Expected: ~102 validation images"
fi
