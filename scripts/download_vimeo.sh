#!/usr/bin/env bash
set -euo pipefail

# Automated downloader for the Vimeo-90K dataset archives.
# Usage:
#   ./download_vimeo.sh [--extract] [--septuplet-only|--triplet-only] /path/to/data
# By default downloads both vimeo_septuplet.zip (~82 GB) and vimeo_triplet.zip (~32 GB)
# into ${TARGET_ROOT}/vimeo90k/.

EXTRACT=false
DOWNLOAD_SEPTUPLET=true
DOWNLOAD_TRIPLET=true

usage() {
  cat <<'EOF' >&2
Usage: download_vimeo.sh [--extract] [--septuplet-only|--triplet-only] /path/to/data

Options:
  --extract         Extract archives after download (requires sufficient disk space)
  --septuplet-only  Download only the vimeo_septuplet archive
  --triplet-only    Download only the vimeo_triplet archive
  -h, --help        Show this help message
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extract)
      EXTRACT=true
      shift
      ;;
    --septuplet-only)
      DOWNLOAD_TRIPLET=false
      shift
      ;;
    --triplet-only)
      DOWNLOAD_SEPTUPLET=false
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

if ! $DOWNLOAD_SEPTUPLET && ! $DOWNLOAD_TRIPLET; then
  echo "Error: at least one of septuplet or triplet must be selected" >&2
  exit 1
fi

TARGET_ROOT="${ARGS[0]}"
TARGET_DIR="${TARGET_ROOT%/}/vimeo90k"
mkdir -p "${TARGET_DIR}"

BASE_URL="https://data.csail.mit.edu/tofu/dataset"

# filename expected_size_bytes
FILES=()
if $DOWNLOAD_SEPTUPLET; then
  FILES+=("vimeo_septuplet.zip 87930374183")
fi
if $DOWNLOAD_TRIPLET; then
  FILES+=("vimeo_triplet.zip 34724558235")
fi

get_size() {
  python3 - <<'PY' "$1"
import os, sys
print(os.path.getsize(sys.argv[1]))
PY
}

to_gib() {
  python3 - <<'PY' "$1"
import sys
print(f"{int(sys.argv[1]) / (1024**3):.1f}")
PY
}

for entry in "${FILES[@]}"; do
  set -- $entry
  filename="$1"
  expected_size="$2"
  url="${BASE_URL}/${filename}"
  destination="${TARGET_DIR}/${filename}"

  if [ -f "${destination}" ]; then
    actual_size=$(get_size "${destination}")
    if [ "${actual_size}" = "${expected_size}" ]; then
      echo "Skipping existing ${filename} (size OK)"
      continue
    else
      echo "Existing ${filename} has unexpected size (${actual_size}), re-downloading"
      rm -f "${destination}"
    fi
  fi

  echo "Downloading ${filename} (~$(to_gib "${expected_size}") GiB)"
  curl -L --continue-at - "${url}" -o "${destination}"

  actual_size=$(get_size "${destination}")
  if [ "${actual_size}" != "${expected_size}" ]; then
    echo "Warning: ${filename} size ${actual_size} differs from expected ${expected_size}" >&2
  else
    echo "Verified ${filename} size ${actual_size} bytes"
  fi

  if $EXTRACT; then
    echo "Extracting ${filename} (this may take a while)"
    unzip -n "${destination}" -d "${TARGET_DIR}"
  fi

done

echo "Vimeo-90K assets available under ${TARGET_DIR}"
