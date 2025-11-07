#!/usr/bin/env bash
set -euo pipefail

# Automated downloader for the Kodak PhotoCD dataset (24 PNG images).
# Usage:
#   ./download_kodak.sh /path/to/data

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/data" >&2
  exit 1
fi

TARGET_ROOT="$1"
TARGET_DIR="${TARGET_ROOT%/}/kodak"
mkdir -p "${TARGET_DIR}"

BASE_URL="https://r0k.us/graphics/kodak/kodak"

for idx in $(seq -w 1 24); do
  filename="kodim${idx}.png"
  url="${BASE_URL}/${filename}"
  destination="${TARGET_DIR}/${filename}"

  if [ -f "${destination}" ]; then
    echo "Skipping existing ${filename}"
    continue
  fi

  echo "Downloading ${filename}"
  curl -L "${url}" -o "${destination}"

done

echo "Kodak dataset ready at ${TARGET_DIR}"
