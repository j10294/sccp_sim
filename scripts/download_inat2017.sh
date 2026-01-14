#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_inat2017.sh /path/to/data_root
#
# Example:
#   bash scripts/download_inat2017.sh ./data

ROOT="${1:-./data}"
YEAR_DIR="${ROOT}/inat2017"
RAW_DIR="${YEAR_DIR}/raw"
EXTRACT_DIR="${YEAR_DIR}/extracted"

mkdir -p "${RAW_DIR}" "${EXTRACT_DIR}"

# Official URLs (iNat 2017)
URL_TRAINVAL_IMAGES="https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz"
URL_TRAINVAL_ANNO="https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val2017.zip"
URL_TEST_IMAGES="https://ml-inat-competition-datasets.s3.amazonaws.com/2017/test2017.tar.gz"

# Known checksum for train_val_images.tar.gz (from torchvision source)
MD5_TRAINVAL_IMAGES="7c784ea5e424efaec655bd392f87301f"

# Downloader: aria2c (best) -> wget -> curl
download() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" ]]; then
    echo "[skip] already exists: $out"
    return 0
  fi

  echo "[down] $url"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -k 1M -c -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$out" "$url"
  else
    # curl
    curl -L --fail --retry 5 --retry-delay 2 -o "$out" "$url"
  fi
}

md5check() {
  local file="$1"
  local expect="$2"
  if command -v md5sum >/dev/null 2>&1; then
    echo "${expect}  ${file}" | md5sum -c -
  elif command -v md5 >/dev/null 2>&1; then
    # macOS
    local got
    got="$(md5 -q "$file")"
    [[ "$got" == "$expect" ]] || { echo "[err] md5 mismatch: $file"; exit 1; }
  else
    echo "[warn] no md5 tool found; skipping checksum verification."
  fi
}

# 1) Train/Val images
TRAINVAL_IMG_TGZ="${RAW_DIR}/train_val_images.tar.gz"
download "${URL_TRAINVAL_IMAGES}" "${TRAINVAL_IMG_TGZ}"
echo "[chk] md5 train_val_images.tar.gz"
md5check "${TRAINVAL_IMG_TGZ}" "${MD5_TRAINVAL_IMAGES}"

# 2) Train/Val annotations
TRAINVAL_ANNO_ZIP="${RAW_DIR}/train_val2017.zip"
download "${URL_TRAINVAL_ANNO}" "${TRAINVAL_ANNO_ZIP}"

# 3) (Optional) Test images
# 주석 해제하면 다운로드/해제까지 수행
TEST_IMG_TGZ="${RAW_DIR}/test2017.tar.gz"
# download "${URL_TEST_IMAGES}" "${TEST_IMG_TGZ}"

echo "[ok] download done."

# ---------------------------
# Extraction
# ---------------------------
echo "[ext] extracting train_val_images.tar.gz (this will take a while)"
mkdir -p "${EXTRACT_DIR}/train_val_images"
tar -xzf "${TRAINVAL_IMG_TGZ}" -C "${EXTRACT_DIR}/train_val_images"

echo "[ext] extracting train_val2017.zip"
mkdir -p "${EXTRACT_DIR}/train_val2017"
unzip -q "${TRAINVAL_ANNO_ZIP}" -d "${EXTRACT_DIR}/train_val2017"

# Optional test extraction
# echo "[ext] extracting test2017.tar.gz"
# mkdir -p "${EXTRACT_DIR}/test2017"
# tar -xzf "${TEST_IMG_TGZ}" -C "${EXTRACT_DIR}/test2017"

echo
echo "[done] iNat2017 is ready at:"
echo "  ${EXTRACT_DIR}/train_val_images"
echo "  ${EXTRACT_DIR}/train_val2017"
