#!/usr/bin/env bash
# Download and extract the PHEME rumour scheme dataset into data/raw/.
# Usage: bash scripts/download_data.sh  (run from repo root)
set -euo pipefail

DEST="data/raw"
TARBALL="$DEST/phemerumourschemedataset.tar.bz2"
EXTRACTED="$DEST/pheme-rumour-scheme-dataset"

mkdir -p "$DEST"

if [ -d "$EXTRACTED" ]; then
  echo "Dataset already extracted at $EXTRACTED — skipping download."
  exit 0
fi

echo "Downloading PHEME rumour scheme dataset..."
curl -L "https://figshare.com/ndownloader/files/4988998" -o "$TARBALL"

echo "Extracting..."
tar -xjf "$TARBALL" -C "$DEST"

echo "Done. Dataset available at $EXTRACTED"
