#!/usr/bin/env bash
# Extract the PHEME rumour scheme dataset tarball into data/raw/.
# Usage: bash scripts/extract_data.sh  (run from repo root)
set -euo pipefail

DEST="data/raw"
TARBALL="$DEST/phemerumourschemedataset.tar.bz2"
EXTRACTED="$DEST/pheme-rumour-scheme-dataset"

mkdir -p "$DEST"

if [ -d "$EXTRACTED" ]; then
  echo "Dataset already extracted at $EXTRACTED — skipping."
  exit 0
fi

echo "Extracting..."
tar -xjf "$TARBALL" -C "$DEST"

echo "Done. Dataset available at $EXTRACTED"
