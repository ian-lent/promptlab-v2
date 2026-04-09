#!/usr/bin/env bash
# Round-0 slop detector training (from repo promptlab-v2 root)
set -euo pipefail
cd "$(dirname "$0")/.."
python data/download_editlens.py --output-dir data/cache/editlens
python data/merge_datasets.py --out-dir data/merged
python detector/train.py --config configs/detector.yaml \
  --train-jsonl data/merged/train.jsonl \
  --val-jsonl data/merged/val.jsonl
python detector/calibrate.py --val-jsonl data/merged/val.jsonl \
  --checkpoint outputs/detector/best \
  --out-json outputs/detector/thresholds.json
