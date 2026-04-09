#!/usr/bin/env bash
# Binary RoBERTa detector from mirror JSONL (no Pangram checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."

# 0) Aeon Kaggle CSV at data/source/essays.csv (see data/source/README.md)
#
# 1) Build mirrors (smoke: --max-samples 500; full: 2500)
# export GROQ_API_KEY=...
# python data/build_mirror_dataset.py --csv data/source/essays.csv --text-col essay \
#   --output-jsonl data/mirror/mirrors.jsonl --max-samples 2500

# 2) Stratified splits
python data/make_mirror_splits.py --input data/mirror/mirrors.jsonl --out-dir data/mirror

# 3) Train
python detector/train.py --config configs/detector_binary.yaml

# 4) Calibrate on val set
python detector/calibrate.py --checkpoint outputs/detector_binary/best \
  --val-jsonl data/mirror/val.jsonl --out-json outputs/detector_binary/thresholds.json
