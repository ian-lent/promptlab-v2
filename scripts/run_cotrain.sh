#!/usr/bin/env bash
# Phase 3: adversarial co-training. Needs GROQ_API_KEY, HF_TOKEN (Pangram), and merged train/val JSONL.
# Cheap check: ./scripts/run_cotrain.sh --smoke
# Deslop-only (no detector retrain): ./scripts/run_cotrain.sh --skip-detector-retrain --smoke
# Iterative remote retrain: add --resume-extra and run 1 round at a time, e.g.
#   ./scripts/run_cotrain.sh --skip-detector-retrain --resume-extra --config configs/cotrain.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"
uv run python cotrain/loop.py "$@"
