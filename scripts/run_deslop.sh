#!/usr/bin/env bash
# Phase 2: one-topic evolutionary deslop (Groq + detector). Example:
#   export GROQ_API_KEY=...  export HF_TOKEN=...  (for Pangram checkpoint)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"
TOPIC="${1:-the ethics of AI in education}"
uv run python deslop/run_topic.py --config configs/deslop.yaml --topic "$TOPIC"
