#!/usr/bin/env bash
# Run T5 rewriter training; patch output_dir onto NFS via a temp YAML (repo configs/*.yaml unchanged).
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NFS_REWRITER_OUT="${NFS_REWRITER_OUT:-/lambda/nfs/outputs/rewriter_${RUN_TAG}}"
mkdir -p "$NFS_REWRITER_OUT"
export WANDB_DIR="${WANDB_DIR:-$NFS_REWRITER_OUT/wandb}"

source "${VENV:-$ROOT/.venv}/bin/activate"
export PYTHONUNBUFFERED=1

SRC_CFG="${REWRITER_CONFIG:-configs/rewriter.yaml}"
export CFG_TMP="$(mktemp "${TMPDIR:-/tmp}/rewriter_lambda_XXXXXX.yaml")"
export NFS_REWRITER_OUT
cleanup() { rm -f "${CFG_TMP:-}"; }
trap cleanup EXIT
cp "$SRC_CFG" "$CFG_TMP"
uv run python -c "
import os
from pathlib import Path
import yaml
p = Path(os.environ['CFG_TMP'])
d = yaml.safe_load(p.read_text(encoding='utf-8'))
d['output_dir'] = os.environ['NFS_REWRITER_OUT']
p.write_text(
    yaml.dump(d, default_flow_style=False, allow_unicode=True, sort_keys=False),
    encoding='utf-8',
)
print('rewriter: patched output_dir ->', d['output_dir'])
"

uv run python rewriter/train.py --config "$CFG_TMP" 2>&1 | tee "$NFS_REWRITER_OUT/train_rewriter.log"
