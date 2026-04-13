#!/usr/bin/env bash
# Run drift-coefficient optimization; patch trajectory_jsonl + optimized_output_path onto NFS (temp YAML).
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NFS_DRIFT_OUT="${NFS_DRIFT_OUT:-/lambda/nfs/outputs/drift_coef_opt_${RUN_TAG}}"
mkdir -p "$NFS_DRIFT_OUT"
export WANDB_DIR="${WANDB_DIR:-$NFS_DRIFT_OUT/wandb}"

source "${VENV:-$ROOT/.venv}/bin/activate"
export PYTHONUNBUFFERED=1

SRC_CFG="${DESLOP_CONFIG:-configs/deslop.yaml}"
export CFG_TMP="$(mktemp "${TMPDIR:-/tmp}/deslop_lambda_XXXXXX.yaml")"
export NFS_DRIFT_OUT
cleanup() { rm -f "${CFG_TMP:-}"; }
trap cleanup EXIT
cp "$SRC_CFG" "$CFG_TMP"
uv run python -c "
import os
from pathlib import Path
import yaml
p = Path(os.environ['CFG_TMP'])
d = yaml.safe_load(p.read_text(encoding='utf-8'))
sec = d.setdefault('drift_coef_opt', {})
sec['trajectory_jsonl'] = str(Path(os.environ['NFS_DRIFT_OUT']) / 'drift_coef_opt_trajectory.jsonl')
sec['optimized_output_path'] = str(Path(os.environ['NFS_DRIFT_OUT']) / 'optimized_drift_coefs.json')
p.write_text(
    yaml.dump(d, default_flow_style=False, allow_unicode=True, sort_keys=False),
    encoding='utf-8',
)
print('drift_coef_opt: patched trajectory_jsonl ->', sec['trajectory_jsonl'])
print('drift_coef_opt: patched optimized_output_path ->', sec['optimized_output_path'])
"

uv run python deslop/drift_coef_opt.py --config "$CFG_TMP" 2>&1 | tee "$NFS_DRIFT_OUT/train_coefs.log"
