#!/usr/bin/env bash
# One-time Lambda Labs GPU instance setup: venv, deps, env file, optional NFS data sync, tmux.
set -euo pipefail

REPO_URL="${REPO_URL:-}"
WORKDIR="${WORKDIR:-$HOME/promptlab-v2}"
VENV="${VENV:-$WORKDIR/.venv}"
NFS_DATA="${NFS_DATA:-/lambda/nfs/promptlab-data}"
ENV_FILE="${ENV_FILE:-$WORKDIR/.env}"

echo "==> Workdir: $WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ -n "$REPO_URL" ]]; then
  if [[ ! -d .git ]]; then
    git clone "$REPO_URL" .
  else
    git pull || true
  fi
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "==> Loaded environment from $ENV_FILE"
else
  echo "WARN: No $ENV_FILE — create it with GROQ_API_KEY, WANDB_API_KEY (optional), etc."
fi

if [[ -d "$NFS_DATA" ]]; then
  echo "==> Syncing data from $NFS_DATA"
  mkdir -p data outputs
  rsync -a "$NFS_DATA/data/" data/ 2>/dev/null || true
  rsync -a "$NFS_DATA/outputs/" outputs/ 2>/dev/null || true
fi

if [[ ! -d "$VENV" ]]; then
  uv venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv sync --extra tracking 2>/dev/null || uv sync

echo "==> Setup complete. Start training inside tmux, e.g.:"
echo "    tmux new -s train"
echo "    source $VENV/bin/activate && cd $WORKDIR && bash scripts/lambda_train_rewriter.sh"
