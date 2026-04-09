#!/usr/bin/env bash
# Build a Colab-friendly zip: code + configs + merged/mirror data, no venv or HF caches.
# Run from inside promptlab-v2:  ./scripts/package_for_colab.sh
# Or:  bash /path/to/promptlab-v2/scripts/package_for_colab.sh /path/to/promptlab-v2
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${ROOT}/../promptlab-v2-colab.zip"
NAME="$(basename "$ROOT")"
PARENT="$(dirname "$ROOT")"

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "Not a project root (missing pyproject.toml): $ROOT" >&2
  exit 1
fi

cd "$PARENT"
echo "Zipping $ROOT -> $OUT"
rm -f "$OUT"
zip -r -q "$OUT" "$NAME" \
  -x "${NAME}/.venv/*" \
  -x "${NAME}/**/.venv/*" \
  -x "${NAME}/*/__pycache__/*" \
  -x "${NAME}/*/*/__pycache__/*" \
  -x "${NAME}/*/*/*/__pycache__/*" \
  -x "${NAME}/*.pyc" \
  -x "${NAME}/**/*.pyc" \
  -x "${NAME}/.git/*" \
  -x "${NAME}/**/.git/*" \
  -x "${NAME}/data/cache/*" \
  -x "${NAME}/outputs/*" \
  -x "${NAME}/*.zip" \
  -x "${NAME}/**/.DS_Store" \
  -x "${NAME}/.pytest_cache/*"

ls -lh "$OUT"
echo "Upload $OUT to Colab (Files sidebar) or Google Drive, then unzip under /content."
