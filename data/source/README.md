# Source essays (Aeon / Kaggle)

Place **`essays.csv`** here for the mirror pipeline. Expected columns include **`essay`** (full text) and **`title`** (optional; the mirror script does not require it).

## Dataset

[**Aeon Essays** on Kaggle](https://www.kaggle.com/datasets/mannacharya/aeon-essays-dataset) (`mannacharya/aeon-essays-dataset`).

### Option A — Manual download

1. Open the dataset page, accept the rules, and click **Download**.
2. Unzip the archive.
3. Copy `essays.csv` to this directory as `data/source/essays.csv` (from the `promptlab-v2` project root).

### Option B — Kaggle API

1. Create an API token at [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New API Token** → save as `~/.kaggle/kaggle.json`.
2. `pip install kaggle`
3. From `promptlab-v2/`:

```bash
python data/download_aeon_essays.py
```

This downloads and unzips the dataset and ensures `data/source/essays.csv` exists.

## Mirror pipeline (after `essays.csv` is in place)

```bash
export GROQ_API_KEY=gsk_...

# Smoke test (~500 essays → ~17 min at 30 req/min for two LLM calls each... actually 2 calls per essay = 1000 req → ~33 min)
python data/build_mirror_dataset.py \
  --csv data/source/essays.csv \
  --text-col essay \
  --output-jsonl data/mirror/mirrors.jsonl \
  --max-samples 500

# Full run (~2500 essays, ~5000 Groq calls → ~80+ min on free tier)
python data/build_mirror_dataset.py \
  --csv data/source/essays.csv \
  --text-col essay \
  --output-jsonl data/mirror/mirrors.jsonl \
  --max-samples 2500

python data/make_mirror_splits.py \
  --input data/mirror/mirrors.jsonl \
  --out-dir data/mirror

python detector/train.py --config configs/detector_binary.yaml
```
