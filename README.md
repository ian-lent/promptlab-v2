# PromptLab v2 — Deslop via adversarial co-training

This package implements a **prompt-level** pipeline to reduce AI “slop” scores from an EditLens-style detector: an evolutionary **deslop optimizer** competes with a **retrainable slop detector**, and a **prompt rewriter** is distilled from logged prompt pairs.

## Layout

- `configs/` — YAML for detector, deslop, co-training, rewriter, and eval
- `data/` — HuggingFace downloads, mirror dataset CLI, dataset merge
- `detector/` — `SlopDetector`, training, inference, calibration
- `deslop/` — prompt bank, LLM mutator, evolutionary optimizer, constraints
- `cotrain/` — adversarial loop, data accumulation, pair logging, stopping
- `rewriter/` — pair prep, QLoRA fine-tuning, inference, optional RL refine stub
- `eval/` — metrics, baselines, end-to-end eval script
- `notebooks/` — exploratory and demo notebooks (stubs; fill in as you run experiments)
- `scripts/` — shell entrypoints

## Environment

```bash
cd promptlab-v2
uv sync   # or: pip install -e .

export HF_TOKEN=hf_...   # only if using gated Pangram models/datasets (not needed for binary mirror detector)
export GROQ_API_KEY=gsk_...
export GOOGLE_API_KEY=...   # optional; Gemini
```

Pangram **EditLens** data and checkpoints are **CC BY-NC-SA 4.0** (noncommercial). Accept the dataset license on Hugging Face before downloading.

## Aeon essays → mirror detector (Kaggle source)

1. Put **`data/source/essays.csv`** in place (see **`data/source/README.md`**: manual download or `python data/download_aeon_essays.py` with Kaggle API credentials).
2. Run the mirror + split + train commands in that README (`GROQ_API_KEY` required). Use **`--max-samples 500`** first (~smoke test), then **`2500`** overnight (~80+ min wall time at ~30 req/min).

## Quick start — binary detector (no Pangram access)

Uses **`FacebookAI/roberta-large`** with a **new 2-class head** (human=0, AI=1). `SlopDetector.score()` is still a weighted average over buckets; with K=2 it equals **P(AI)**.

1. **Groq mirror data** (aim for ≥2000 successful pairs → ≥4000 JSONL rows):  
   `python data/build_mirror_dataset.py --csv /path/to/essays.csv --text-col essay --output-jsonl data/mirror/mirrors.jsonl --max-samples 2500`
2. **Splits**: `python data/make_mirror_splits.py --input data/mirror/mirrors.jsonl --out-dir data/mirror`
3. **Train**: `python detector/train.py --config configs/detector_binary.yaml`
4. **Load at inference**: `SlopDetector(checkpoint="outputs/detector_binary/best")` (local folder; no HF token).

Shell recipe: `scripts/run_training_binary.sh` (uncomment the mirror step and set your CSV).

## Quick start — EditLens checkpoint (Round 0, gated)

1. Download data: `python data/download_editlens.py --output-dir data/cache`
2. (Optional) Build mirrors: `python data/build_mirror_dataset.py --help`
3. Merge: `python data/merge_datasets.py --config configs/detector.yaml`
4. Train: `python detector/train.py --config configs/detector.yaml --extra-data path/to/extra.jsonl`

Run Python with `promptlab-v2` as the working directory so imports resolve (`detector`, `deslop`, …), or install the package in editable mode.

## Phase 3 — co-training (adversarial loop)

Each round: sample topics → evolutionary deslop against the current detector → append “fool” essays → retrain detector → next round.

```bash
export GROQ_API_KEY=...
export HF_TOKEN=...
cd promptlab-v2
uv run python cotrain/loop.py --smoke
```

### Remote GPU retrain pattern (recommended on Mac/MPS)

If `roberta-large` retraining OOMs on Apple MPS, run deslop locally but retrain the detector on a GPU box:

1) **Local**: generate adversarial essays + merged train JSONL for the round:

```bash
cd promptlab-v2
export GROQ_API_KEY=...
export HF_TOKEN=...
uv run python cotrain/loop.py --skip-detector-retrain --resume-extra --smoke
# Look for: {"round": 1, "train_jsonl_for_retrain": "outputs/detector_cotrain/train_merged_r1.jsonl"}
```

2) **Remote (Colab / GPU VM)**: run detector training using that merged JSONL:

```bash
python detector/train.py --config configs/detector.yaml \
  --train-jsonl outputs/detector_cotrain/train_merged_r1.jsonl \
  --val-jsonl data/merged/val.jsonl \
  --output-dir outputs/detector_cotrain/round_1
```

3) **Back local**: download `outputs/detector_cotrain/round_1/best/` and use it as the next checkpoint:

```bash
uv run python cotrain/loop.py --skip-detector-retrain --resume-extra --smoke \
  --checkpoint outputs/detector_cotrain/round_1/best
```

## Parent repository

The older **slop-minimization** code and `slop_configs/` lexicons live in the repo root; `eval/metrics.py` can load lexicon paths relative to the parent project when configured.

## Implementation notes

- **EditLens HF schema**: If `data/download_editlens.py` maps scores wrong, inspect `datasets.load_dataset("pangram/editlens_iclr")` feature names and adjust `--score-field` / `configs/detector.yaml`.
- **Detector training**: `configs/detector_binary.yaml` + `fresh_classification_head: true` trains **K=2** on mirror JSONL. `configs/detector.yaml` fine-tunes the gated **EditLens** head (K=11) when you have access. Same `SlopDetector` API for both; increase `num_buckets` when you add EditLens-style continuous labels.
- **trl / SFT**: `rewriter/train.py` targets **trl** `SFTConfig` (`max_seq_length`, `dataset_text_field`). If your `trl` version errors on args, align with its docs or pin `trl>=0.9,<0.11`.
- **Co-training**: `cotrain/loop.py` shells out to `detector/train.py` with a merged JSONL each round; ensure merged `val.jsonl` exists before long runs.
