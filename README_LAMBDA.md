# Lambda Labs training workflow

This project trains the T5 rewriter (`rewriter/train.py`) and drift-coefficient search (`deslop/drift_coef_opt.py`) on [Lambda Labs](https://cloud.lambda.ai) GPUs.

**Persistent outputs:** `lambda_train_rewriter.sh` and `lambda_train_coefs.sh` copy your YAML to a temp file, patch paths onto NFS, then delete the temp file—your checked-in `configs/*.yaml` is never modified. By default each run uses `RUN_TAG` (default: timestamp) so outputs go to `/lambda/nfs/outputs/rewriter_${RUN_TAG}` or `/lambda/nfs/outputs/drift_coef_opt_${RUN_TAG}`. Override with `NFS_REWRITER_OUT` / `NFS_DRIFT_OUT` if you prefer fixed paths.

## 1. Launch an instance

In the Lambda console, pick a GPU that fits `t5-base` + LoRA (e.g. A10 or A100). Note the instance IP and SSH command.

## 2. Attach persistent storage

Mount or attach your Lambda **persistent volume** (often exposed under `/lambda/nfs/`). Create subfolders you will use for data and outputs, for example:

- `/lambda/nfs/promptlab-data/data`
- `/lambda/nfs/promptlab-data/outputs`
- `/lambda/nfs/outputs/rewriter`
- `/lambda/nfs/outputs/drift_coef_opt`

## 3. SSH setup

From your laptop:

```bash
ssh <user>@<instance-ip>
```

Copy your `.env` (at minimum `GROQ_API_KEY`; optional `WANDB_API_KEY`) to the instance, e.g. `~/promptlab-v2/.env`.

## 4. Run setup

On the instance, set `REPO_URL` to your git remote (or omit if you already cloned):

```bash
export REPO_URL="https://github.com/your-org/promptlab-v2.git"
export WORKDIR="$HOME/promptlab-v2"
bash scripts/lambda_setup.sh
```

Adjust `NFS_DATA` in the script or export it before running if your data lives at a different NFS path.

## 5. Launch training (tmux)

Detach-safe session:

```bash
tmux new -s train
source "$HOME/promptlab-v2/.venv/bin/activate"
cd "$HOME/promptlab-v2"
bash scripts/lambda_train_rewriter.sh
# or: bash scripts/lambda_train_coefs.sh
```

Detach with `Ctrl-b` then `d`.

### Smoke test (before a full rewriter run)

End-to-end check (~1–2 minutes) with short training and frequent eval so `RewriterSlopCallback` runs (Groq + detector):

```bash
cd "$HOME/promptlab-v2"
source .venv/bin/activate
uv run python rewriter/train.py --config configs/rewriter.yaml \
  --override max_steps=10 --override eval_steps=5 --override lora_r=4
```

You should see a JSON line logging `rewriter_slop_callback` with `eval_strategy` / `eval_steps` on the first evaluation.

### LoRA rank ablation (r = 4, 8, 16)

There is no multi-rank orchestrator: run **three separate jobs** (e.g. three `tmux` windows or sequential runs). Before each job, set rank in YAML **or** pass `--override lora_r=...`, and give each run a distinct tag so checkpoints do not collide:

```bash
# Window 1
RUN_TAG=r4 bash scripts/lambda_train_rewriter.sh   # plus --override lora_r=4 if not in YAML

# Window 2
RUN_TAG=r8 bash scripts/lambda_train_rewriter.sh

# Window 3
RUN_TAG=r16 bash scripts/lambda_train_rewriter.sh  # plus --override lora_r=16
```

Use `REWRITER_CONFIG` if you maintain one YAML per rank.

## 6. Monitor

- `tmux attach -t train` to reconnect.
- Tail logs: `tail -f /lambda/nfs/outputs/rewriter_<RUN_TAG>/train_rewriter.log` (path matches your `RUN_TAG`; drift: `drift_coef_opt_<RUN_TAG>/train_coefs.log`).
- If `WANDB_API_KEY` is set and `uv sync --extra tracking` was run, open the W&B run in the browser.

## 7. Copy outputs back

From your laptop (example):

```bash
rsync -avz <user>@<instance-ip>:/lambda/nfs/outputs/ ./lambda_outputs/
```

## Drift features and co-training

- Evolutionary logs only contain drift components when `alignment_reference_mode` is non-null (e.g. `topic`). Otherwise the feature pool for `drift_coef_opt` stays empty until you run `deslop/export_pair_drift_features.py` on held-out pairs.
- For strict holdout alignment with the rewriter, build `outputs/cotrain/splits/rewriter_split_manifest.json` first (via `rewriter/train.py` or `rewriter.dataset.ensure_split_manifest`), export **test** split features, then point `drift_coef_opt.features_jsonl` at that file.

**Reporting note:** Appending essay rows from deslop logs into a shared feature pool (no `pair_row_id`) is the practical default for plotting a smooth Adam trajectory; `export_pair_drift_features.py` is the stricter, production-style holdout path. Document both in the report (method + limitations).

## Small data / overfitting

With only a few co-training rounds, validation loss may **rise** while training loss falls—especially at **LoRA r=16**. That is a valid report observation; mitigate with more `cotrain` rounds, lower rank, or higher `weight_decay`.
