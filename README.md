markdown# PromptLab v2 — AI Slop Reduction via Adversarial Co-Training

[![Open Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ian-lent/promptlab-v2/blob/main/notebooks/promptlab_v2_demo.ipynb)

STAT 4830 · University of Pennsylvania

A two-stage pipeline that reduces the AI-detectability of generated essays. An evolutionary optimizer (deslop) searches the prompt space guided by a fixed classifier, accumulating contrastive essay pairs. A T5-base model fine-tuned with LoRA is then trained to perform essay-to-essay rewriting in a single forward pass.

**Final result:** 0.0998 mean slop on 20 held-out val pairs — a 74% reduction from the 0.391 prompt-mode baseline.

---

## How it works

### Stage 1 — deslop optimizer

The optimizer maintains a population of prompt candidates and applies mutation and crossover operations guided by `llama-3.1-8b-instant`. Essays are generated via `llama-3.3-70b-versatile` through the Groq API and scored by a fixed detector (`editlens/roberta-large`, "SlopDetector"). The scoring function includes a drift penalty to prevent the optimizer from gaming the detector with incoherent text:
optimization_slop = detector_slop
+ α·(1 − cosine_similarity)
+ β·(1 − ROUGE-L)
+ γ·BERTScore_penalty

**Few-shot injection** makes the search self-improving: winning low-slop essays from prior rounds are prepended as examples to future generation calls, so the optimizer accumulates stylistic knowledge rather than rediscovering it each round.

Output: **372 co-training pairs** (40 contrastive, gap > 0.1) across **77 unique topics**.

### Stage 2 — T5+LoRA rewriter

T5-base + LoRA (r=16, ~800K trainable parameters) is trained on the contrastive pairs to rewrite high-slop essays directly to low-slop essays in a single forward pass. Key training decisions:

- **AdamW** with `weight_decay=0.01`
- **Cosine LR decay** from `LR=1e-4` with 50 warmup steps
- **Curriculum ordering**: training pairs sorted by contrastive gap descending (hardest signal first)
- **Early stopping on `slop_mean`**, not `eval_loss` — the two diverge after step 50 as the model memorizes
- **Deterministic beam search** (`num_beams=4`, `do_sample=False`) in the eval callback for stable checkpoint comparisons

---

## Results

| Run | Architecture | Train Pairs | Val Pairs | Best Slop | Best Step | Notes |
|-----|-------------|-------------|-----------|-----------|-----------|-------|
| Organic baseline | Prompt→Prompt | ~34 | 4 | 0.391 | ~150 | Starting point |
| + 2272 Alpaca | Prompt→Prompt | 2306 | 14 | 0.580 | 100 | ❌ Wrong distribution |
| Organic long run | Prompt→Prompt | ~34 | 14 | 0.759 | 1300 | ❌ Overfitting |
| Essay pivot, 11 pairs | Essay→Essay | 11 | 6 | 0.189 | 50 | ✓ Architecture pivot −52% |
| Essay, 21 pairs | Essay→Essay | 21 | 12 | 0.247 | 50 | Harder val set |
| LR=3e-4, r=16 | Essay→Essay | 40 | 20 | 0.1958 | 50 | LR sweep |
| **LR=1e-4, r=16 ★** | **Essay→Essay** | **40** | **20** | **0.0998** | **50** | **BEST** |
| LR=1e-4, r=8 (ablation) | Essay→Essay | 40 | 20 | 0.1058 | 50 | r=16 confirmed (+0.006 only) |

---

## Repo layout
```
promptlab-v2/
├── configs/
│   └── topics_alpaca_diverse.yaml   # 100-topic set used for final run
├── cotrain/
│   └── loop.py                      # co-training loop; saves best_essay alongside each pair
├── deslop/                          # evolutionary optimizer
├── detector/
│   └── model.py                     # SlopDetector (editlens/roberta-large)
├── rewriter/
│   ├── train.py                     # T5+LoRA training (essay mode, Seq2SeqTrainer)
│   ├── essay_dataset.py             # curriculum dataset with gap-descending sort
│   └── generate_essay_pairs.py      # generates contrastive pairs from cotrain output
├── notebooks/
│   ├── promptlab_v2_demo.ipynb      # 3-adapter comparison demo (v1/v2/v3)
│   ├── cotrain_colab.ipynb          # co-training run notebook
│   └── cotrain_report.ipynb         # co-training results and analysis
├── figures/
│   ├── fig1_optimization_formula.svg
│   ├── fig2_stage1_pipeline.svg
│   ├── fig3_few_shot_injection.svg
│   ├── fig4_unified_pipeline.svg
│   └── promptlab_v2_diagrams.html   # all diagrams in one viewer
├── docs/
│   └── promptlab_v2_report_final.md # full experimental report
└── outputs/
    └── cotrain/
        └── prompt_pairs.jsonl       # 372 accumulated co-training pairs
```

---

## Setup

```bash
cd promptlab-v2
pip install -e .

export GROQ_API_KEY=gsk_...
export HF_TOKEN=hf_...   # only needed for gated Pangram models
```

---

## Running the optimizer

```bash
python cotrain/loop.py --smoke          # quick smoke test (5 topics, 3 rounds)
python cotrain/loop.py                  # full run across all topics
```

Pairs are saved to `outputs/cotrain/prompt_pairs.jsonl` with `best_essay` embedded for fast rewriter training.

---

## Training the rewriter

```bash
# Generate contrastive pairs from accumulated cotrain output
python rewriter/generate_essay_pairs.py

# Train T5+LoRA rewriter (essay mode)
python rewriter/train.py \
  --lr 1e-4 \
  --lora_r 16 \
  --mode essay
```

Best adapter saved to `outputs/rewriter/lora_adapter/` and synced to Google Drive.

---

## Demo

Open [`notebooks/promptlab_v2_demo.ipynb`](notebooks/promptlab_v2_demo.ipynb) in Colab. The notebook loads all three adapter generations (v1 prompt-mode, v2 essay pivot, v3 final) and runs a side-by-side comparison on any topic you choose.

Adapter paths are hardcoded to Google Drive locations from the final training runs:

| Adapter | Drive path | Val slop |
|---------|-----------|----------|
| v1: prompt-mode | `promptlab-v2-outputs/2026-04-14_17-48/...` | ~0.391 |
| v2: essay pivot | `promptlab-v2-outputs/2026-04-15_04-40_.../` | ~0.189 |
| v3: final ★ | `promptlab-v2-outputs/2026-04-19_16-47_FINAL_MODEL/` | **0.0998** |

---

## Key findings

1. **Data quality > quantity** — 2272 Alpaca pairs degraded performance; wrong distribution produces contradictory gradients
2. **Essay→Essay beats Prompt→Prompt by 52%** — one less step of indirection from the detector signal
3. **Best checkpoint always at step 50** — the NQM noise floor with 40 pairs/800K params; early stopping on `slop_mean` is essential
4. **LR=1e-4 beats 3e-4 by 2×** — conservative LR avoids overshooting in the ~50 useful steps
5. **r=16 vs r=8 gap is only 0.006** — both well-regularized; r=16 confirmed appropriate

---

## Notes

- EditLens (`pangram/editlens_roberta-large`) is **CC BY-NC-SA 4.0** — noncommercial use only
- Groq API is used for all LLM calls; no local GPU required for the optimizer stage
- The rewriter requires a CUDA GPU for training (tested on A100/T4 via Colab)
- T5-base compresses rather than rewrites at length due to summarization pretraining — a larger generative backbone would improve output fluency while preserving the slop reduction
