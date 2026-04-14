#!/usr/bin/env python3
"""
Sample Alpaca-style instructions (empty ``input`` only), cluster with MiniLM + k-means, and
derive essay topics for co-training.

**Dataset note:** The Hub id ``tatsu-lab/alpaca-cleaned`` is not published. This script defaults
to ``yahma/alpaca-cleaned`` (standard cleaned Alpaca). Use ``--dataset tatsu-lab/alpaca`` for the
original Stanford release.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

MAPPING_RULES_DOC = """
Alpaca instruction → essay topic (heuristic rules)
-------------------------------------------------
  Explain X        →  the nature and implications of X
  Describe X       →  X and its significance
  Compare X and Y  →  the relative merits of X versus Y
                     (also: Compare X to Y / Compare X with Y)
  Should X         →  whether X
  What is X        →  X and what it means
  What are X       →  X and what it means
  (otherwise)      →  use the instruction text unchanged
"""


def alpaca_instruction_to_essay_topic(instruction: str) -> str:
    """Apply the mapping rules above (case-insensitive, leading pattern only)."""
    t = instruction.strip()
    if not t:
        return t

    m = re.match(r"^Explain\s+(.+)$", t, re.IGNORECASE | re.DOTALL)
    if m:
        return f"the nature and implications of {m.group(1).strip()}"

    m = re.match(r"^Describe\s+(.+)$", t, re.IGNORECASE | re.DOTALL)
    if m:
        return f"{m.group(1).strip()} and its significance"

    m = re.match(r"^Compare\s+(.+?)\s+(?:and|to|with)\s+(.+)$", t, re.IGNORECASE | re.DOTALL)
    if m:
        left, right = m.group(1).strip(), m.group(2).strip()
        return f"the relative merits of {left} versus {right}"

    m = re.match(r"^Should\s+(.+)$", t, re.IGNORECASE | re.DOTALL)
    if m:
        return f"whether {m.group(1).strip()}"

    m = re.match(r"^What\s+is\s+(.+)$", t, re.IGNORECASE | re.DOTALL)
    if m:
        return f"{m.group(1).strip()} and what it means"

    m = re.match(r"^What\s+are\s+(.+)$", t, re.IGNORECASE | re.DOTALL)
    if m:
        return f"{m.group(1).strip()} and what it means"

    return t


def _empty_input(example: dict) -> bool:
    inp = example.get("input")
    return not str(inp or "").strip()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample Alpaca rows, k-means diverse instructions, map to essay topics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MAPPING_RULES_DOC,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="yahma/alpaca-cleaned",
        help="HF dataset id (default: yahma/alpaca-cleaned; tatsu-lab/alpaca-cleaned is not on Hub).",
    )
    p.add_argument("--sample-size", type=int, default=500, help="Random instructions after filter.")
    p.add_argument("--k", type=int, default=25, help="k-means clusters.")
    p.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("configs/topics_alpaca_diverse.yaml"),
        help="YAML path written after review acknowledgment.",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Write --output after acknowledgment (see --i-acknowledge-manual-review).",
    )
    p.add_argument(
        "--i-acknowledge-manual-review",
        action="store_true",
        help="Non-interactive: confirm you reviewed printed topics (required with --save).",
    )
    args = p.parse_args()

    print(MAPPING_RULES_DOC.strip(), flush=True)
    print(flush=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset {args.dataset!r} ...", flush=True)
    try:
        ds = load_dataset(args.dataset, split="train")
    except Exception as e:
        if "tatsu-lab/alpaca-cleaned" in args.dataset:
            print(
                "Hint: tatsu-lab/alpaca-cleaned is not on the Hub. "
                "Use --dataset yahma/alpaca-cleaned or tatsu-lab/alpaca.",
                file=sys.stderr,
            )
        raise e
    ds_empty = ds.filter(_empty_input)
    n_avail = len(ds_empty)
    if n_avail < args.sample_size:
        print(
            f"Only {n_avail} rows with empty input; using all of them (requested {args.sample_size}).",
            flush=True,
        )
        n_take = n_avail
    else:
        n_take = args.sample_size

    indices = rng.sample(range(n_avail), n_take)
    subset = ds_empty.select(indices)
    instructions = [str(subset[i]["instruction"]).strip() for i in range(len(subset))]

    print(f"Embedding {len(instructions)} instructions with {args.embedding_model!r} ...", flush=True)
    model = SentenceTransformer(args.embedding_model)
    emb = model.encode(instructions, show_progress_bar=True, convert_to_numpy=True)

    print(f"K-means (k={args.k}, seed={args.seed}) ...", flush=True)
    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    labels = kmeans.fit_predict(emb)
    centers = kmeans.cluster_centers_

    print(
        "\n=== Nearest assigned point per cluster (L2 distance to that cluster centroid) ===\n",
        flush=True,
    )
    representatives: list[dict] = []
    for k in range(args.k):
        mask = labels == k
        if np.any(mask):
            member_idx = np.where(mask)[0]
            dists = np.linalg.norm(emb[member_idx] - centers[k], axis=1)
            rel = int(np.argmin(dists))
            idx = int(member_idx[rel])
            dist = float(dists[rel])
        else:
            dists = np.linalg.norm(emb - centers[k], axis=1)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
        instr = instructions[idx]
        converted = alpaca_instruction_to_essay_topic(instr)
        representatives.append(
            {
                "cluster": k,
                "nearest_instruction": instr,
                "distance_to_centroid": dist,
                "essay_topic": converted,
            }
        )
        print(f"cluster={k:2d}  distance={dist:.4f}", flush=True)
        print(f"  instruction: {instr}", flush=True)
        print(f"  essay_topic: {converted}", flush=True)
        print(flush=True)

    print("=== Review pairs (original → converted) ===\n", flush=True)
    for r in representatives:
        print(f"[{r['cluster']:2d}] {r['nearest_instruction']!r}", flush=True)
        print(f"     → {r['essay_topic']!r}", flush=True)
        print(flush=True)

    if args.save:
        if not args.i_acknowledge_manual_review:
            try:
                resp = input(
                    "Manual review complete? Type YES (all caps) to write "
                    f"{args.output}: "
                ).strip()
            except EOFError:
                print("No TTY; use --i-acknowledge-manual-review with --save.", file=sys.stderr)
                sys.exit(1)
            if resp != "YES":
                print("Not saving (expected exact YES).", flush=True)
                return
        else:
            print(
                f"Saving to {args.output} (--i-acknowledge-manual-review).",
                flush=True,
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "description": (
                "Essay topics from Alpaca-style instructions (empty input only). "
                "One representative per k-means cluster (diverse coverage)."
            ),
            "dataset": args.dataset,
            "sample_size": len(instructions),
            "k_means_k": args.k,
            "embedding_model": args.embedding_model,
            "seed": args.seed,
            "mapping_rules": {
                "Explain X": "the nature and implications of X",
                "Describe X": "X and its significance",
                "Compare X and Y (or to/with)": "the relative merits of X versus Y",
                "Should X": "whether X",
                "What is X / What are X": "X and what it means",
                "default": "instruction text unchanged",
            },
            "topics": [r["essay_topic"] for r in representatives],
            "provenance": [
                {
                    "cluster": r["cluster"],
                    "nearest_instruction": r["nearest_instruction"],
                    "distance_to_centroid": r["distance_to_centroid"],
                    "essay_topic": r["essay_topic"],
                }
                for r in representatives
            ],
        }
        args.output.write_text(
            yaml.dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(json.dumps({"saved": str(args.output.resolve()), "n_topics": len(doc["topics"])}), flush=True)
    else:
        print(
            "Not writing YAML (pass --save, then type YES or use --i-acknowledge-manual-review).",
            flush=True,
        )


if __name__ == "__main__":
    main()
