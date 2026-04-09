#!/usr/bin/env python3
"""
Split mirror JSONL (human/AI rows with score 0/1) into train/val/test.

Use after build_mirror_dataset.py. Target ≥2000 successful mirror pairs (≈4000 rows)
before training the binary detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Mirror JSONL from build_mirror_dataset.py")
    ap.add_argument("--out-dir", type=Path, default=Path("data/mirror"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    df = pd.DataFrame(rows)
    if "text" not in df.columns or "score" not in df.columns:
        raise SystemExit("Each row must have 'text' and 'score' (0=human, 1=AI)")
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20]
    df["label"] = (df["score"].astype(float) >= 0.5).astype(int)

    n_ai = int(df["label"].sum())
    n_human = int(len(df) - n_ai)
    n_pairs = min(n_human, n_ai)
    print(f"Rows: {len(df)}  (human={n_human}, AI={n_ai}, balanced_pairs≈{n_pairs})")
    if n_pairs < 2000:
        print(
            "Warning: aim for ≥2000 mirror pairs (one human + one AI per source essay → "
            "≥4000 rows). Use more source essays or run build_mirror_dataset with a larger CSV."
        )

    idx = df.index.to_numpy()
    strat = df["label"].to_numpy()
    temp_size = args.val_ratio + args.test_ratio
    try:
        train_idx, temp_idx = train_test_split(
            idx, test_size=temp_size, random_state=args.seed, stratify=strat
        )
        strat_temp = df.loc[temp_idx, "label"]
        rel_test = args.test_ratio / temp_size if temp_size > 0 else 0.5
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=rel_test, random_state=args.seed, stratify=strat_temp
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(idx, test_size=temp_size, random_state=args.seed)
        rel_test = args.test_ratio / temp_size if temp_size > 0 else 0.5
        val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test, random_state=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_split(name: str, indices) -> None:
        sub = df.loc[indices]
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for _, row in sub.iterrows():
                rec = {
                    "text": row["text"],
                    "score": float(row["score"]),
                    "domain": str(row.get("domain", "mirror_essay")),
                    "source": str(row.get("source", "mirror")),
                    "split": name,
                    "round": int(row.get("round", 0)),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {path} ({len(sub)} rows)")

    write_split("train", train_idx)
    write_split("val", val_idx)
    write_split("test", test_idx)


if __name__ == "__main__":
    main()
