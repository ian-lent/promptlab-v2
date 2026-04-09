#!/usr/bin/env python3
"""Merge EditLens JSONL + mirror JSONL into stratified train/val/test splits."""

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
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def bucketize(score: float, n_buckets: int = 11) -> int:
    import math

    b = int(math.floor(float(score) * n_buckets))
    return max(0, min(n_buckets - 1, b))


def _editlens_val_path(train_path: Path, explicit: Path | None) -> Path:
    """HF `pangram/editlens_iclr` uses split name `val` → `val.jsonl`; some caches use `validation.jsonl`."""
    if explicit is not None:
        return explicit
    d = train_path.parent
    for name in ("validation.jsonl", "val.jsonl"):
        p = d / name
        if p.is_file():
            return p
    return d / "validation.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--editlens-train", type=Path, default=Path("data/cache/editlens/train.jsonl"))
    ap.add_argument(
        "--editlens-val",
        type=Path,
        default=None,
        help="Defaults to validation.jsonl or val.jsonl next to --editlens-train (whichever exists)",
    )
    ap.add_argument("--editlens-test", type=Path, default=Path("data/cache/editlens/test.jsonl"))
    ap.add_argument("--mirror-jsonl", type=Path, default=None, help="Optional mirror pairs JSONL")
    ap.add_argument("--out-dir", type=Path, default=Path("data/merged"))
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-buckets", type=int, default=11)
    args = ap.parse_args()

    val_path = _editlens_val_path(args.editlens_train, args.editlens_val)

    records: list[dict] = []
    for p in (args.editlens_train, val_path, args.editlens_test):
        if p.exists():
            records.extend(load_jsonl(p))
        else:
            print(f"Skip missing {p}")

    if args.mirror_jsonl and args.mirror_jsonl.exists():
        records.extend(load_jsonl(args.mirror_jsonl))

    if not records:
        raise SystemExit("No records loaded. Run data/download_editlens.py first.")

    df = pd.DataFrame(records)
    for col in ("text", "score", "domain", "source"):
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in merged data")
    df["bucket"] = df["score"].apply(lambda s: bucketize(float(s), args.n_buckets))

    strat = df["domain"].astype(str) + "_" + df["bucket"].astype(str)
    idx = df.index.values
    temp_size = args.val_ratio + args.test_ratio
    try:
        train_idx, temp_idx = train_test_split(
            idx, test_size=temp_size, random_state=args.seed, stratify=strat
        )
        strat_temp = strat.loc[temp_idx]
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
        sub = df.loc[indices].copy()
        sub["split"] = name
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for _, row in sub.iterrows():
                rec = {
                    "text": row["text"],
                    "score": float(row["score"]),
                    "domain": row["domain"],
                    "source": row["source"],
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
