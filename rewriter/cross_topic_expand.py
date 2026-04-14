#!/usr/bin/env python3
"""
Cross-topic expansion: reuse learned cluster templates on new topics.

Reads:
  - outputs/cotrain/prompt_pairs.jsonl (organic)
  - outputs/rewriter/clusters.json
  - configs/topics_alpaca_diverse.yaml (topic pool)

Writes:
  - outputs/cotrain/prompt_pairs_cross_topic.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from rewriter.dataset import load_pairs_jsonl, pair_row_id


def _load_topics_yaml(path: Path) -> list[str]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict) or "topics" not in doc:
        raise SystemExit(f"Topic YAML must be a mapping with 'topics:' list: {path}")
    raw = doc.get("topics")
    if not isinstance(raw, list):
        raise SystemExit(f"'topics' must be a list: {path}")
    return [str(t).strip() for t in raw if str(t).strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Expand organic prompt pairs across new topics.")
    p.add_argument("--config", type=Path, default=Path("configs/rewriter.yaml"))
    p.add_argument("--pairs", type=Path, default=Path("outputs/cotrain/prompt_pairs.jsonl"))
    p.add_argument("--clusters", type=Path, default=Path("outputs/rewriter/clusters.json"))
    p.add_argument("--topics-yaml", type=Path, default=Path("configs/topics_alpaca_diverse.yaml"))
    p.add_argument("--per-pair", type=int, default=10)
    args = p.parse_args()

    repo_root = args.config.resolve().parent.parent if args.config.is_file() else Path.cwd()

    pairs_path = args.pairs
    if not pairs_path.is_absolute():
        pairs_path = (repo_root / pairs_path).resolve()
    organic = load_pairs_jsonl(pairs_path)
    if not organic:
        raise SystemExit(f"No organic pairs found: {pairs_path}")

    clusters_path = args.clusters
    if not clusters_path.is_absolute():
        clusters_path = (repo_root / clusters_path).resolve()
    if not clusters_path.is_file():
        raise SystemExit(f"Missing clusters file: {clusters_path}")
    clusters_doc = json.loads(clusters_path.read_text(encoding="utf-8"))
    clusters = list(clusters_doc.get("clusters") or [])
    if not clusters:
        raise SystemExit(f"No clusters in {clusters_path}")

    # Map organic pair_row_id -> cluster_id
    pid_to_cluster: dict[str, int] = {}
    cid_to_template: dict[int, str] = {}
    for c in clusters:
        cid = int(c.get("cluster_id", -1))
        cid_to_template[cid] = str(c.get("canonical_template", "")).strip()
        for pid in c.get("member_pair_ids", []) or []:
            pid_to_cluster[str(pid)] = cid

    topics_path = args.topics_yaml
    if not topics_path.is_absolute():
        topics_path = (repo_root / topics_path).resolve()
    if not topics_path.is_file():
        raise SystemExit(f"Missing topics YAML: {topics_path}")
    topic_pool = _load_topics_yaml(topics_path)
    if not topic_pool:
        raise SystemExit(f"No topics loaded from {topics_path}")

    organic_topics = {str(r.get("topic", "")).strip() for r in organic if str(r.get("topic", "")).strip()}
    available = [t for t in topic_pool if t not in organic_topics]
    if not available:
        raise SystemExit("No available topics after removing organic topics (pool exhausted).")

    rng = random.Random(42)
    per_pair = max(0, int(args.per_pair))

    out_path = repo_root / "outputs" / "cotrain" / "prompt_pairs_cross_topic.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    used_topic_cluster: set[tuple[str, int]] = set()
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in organic:
            pid = pair_row_id(r)
            cid = pid_to_cluster.get(pid)
            if cid is None:
                continue
            tmpl = cid_to_template.get(cid, "")
            if not tmpl:
                continue
            # sample topics not yet used for this cluster
            rng.shuffle(available)
            sampled: list[str] = []
            for t in available:
                key = (t, cid)
                if key in used_topic_cluster:
                    continue
                sampled.append(t)
                used_topic_cluster.add(key)
                if len(sampled) >= per_pair:
                    break
            for new_topic in sampled:
                out_prompt = tmpl.replace("{topic}", new_topic)
                rec = {
                    "topic": new_topic,
                    "input_prompt": str(r.get("input_prompt", r.get("input", ""))).strip(),
                    "output_prompt": out_prompt,
                    "source": "cross_topic",
                    "source_pair_id": pid,
                    "cluster_id": cid,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(json.dumps({"cross_topic_written": str(out_path), "n_written": n_written}), flush=True)


if __name__ == "__main__":
    main()

