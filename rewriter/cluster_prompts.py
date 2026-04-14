#!/usr/bin/env python3
"""
Cluster evolved output prompts from co-training pairs and derive canonical templates.

Writes: outputs/rewriter/clusters.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from deslop import similarity as sim
from rewriter.dataset import load_pairs_jsonl, pair_row_id


def _cosine_distance_matrix(embs: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance matrix (1 - cosine_sim). embs: (n, d)."""
    x = torch.nn.functional.normalize(embs, p=2, dim=1)
    sims = x @ x.T
    return 1.0 - sims


def _medoid_index(cluster_embs: torch.Tensor) -> int:
    """
    Return index of the medoid (min mean cosine distance to other members).
    """
    if cluster_embs.size(0) == 1:
        return 0
    d = _cosine_distance_matrix(cluster_embs)
    mean_d = d.mean(dim=1)
    return int(torch.argmin(mean_d).item())


def _canonicalize_template(prompt: str, *, topic: str) -> tuple[str, bool]:
    """
    Replace the literal topic substring with '{topic}' when possible.
    """
    t = (topic or "").strip()
    if not t:
        return prompt, False
    if t in prompt:
        return prompt.replace(t, "{topic}"), True
    # simple fallback: try case-insensitive replacement
    low_p = prompt.lower()
    low_t = t.lower()
    i = low_p.find(low_t)
    if i >= 0:
        return prompt[:i] + "{topic}" + prompt[i + len(t) :], True
    return prompt, False


def _load_cfg(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="Cluster evolved output prompts into canonical templates.")
    p.add_argument("--config", type=Path, default=Path("configs/rewriter.yaml"))
    p.add_argument("--pairs", type=Path, default=Path("outputs/cotrain/prompt_pairs.jsonl"))
    args = p.parse_args()

    cfg_path = args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = _load_cfg(cfg_path)
    repo_root = cfg_path.resolve().parent.parent

    pairs_path = args.pairs
    if not pairs_path.is_absolute():
        pairs_path = (repo_root / pairs_path).resolve()
    pairs = load_pairs_jsonl(pairs_path)
    if not pairs:
        raise SystemExit(f"No pairs found in {pairs_path}")

    out_prompts: list[str] = []
    topics: list[str] = []
    pair_ids: list[str] = []
    for r in pairs:
        out = str(r.get("output_prompt", r.get("output", ""))).strip()
        if not out:
            continue
        out_prompts.append(out)
        topics.append(str(r.get("topic", "")).strip())
        pair_ids.append(pair_row_id(r))

    if not out_prompts:
        raise SystemExit(f"No output_prompt fields found in {pairs_path}")

    k = int(cfg.get("n_clusters", cfg.get("kmeans_k", 5)) or 5)
    k = max(1, min(k, len(out_prompts)))

    model_name = str(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    embedder = sim._get_embedder(model_name)
    embs = embedder.encode(out_prompts, convert_to_tensor=True)
    if not isinstance(embs, torch.Tensor):
        embs = torch.tensor(embs)
    embs = embs.float()

    try:
        from sklearn.cluster import KMeans
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "scikit-learn is required for k-means clustering. "
            "Install it (e.g. `uv add scikit-learn`) and re-run."
        ) from e

    km = KMeans(n_clusters=k, random_state=int(cfg.get("kmeans_seed", 42)), n_init="auto")
    labels = km.fit_predict(embs.cpu().numpy())
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    clusters: list[dict[str, Any]] = []
    warn_small: list[int] = []
    for cid in range(k):
        idxs = [i for i, lab in enumerate(labels) if int(lab) == cid]
        if not idxs:
            continue
        c_embs = embs[idxs]
        mid_local = _medoid_index(c_embs)
        mid_i = idxs[mid_local]
        medoid_prompt = out_prompts[mid_i]
        medoid_topic = topics[mid_i]
        canonical_template, replaced = _canonicalize_template(medoid_prompt, topic=medoid_topic)
        if not replaced and "{topic}" not in canonical_template:
            # Keep output stable but signal in report.
            canonical_template = canonical_template
        member_ids = [pair_ids[i] for i in idxs]
        member_topics = [topics[i] for i in idxs if topics[i]]
        if len(idxs) < 3:
            warn_small.append(cid)
        clusters.append(
            {
                "cluster_id": cid,
                "n_organic": len(idxs),
                "medoid_prompt": medoid_prompt,
                "canonical_template": canonical_template,
                "member_pair_ids": member_ids,
                "member_topics": member_topics,
                # Downstream helpers (not required by schema, but useful).
                "centroid": centroids[cid].tolist(),
            }
        )

    out_obj = {"k": k, "clusters": clusters}
    out_path = repo_root / "outputs" / "rewriter" / "clusters.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable report
    print(f"clusters.json written: {out_path} (k={k}, n_pairs={len(out_prompts)})")
    for c in sorted(clusters, key=lambda x: int(x["cluster_id"])):
        cid = int(c["cluster_id"])
        n = int(c["n_organic"])
        print("\n" + "=" * 80)
        print(f"cluster {cid}  n={n}")
        print("- canonical_template:")
        print(c["canonical_template"])
        if c["member_topics"]:
            print("- member_topics:")
            for t in c["member_topics"][:60]:
                print(f"  - {t}")
            if len(c["member_topics"]) > 60:
                print(f"  ... ({len(c['member_topics']) - 60} more)")
        else:
            print("- member_topics: (none)")

    if warn_small:
        for cid in warn_small:
            print(f"WARNING: cluster {cid} has fewer than 3 organic members", flush=True)


if __name__ == "__main__":
    main()
