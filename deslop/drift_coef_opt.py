#!/usr/bin/env python3
"""
Gradient-based search for drift penalty coefficients (alpha_semantic, alpha_rouge, alpha_bertscore).

The evolutionary objective minimized per essay is (conceptually)::

    optimization_slop = detector_slop
        + alpha_semantic * drift_semantic
        + alpha_rouge * drift_rouge_l
        + alpha_bertscore * drift_bertscore

where each drift_* term is already a non-negative penalty in ``[0, 1]`` scale from
``composite_drift_penalty`` (see ``deslop/similarity.py``). Tuning the alphas is a constrained
continuous optimization problem: we want more essays below a slop threshold (``fool`` the
detector) without exploding topic drift. This module approximates the **fool rate** with a
smooth sigmoid in ``optimization_slop`` so PyTorch can optimize via Adam, and compares against a
coarse **grid search** baseline (no gradients).

Feature rows (JSONL) may be produced from deslop essay logs (which already store drift
components) or from ``deslop/export_pair_drift_features.py`` for held-out pairs aligned with
``rewriter_split_manifest.json``.

For course reports, the **essay pool** from logs is usually the primary curve to plot (more rows,
smoother Adam trajectory); the **export** path is the rigorous holdout alternative—state limitations
honestly if you rely on the pool.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml


def _maybe_wandb_init(project: str | None, entity: str | None, config: dict[str, Any]) -> Any:
    import os

    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    kwargs: dict[str, Any] = {"project": project or "promptlab-v2", "config": config}
    if entity:
        kwargs["entity"] = entity
    return wandb.init(**kwargs)


def normalize_feature_row(r: dict[str, Any]) -> dict[str, float] | None:
    """Map heterogeneous log keys to a single feature vector; skip incomplete rows."""
    s = r.get("detector_slop")
    if s is None and r.get("slop_score") is not None:
        s = r["slop_score"]
    if s is None:
        return None
    if r.get("drift_semantic") is None or r.get("drift_rouge_l") is None:
        return None
    d_b = float(r.get("drift_bertscore", 0.0))
    if float(r.get("bertscore_applied", 1.0)) < 0.5:
        d_b = 0.0
    return {
        "detector_slop": float(s),
        "drift_semantic": float(r["drift_semantic"]),
        "drift_rouge_l": float(r["drift_rouge_l"]),
        "drift_bertscore": d_b,
        "pair_row_id": str(r.get("pair_row_id", "")),
    }


def feature_file_has_pair_row_ids(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("pair_row_id"):
            return True
    return False


def load_feature_rows(
    path: Path,
    *,
    allowed_ids: set[str] | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        norm = normalize_feature_row(raw)
        if norm is None:
            continue
        pid = norm.get("pair_row_id", "")
        if allowed_ids is not None:
            if not pid or pid not in allowed_ids:
                continue
        rows.append(norm)
    return rows


def load_split_ids(manifest_path: Path, split_name: str) -> set[str] | None:
    if not manifest_path.is_file():
        return None
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = m.get("splits", {}).get(split_name)
    if not ids:
        return set()
    return set(str(x) for x in ids)


def append_deslop_round_to_feature_pool(round_log_path: Path, pool_path: Path) -> int:
    """
    Append essay rows from a deslop JSONL log into a pooled feature file.

    Only lines that contain drift_semantic / drift_rouge_l (alignment was enabled) are copied.
    Deduplicates by a short hash of the essay text so reruns do not explode the pool.
    """
    if not round_log_path.is_file():
        return 0
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if pool_path.is_file():
        for line in pool_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            essay = str(r.get("essay", ""))[:4000]
            seen.add(str(hash(essay)))
    added = 0
    with pool_path.open("a", encoding="utf-8") as wf:
        for line in round_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "drift_semantic" not in r or "drift_rouge_l" not in r:
                continue
            essay = str(r.get("essay", ""))[:4000]
            h = str(hash(essay))
            if h in seen:
                continue
            seen.add(h)
            row = {
                "detector_slop": float(r.get("slop_score", r.get("detector_slop", 1.0))),
                "drift_semantic": float(r["drift_semantic"]),
                "drift_rouge_l": float(r["drift_rouge_l"]),
                "drift_bertscore": float(r.get("drift_bertscore", 0.0)),
                "bertscore_applied": float(r.get("bertscore_applied", 0.0)),
                "topic": r.get("topic", ""),
                "source": "deslop_log",
            }
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
    return added


def _stack_features(rows: list[dict[str, float]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.tensor([x["detector_slop"] for x in rows], dtype=torch.float32)
    d0 = torch.tensor([x["drift_semantic"] for x in rows], dtype=torch.float32)
    d1 = torch.tensor([x["drift_rouge_l"] for x in rows], dtype=torch.float32)
    d2 = torch.tensor([x["drift_bertscore"] for x in rows], dtype=torch.float32)
    return s, d0, d1, d2


def optimization_slop(
    s: torch.Tensor,
    d0: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Broadcast (N,) with alpha (3,)."""
    return s + alpha[0] * d0 + alpha[1] * d1 + alpha[2] * d2


def fool_rate_hard(opt_slop: torch.Tensor, threshold: float) -> float:
    return float((opt_slop < threshold).float().mean().item())


def fool_rate_soft(opt_slop: torch.Tensor, threshold: float, k: float) -> torch.Tensor:
    """Differentiable surrogate: high when opt_slop is below threshold."""
    return torch.sigmoid(-k * (opt_slop - threshold)).mean()


@dataclass
class CoefSearchResult:
    alpha_semantic: float
    alpha_rouge: float
    alpha_bertscore: float
    fool_rate_hard: float
    mean_optimization_slop: float
    name: str = ""


@dataclass
class DriftCoefOptimizer:
    """
    Compare grid search vs Adam on a sigmoid-smoothed fool rate.

    Non-negativity: unconstrained parameters ``u`` with ``alpha = softplus(u)`` so Adam never
    needs projection; gradients remain well-defined at zero.
    """

    slop_threshold: float = 0.5
    sigmoid_temperature: float = 40.0

    def grid_search(
        self,
        rows: list[dict[str, float]],
        *,
        grid_semantic: list[float],
        grid_rouge: list[float],
        grid_bertscore: list[float],
    ) -> CoefSearchResult:
        if not rows:
            return CoefSearchResult(1.0, 0.35, 0.5, 0.0, float("nan"), name="grid_empty")
        s, d0, d1, d2 = _stack_features(rows)
        best: CoefSearchResult | None = None
        for a0 in grid_semantic:
            for a1 in grid_rouge:
                for a2 in grid_bertscore:
                    alpha = torch.tensor([a0, a1, a2], dtype=torch.float32)
                    opt = optimization_slop(s, d0, d1, d2, alpha)
                    fr = fool_rate_hard(opt, self.slop_threshold)
                    mean_slop = float(opt.mean().item())
                    cand = CoefSearchResult(
                        float(a0), float(a1), float(a2), fr, mean_slop, name="grid"
                    )
                    if best is None or fr > best.fool_rate_hard or (
                        math.isclose(fr, best.fool_rate_hard) and mean_slop < best.mean_optimization_slop
                    ):
                        best = cand
        assert best is not None
        return best

    def adam_optimize(
        self,
        rows: list[dict[str, float]],
        *,
        steps: int,
        lr: float,
        init_alpha: tuple[float, float, float] = (1.0, 0.35, 0.5),
        log_fn: Any | None = None,
    ) -> tuple[CoefSearchResult, list[dict[str, Any]]]:
        trajectory: list[dict[str, Any]] = []
        if not rows:
            return (
                CoefSearchResult(1.0, 0.35, 0.5, 0.0, float("nan"), name="adam_empty"),
                trajectory,
            )
        s, d0, d1, d2 = _stack_features(rows)
        # Invert softplus at initialization
        a0, a1, a2 = init_alpha
        u_init = [
            math.log(max(math.expm1(a0), 1e-6)),
            math.log(max(math.expm1(a1), 1e-6)),
            math.log(max(math.expm1(a2), 1e-6)),
        ]
        u = torch.nn.Parameter(torch.tensor(u_init, dtype=torch.float32))
        opt = torch.optim.Adam([u], lr=lr)
        k = float(self.sigmoid_temperature)
        thresh = float(self.slop_threshold)
        best_state: CoefSearchResult | None = None
        for step in range(steps):
            opt.zero_grad()
            alpha = F.softplus(u)
            opt_slop = optimization_slop(s, d0, d1, d2, alpha)
            soft_fr = fool_rate_soft(opt_slop, thresh, k)
            loss = -soft_fr
            loss.backward()
            opt.step()
            with torch.no_grad():
                alpha_v = F.softplus(u)
                hard_fr = fool_rate_hard(
                    optimization_slop(s, d0, d1, d2, alpha_v), thresh
                )
                mean_sl = float(optimization_slop(s, d0, d1, d2, alpha_v).mean().item())
            rec = {
                "step": step,
                "alpha_semantic": float(alpha_v[0].item()),
                "alpha_rouge": float(alpha_v[1].item()),
                "alpha_bertscore": float(alpha_v[2].item()),
                "fool_rate_approx": float(soft_fr.item()),
                "fool_rate_hard": hard_fr,
                "mean_optimization_slop": mean_sl,
                "sigmoid_temperature": float(self.sigmoid_temperature),
                "slop_threshold": float(self.slop_threshold),
            }
            trajectory.append(rec)
            if log_fn:
                log_fn(rec)
            cand = CoefSearchResult(
                float(alpha_v[0].item()),
                float(alpha_v[1].item()),
                float(alpha_v[2].item()),
                hard_fr,
                mean_sl,
                name=f"adam_step_{step}",
            )
            if best_state is None or hard_fr > best_state.fool_rate_hard:
                best_state = cand
        assert best_state is not None
        return best_state, trajectory


def sensitivity_sweep(
    rows: list[dict[str, float]],
    *,
    base: tuple[float, float, float],
    sweep_dim: int,
    values: list[float],
    threshold: float,
) -> list[tuple[float, float]]:
    """Return (value, fool_rate_hard) holding other two coefficients at ``base``."""
    s, d0, d1, d2 = _stack_features(rows)
    out: list[tuple[float, float]] = []
    for v in values:
        a = list(base)
        a[sweep_dim] = v
        alpha = torch.tensor(a, dtype=torch.float32)
        opt = optimization_slop(s, d0, d1, d2, alpha)
        out.append((v, fool_rate_hard(opt, threshold)))
    return out


def run_drift_coef_optimization_from_config_section(
    section: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """
    Execute grid + Adam from a ``drift_coef_opt`` YAML dict. Writes JSON to
    ``optimized_output_path``. Returns the on-disk document or None if skipped.
    """
    if not section.get("enabled", False):
        return None
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    from rewriter.dataset import (
        SPLIT_MANIFEST_PATH_DEFAULT,
        assert_split_manifest_matches_rewriter_and_drift_configs,
        resolve_split_manifest_path,
    )

    assert_split_manifest_matches_rewriter_and_drift_configs(cwd=repo_root)
    feat_path = Path(section.get("features_jsonl", "outputs/cotrain/drift_features_holdout.jsonl"))
    if not feat_path.is_absolute():
        feat_path = (repo_root / feat_path).resolve()
    if not feat_path.is_file():
        print(f"drift_coef_opt: skip — features file missing: {feat_path}", flush=True)
        return None

    manifest = resolve_split_manifest_path(
        section.get("split_manifest_path", SPLIT_MANIFEST_PATH_DEFAULT),
        cwd=repo_root,
    )
    split_name = str(section.get("split_for_optimization", "test"))
    allowed = load_split_ids(manifest, split_name)
    use_manifest = bool(allowed)
    rows = load_feature_rows(feat_path, allowed_ids=allowed if use_manifest else None)
    if not rows:
        if use_manifest and feature_file_has_pair_row_ids(feat_path):
            print(
                "drift_coef_opt: skip — manifest split matched no rows in features file.",
                flush=True,
            )
            return None
        # Pool from deslop logs: rows lack pair_row_id — use full file.
        rows = load_feature_rows(feat_path, allowed_ids=None)
    if len(rows) < 3:
        print(f"drift_coef_opt: skip — too few feature rows ({len(rows)}).", flush=True)
        return None

    opt = DriftCoefOptimizer(
        slop_threshold=float(section.get("slop_threshold", 0.5)),
        sigmoid_temperature=float(section.get("sigmoid_temperature", 40.0)),
    )
    grid_semantic = [float(x) for x in section.get("grid_semantic", [0.5, 1.0, 1.5])]
    grid_rouge = [float(x) for x in section.get("grid_rouge", [0.2, 0.35, 0.5])]
    grid_bertscore = [float(x) for x in section.get("grid_bertscore", [0.0, 0.25, 0.5])]

    grid_best = opt.grid_search(
        rows,
        grid_semantic=grid_semantic,
        grid_rouge=grid_rouge,
        grid_bertscore=grid_bertscore,
    )

    wandb_run = _maybe_wandb_init(
        section.get("wandb_project"),
        section.get("wandb_entity"),
        {
            "mode": "drift_coef_opt",
            "sigmoid_temperature": float(opt.sigmoid_temperature),
            "slop_threshold": float(opt.slop_threshold),
            **section,
        },
    )
    if wandb_run is not None:
        import wandb

        wandb.log(
            {
                "meta/sigmoid_temperature": float(opt.sigmoid_temperature),
                "meta/slop_threshold": float(opt.slop_threshold),
            }
        )

    def _wandb_log(rec: dict[str, Any]) -> None:
        if wandb_run is not None:
            import wandb

            wandb.log(rec)

    adam_best, traj = opt.adam_optimize(
        rows,
        steps=int(section.get("adam_steps", 150)),
        lr=float(section.get("adam_lr", 0.08)),
        init_alpha=(
            float(grid_best.alpha_semantic),
            float(grid_best.alpha_rouge),
            float(grid_best.alpha_bertscore),
        ),
        log_fn=lambda r: _wandb_log(r),
    )

    traj_path = Path(
        section.get("trajectory_jsonl", "outputs/cotrain/drift_coef_opt_trajectory.jsonl")
    )
    if not traj_path.is_absolute():
        traj_path = (repo_root / traj_path).resolve()
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with traj_path.open("w", encoding="utf-8") as wf:
        for row in traj:
            wf.write(json.dumps(row) + "\n")

    sweep_vals = [float(x) for x in section.get("sensitivity_grid", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5])]
    base_adam = (adam_best.alpha_semantic, adam_best.alpha_rouge, adam_best.alpha_bertscore)
    sens_sem = sensitivity_sweep(
        rows, base=base_adam, sweep_dim=0, values=sweep_vals, threshold=opt.slop_threshold
    )
    sens_rouge = sensitivity_sweep(
        rows, base=base_adam, sweep_dim=1, values=sweep_vals, threshold=opt.slop_threshold
    )
    sens_bert = sensitivity_sweep(
        rows, base=base_adam, sweep_dim=2, values=sweep_vals, threshold=opt.slop_threshold
    )

    out_doc = {
        "sigmoid_temperature": float(opt.sigmoid_temperature),
        "slop_threshold": float(opt.slop_threshold),
        "drift_weights": {
            "alpha_semantic": adam_best.alpha_semantic,
            "alpha_rouge": adam_best.alpha_rouge,
            "alpha_bertscore": adam_best.alpha_bertscore,
        },
        "grid_search": {
            "alpha_semantic": grid_best.alpha_semantic,
            "alpha_rouge": grid_best.alpha_rouge,
            "alpha_bertscore": grid_best.alpha_bertscore,
            "fool_rate_hard": grid_best.fool_rate_hard,
            "mean_optimization_slop": grid_best.mean_optimization_slop,
        },
        "adam": {
            "alpha_semantic": adam_best.alpha_semantic,
            "alpha_rouge": adam_best.alpha_rouge,
            "alpha_bertscore": adam_best.alpha_bertscore,
            "fool_rate_hard": adam_best.fool_rate_hard,
            "mean_optimization_slop": adam_best.mean_optimization_slop,
        },
        "sensitivity_alpha_semantic": [
            {"alpha_semantic": a, "fool_rate_hard": f} for a, f in sens_sem
        ],
        "sensitivity_alpha_rouge": [{"alpha_rouge": a, "fool_rate_hard": f} for a, f in sens_rouge],
        "sensitivity_alpha_bertscore": [
            {"alpha_bertscore": a, "fool_rate_hard": f} for a, f in sens_bert
        ],
        "n_rows": len(rows),
        "trajectory_jsonl": str(traj_path),
    }
    out_path = Path(
        section.get("optimized_output_path", "outputs/cotrain/optimized_drift_coefs.json")
    )
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"drift_coef_opt": "wrote", "path": str(out_path), "adam": out_doc["adam"]}), flush=True)

    if wandb_run is not None:
        import wandb

        wandb.log(
            {
                "final/grid_fool_rate": grid_best.fool_rate_hard,
                "final/adam_fool_rate": adam_best.fool_rate_hard,
                "final/alpha_semantic": adam_best.alpha_semantic,
                "final/alpha_rouge": adam_best.alpha_rouge,
                "final/alpha_bertscore": adam_best.alpha_bertscore,
            }
        )
        wandb.finish()

    return out_doc


def load_optimized_drift_weights(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    w = doc.get("drift_weights", doc)
    keys = ("alpha_semantic", "alpha_rouge", "alpha_bertscore")
    if not all(k in w for k in keys):
        return None
    return {k: float(w[k]) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize drift penalty coefficients (Idea 3).")
    ap.add_argument("--config", type=Path, default=Path("configs/deslop.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    section = cfg.get("drift_coef_opt") or {}
    run_drift_coef_optimization_from_config_section(section, repo_root=Path.cwd())


if __name__ == "__main__":
    main()
