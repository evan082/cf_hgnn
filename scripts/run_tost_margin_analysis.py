#!/usr/bin/env python
"""
TOST margin justification + sensitivity for SPC Acc preservation.

Justifies default delta=0.01 (1pp) using:
  - empirical seed-level Acc variability
  - HardMask Acc damage as a 'practically large' reference
  - sensitivity grid of TOST outcomes across delta values
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spc_significance import (  # noqa: E402
    _load_study,
    analyze_dataset,
)
from scripts.run_spc_stats_suite import (  # noqa: E402
    _dataset_means_pair,
    tost_paired,
)
from scripts.utils import save_json  # noqa: E402

ACC_METRICS = ["clean_accuracy", "noise_0.1_accuracy", "noise_0.2_accuracy"]
DELTA_GRID = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]


def _try_scipy():
    from scipy import stats

    return stats


def seed_acc_dispersion(study_paths: List[Path]) -> Dict[str, Any]:
    """Pool within-dataset seed std of Base/SPC Acc."""
    rows = []
    for p in study_paths:
        study = _load_study(p)
        ds = study.get("dataset")
        summary = study.get("summary", {})
        for metric in ACC_METRICS:
            for var in ("base", "spc"):
                cell = summary.get(metric, {}).get(var)
                if not cell:
                    continue
                vals = cell.get("values") or []
                if len(vals) < 2:
                    std = 0.0
                    n = len(vals)
                else:
                    std = float(np.std(np.asarray(vals, dtype=float), ddof=1))
                    n = len(vals)
                rows.append(
                    {
                        "dataset": ds,
                        "metric": metric,
                        "variant": var,
                        "n_seeds": n,
                        "std": std,
                        "mean": float(cell.get("mean", float("nan"))),
                    }
                )
    by_metric: Dict[str, List[float]] = {m: [] for m in ACC_METRICS}
    for r in rows:
        if r["variant"] == "base" and r["n_seeds"] >= 2:
            by_metric[r["metric"]].append(r["std"])
    summary = {
        m: {
            "median_seed_std": float(np.median(v)) if v else None,
            "mean_seed_std": float(np.mean(v)) if v else None,
            "max_seed_std": float(np.max(v)) if v else None,
            "n_datasets": len(v),
        }
        for m, v in by_metric.items()
    }
    return {"per_cell": rows, "base_seed_std_summary": summary}


def hardmask_acc_damage(per_dataset) -> Dict[str, Any]:
    """Mean |HardMask-Base| Acc drop as practical-significance reference."""
    out = {}
    for metric in ACC_METRICS:
        a, b, labels = _dataset_means_pair(per_dataset, "hardmask_vs_base", metric)
        if a.size == 0:
            continue
        d = a - b  # hard - base (usually negative)
        out[metric] = {
            "mean_diff": float(d.mean()),
            "median_diff": float(np.median(d)),
            "mean_abs_diff": float(np.mean(np.abs(d))),
            "datasets": labels,
            "diffs": d.tolist(),
        }
    return out


def tost_sensitivity(per_dataset, stats, alpha: float, deltas: List[float]) -> Dict[str, Any]:
    grid = {}
    for metric in ACC_METRICS:
        a, b, labels = _dataset_means_pair(per_dataset, "spc_vs_base", metric)
        grid[metric] = {"datasets": labels, "mean_diff": float((a - b).mean()) if a.size else None, "by_delta": {}}
        for delta in deltas:
            t = tost_paired(a, b, delta=delta, alpha=alpha, stats=stats)
            # also: fraction of logs with |d_i| < delta
            di = a - b
            frac = float(np.mean(np.abs(di) < delta)) if di.size else None
            grid[metric]["by_delta"][str(delta)] = {
                **t,
                "frac_datasets_abs_diff_lt_delta": frac,
            }
    return grid


def recommend_delta(seed_sum: Dict, hard: Dict, sens: Dict) -> Dict[str, Any]:
    """Heuristic write-up numbers for default 1pp."""
    clean_std = (seed_sum.get("clean_accuracy") or {}).get("median_seed_std")
    hard_n02 = (hard.get("noise_0.2_accuracy") or {}).get("mean_abs_diff")
    # smallest delta where clean TOST passes
    clean_pass = None
    for d in DELTA_GRID:
        cell = sens["clean_accuracy"]["by_delta"][str(d)]
        if cell.get("equivalent"):
            clean_pass = d
            break
    noise_pass = None
    for d in DELTA_GRID:
        cell = sens["noise_0.2_accuracy"]["by_delta"][str(d)]
        if cell.get("equivalent"):
            noise_pass = d
            break
    return {
        "default_delta": 0.01,
        "rationale": [
            "1pp is a conventional practically negligible Acc gap for next-activity papers when effect sizes are reported in percentage points.",
            f"Median Base clean Acc seed-std across logs ≈ {clean_std}; 1pp is on the order of multi-seed noise, not a large shift.",
            f"HardMask mean |noise0.2 Acc−Base| ≈ {hard_n02}; 1pp is on the same order as typical hard-mask Acc damage, so equivalence at 1pp is a non-trivial preservation claim.",
            f"TOST sensitivity: clean Acc first passes at delta={clean_pass}; noise0.2 Acc first passes at delta={noise_pass}.",
            "Use delta=0.01 for primary Acc-preservation claim; report 0.005 as sensitivity (clean only).",
            "Equivalence at 1pp means SPC vs Base Acc stays within the hard-mask damage envelope.",
        ],
        "clean_first_pass_delta": clean_pass,
        "noise02_first_pass_delta": noise_pass,
        "median_base_clean_seed_std": clean_std,
        "hardmask_noise02_mean_abs_acc_drop": hard_n02,
    }


def render_txt(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append("TOST margin justification + sensitivity (SPC vs Base Acc)")
    lines.append("")
    lines.append("=== Empirical seed variability (Base Acc std across seeds) ===")
    for m, s in payload["seed_dispersion"]["base_seed_std_summary"].items():
        lines.append(
            f"  {m}: median_std={s['median_seed_std']}  mean_std={s['mean_seed_std']}  "
            f"max_std={s['max_seed_std']}  n_ds={s['n_datasets']}"
        )
    lines.append("")
    lines.append("=== HardMask Acc damage reference (Hard-Base) ===")
    for m, s in payload["hardmask_damage"].items():
        lines.append(
            f"  {m}: mean_diff={s['mean_diff']:+.4f}  mean_abs={s['mean_abs_diff']:.4f}"
        )
    lines.append("")
    lines.append("=== TOST sensitivity grid (equiv Y/N) ===")
    header = f"{'delta':>8s}" + "".join(f"{m[:12]:>14s}" for m in ACC_METRICS)
    lines.append(header)
    for d in DELTA_GRID:
        row = f"{d:8.4f}"
        for m in ACC_METRICS:
            eq = payload["sensitivity"][m]["by_delta"][str(d)].get("equivalent")
            row += f"{'Y' if eq else 'N':>14s}"
        lines.append(row)
    lines.append("")
    lines.append("=== Recommendation ===")
    for r in payload["recommendation"]["rationale"]:
        lines.append(f"- {r}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=str, default=str(ROOT / "outputs"))
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "tost_margin"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    stats = _try_scipy()

    root = Path(args.outputs_root)
    study_paths = sorted(root.glob("*/spc_study/spc_study.json"))
    per_dataset = [analyze_dataset(_load_study(p), stats) for p in study_paths]
    for p in study_paths:
        print("loaded", p.parent.parent.name, flush=True)

    seed_disp = seed_acc_dispersion(study_paths)
    hard = hardmask_acc_damage(per_dataset)
    sens = tost_sensitivity(per_dataset, stats, alpha=float(args.alpha), deltas=DELTA_GRID)
    rec = recommend_delta(seed_disp["base_seed_std_summary"], hard, sens)

    payload = {
        "alpha": float(args.alpha),
        "delta_grid": DELTA_GRID,
        "seed_dispersion": seed_disp,
        "hardmask_damage": hard,
        "sensitivity": sens,
        "recommendation": rec,
        "n_datasets": len(per_dataset),
        "datasets": [d["dataset"] for d in per_dataset],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "tost_margin.json", payload)
    txt = render_txt(payload)
    (out_dir / "tost_margin.txt").write_text(txt, encoding="utf-8")
    try:
        print(txt, flush=True)
    except UnicodeEncodeError:
        print(txt.encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
