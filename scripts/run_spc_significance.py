#!/usr/bin/env python
"""
SPC significance tests from existing spc_study.json files.

Within-dataset: paired t-test across seeds (n=3).
Cross-dataset: paired t + Wilcoxon on per-dataset mean deltas (n=#datasets).

Comparisons: SPC vs Base, SPC vs Soft, Soft vs Base, HardMask vs Base.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import save_json

METRICS = [
    "clean_accuracy",
    "noise_0.1_accuracy",
    "noise_0.2_accuracy",
    "noise_0.1_legal_rate",
    "noise_0.2_legal_rate",
]

COMPARISONS = [
    ("spc", "base", "spc_vs_base"),
    ("spc", "soft", "spc_vs_soft"),
    ("soft", "base", "soft_vs_base"),
    ("hardmask", "base", "hardmask_vs_base"),
]

VARIANT_ALIASES = {
    "base": "base",
    "hardmask": "hardmask",
    "soft": "soft",
    "softlegal": "soft",
    "spc": "spc",
}


def _try_import_scipy():
    from scipy import stats

    return stats


def _cohen_dz(diffs: np.ndarray) -> Optional[float]:
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size < 2:
        return None
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0 if float(np.mean(diffs)) == 0.0 else None
    return float(np.mean(diffs) / sd)


def _paired_tests(a: np.ndarray, b: np.ndarray, stats) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return {"status": "missing", "n": 0}
    diffs = a - b
    if np.allclose(diffs, 0.0):
        return {
            "status": "identical",
            "n": int(diffs.size),
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "t_stat": None,
            "t_pvalue": None,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "cohens_dz": 0.0,
            "a_mean": float(np.mean(a)),
            "b_mean": float(np.mean(b)),
            "diffs": diffs.tolist(),
        }

    out: Dict[str, Any] = {
        "status": "ok",
        "n": int(diffs.size),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0,
        "a_mean": float(np.mean(a)),
        "b_mean": float(np.mean(b)),
        "diffs": diffs.tolist(),
        "cohens_dz": _cohen_dz(diffs),
    }

    if diffs.size >= 2:
        t_res = stats.ttest_rel(a, b)
        out["t_stat"] = float(t_res.statistic) if t_res.statistic is not None else None
        out["t_pvalue"] = float(t_res.pvalue) if t_res.pvalue is not None else None
    else:
        out["t_stat"] = None
        out["t_pvalue"] = None

    # Wilcoxon needs nonzero diffs; with n=3 two-sided min p ~ 0.25
    nz = diffs[np.abs(diffs) > 1e-15]
    if nz.size >= 1:
        try:
            w_res = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
            out["wilcoxon_stat"] = float(w_res.statistic)
            out["wilcoxon_pvalue"] = float(w_res.pvalue)
        except ValueError:
            out["wilcoxon_stat"] = None
            out["wilcoxon_pvalue"] = None
    else:
        out["wilcoxon_stat"] = None
        out["wilcoxon_pvalue"] = None

    return out


def _load_study(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _values_by_seed(study: Dict[str, Any], metric: str) -> Dict[str, Dict[int, float]]:
    """variant -> {seed: value}."""
    out: Dict[str, Dict[int, float]] = {}
    for row in study.get("per_run", []):
        var = VARIANT_ALIASES.get(str(row.get("variant", "")).lower())
        if var is None:
            continue
        if metric not in row:
            continue
        seed = int(row["seed"])
        out.setdefault(var, {})[seed] = float(row[metric])
    return out


def _aligned_pair(
    by_var: Dict[str, Dict[int, float]], left: str, right: str
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    seeds = sorted(set(by_var.get(left, {})) & set(by_var.get(right, {})))
    a = np.array([by_var[left][s] for s in seeds], dtype=float)
    b = np.array([by_var[right][s] for s in seeds], dtype=float)
    return a, b, seeds


def analyze_dataset(study: Dict[str, Any], stats) -> Dict[str, Any]:
    dataset = str(study.get("dataset", "unknown"))
    per_cmp: Dict[str, Any] = {}
    for left, right, name in COMPARISONS:
        per_cmp[name] = {}
        for metric in METRICS:
            by_var = _values_by_seed(study, metric)
            a, b, seeds = _aligned_pair(by_var, left, right)
            res = _paired_tests(a, b, stats)
            res["seeds"] = seeds
            per_cmp[name][metric] = res
    return {"dataset": dataset, "comparisons": per_cmp}


def cross_dataset(
    per_dataset: List[Dict[str, Any]], stats
) -> Dict[str, Any]:
    """Pair dataset-level means: use mean_diff from within-dataset tests as Δ."""
    out: Dict[str, Any] = {}
    for _left, _right, name in COMPARISONS:
        out[name] = {}
        for metric in METRICS:
            deltas = []
            labels = []
            a_means = []
            b_means = []
            for ds in per_dataset:
                cell = ds["comparisons"][name][metric]
                if cell.get("status") in ("missing",):
                    continue
                if cell.get("n", 0) == 0:
                    continue
                labels.append(ds["dataset"])
                deltas.append(float(cell["mean_diff"]))
                a_means.append(float(cell["a_mean"]))
                b_means.append(float(cell["b_mean"]))
            a = np.asarray(a_means, dtype=float)
            b = np.asarray(b_means, dtype=float)
            res = _paired_tests(a, b, stats)
            res["datasets"] = labels
            # mean_diff from a-b equals mean of within-dataset mean_diffs when complete
            out[name][metric] = res
    return out


def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_num(x: Optional[float], digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def render_txt(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("SPC significance tests")
    lines.append("Within-dataset: paired t across seeds (n=3 typically).")
    lines.append("Cross-dataset: paired t + Wilcoxon on per-dataset means.")
    lines.append("Alpha=0.05 two-sided; Wilcoxon with n=3 has very low power.")
    lines.append("")

    lines.append("=== Cross-dataset (primary) ===")
    for _l, _r, name in COMPARISONS:
        lines.append(f"\n[{name}]")
        lines.append(
            f"{'metric':28s} {'mean_d':>8s} {'t':>8s} {'p_t':>10s} {'p_W':>10s} {'dz':>7s} status"
        )
        for metric in METRICS:
            cell = payload["cross_dataset"][name][metric]
            status = cell.get("status", "")
            sig = ""
            p = cell.get("t_pvalue")
            if status == "ok" and p is not None and p < 0.05:
                sig = " *"
            lines.append(
                f"{metric:28s} {_fmt_num(cell.get('mean_diff')):>8s} "
                f"{_fmt_num(cell.get('t_stat')):>8s} {_fmt_p(cell.get('t_pvalue')):>10s} "
                f"{_fmt_p(cell.get('wilcoxon_pvalue')):>10s} {_fmt_num(cell.get('cohens_dz'), 2):>7s} "
                f"{status}{sig}"
            )

    lines.append("\n=== Within-dataset (spc_vs_base) ===")
    for ds in payload["per_dataset"]:
        lines.append(f"\n# {ds['dataset']}")
        lines.append(
            f"{'metric':28s} {'mean_d':>8s} {'p_t':>10s} status"
        )
        for metric in METRICS:
            cell = ds["comparisons"]["spc_vs_base"][metric]
            p = cell.get("t_pvalue")
            sig = " *" if cell.get("status") == "ok" and p is not None and p < 0.05 else ""
            lines.append(
                f"{metric:28s} {_fmt_num(cell.get('mean_diff')):>8s} "
                f"{_fmt_p(p):>10s} {cell.get('status')}{sig}"
            )

    lines.append("\n* p_t < 0.05 (two-sided paired t)")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="SPC significance from spc_study.json")
    ap.add_argument(
        "--outputs-root",
        type=str,
        default=str(ROOT / "outputs"),
        help="Root containing */spc_study/spc_study.json",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs" / "spc_significance"),
    )
    args = ap.parse_args()
    stats = _try_import_scipy()

    root = Path(args.outputs_root)
    study_paths = sorted(root.glob("*/spc_study/spc_study.json"))
    if not study_paths:
        print(f"No spc_study.json under {root}", flush=True)
        return 1

    per_dataset = []
    for p in study_paths:
        study = _load_study(p)
        per_dataset.append(analyze_dataset(study, stats))
        print(f"loaded {p.parent.parent.name}", flush=True)

    cross = cross_dataset(per_dataset, stats)
    payload = {
        "n_datasets": len(per_dataset),
        "datasets": [d["dataset"] for d in per_dataset],
        "metrics": METRICS,
        "comparisons": [c[2] for c in COMPARISONS],
        "alpha": 0.05,
        "notes": [
            "Within-dataset paired t uses n=seeds (typically 3); low power.",
            "Wilcoxon with n=3 rarely reaches p<0.05; reported for completeness.",
            "Cross-dataset tests use per-dataset mean metric values (paired across variants).",
            "McNemar not run in this script; see scripts/run_spc_mcnemar.py.",
        ],
        "per_dataset": per_dataset,
        "cross_dataset": cross,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "significance.json"
    txt_path = out_dir / "significance.txt"
    save_json(json_path, payload)
    txt = render_txt(payload)
    txt_path.write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
