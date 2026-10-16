#!/usr/bin/env python
"""
SPC stats suite (cross-dataset primary):

  1) Paired t + 95% CI (Student + bootstrap) + Cohen's dz
  2) Wilcoxon signed-rank (paired, two-sided)
  3) Paired permutation test
  4) Holm correction on a predeclared superiority family
  5) TOST equivalence for accuracy preservation
  6) Pointer / summary of sample-level McNemar (run_spc_mcnemar.py)

Reads existing outputs/*/spc_study/spc_study.json (no retrain).
Optionally merges outputs/spc_mcnemar/spc_mcnemar.json if present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spc_significance import (  # noqa: E402
    COMPARISONS,
    METRICS,
    _cohen_dz,
    _load_study,
    analyze_dataset,
)
from scripts.utils import save_json  # noqa: E402


def _seed_key(s: str) -> int:
    return zlib.adler32(s.encode("utf-8")) & 0x7FFFFFFF

# Predeclared superiority family for Holm (cross-dataset, two-sided).
# Primary claim: noise PCP up. Secondary: Soft gate lift / HardMask legal.
HOLM_FAMILY: List[Tuple[str, str]] = [
    ("spc_vs_base", "noise_0.1_legal_rate"),
    ("spc_vs_base", "noise_0.2_legal_rate"),
    ("spc_vs_soft", "noise_0.1_legal_rate"),
    ("spc_vs_soft", "noise_0.2_legal_rate"),
    ("hardmask_vs_base", "noise_0.1_legal_rate"),
    ("hardmask_vs_base", "noise_0.2_legal_rate"),
]

# TOST targets for "accuracy preservation" (SPC vs Base).
TOST_TARGETS: List[Tuple[str, str]] = [
    ("spc_vs_base", "clean_accuracy"),
    ("spc_vs_base", "noise_0.1_accuracy"),
    ("spc_vs_base", "noise_0.2_accuracy"),
]


def _try_scipy():
    from scipy import stats

    return stats


def holm_correction(pvalues: Sequence[Optional[float]], alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Holm–Bonferroni step-down; None p -> skipped."""
    indexed = [(i, float(p)) for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    m = len(indexed)
    out: List[Dict[str, Any]] = [
        {"raw_p": None if p is None else float(p), "holm_p": None, "reject": False, "rank": None}
        for p in pvalues
    ]
    if m == 0:
        return out
    indexed.sort(key=lambda x: x[1])
    holm_adj = [None] * m
    prev = 0.0
    for rank, (orig_i, p) in enumerate(indexed):
        adj = (m - rank) * p
        adj = min(1.0, max(prev, adj))  # enforce monotonicity
        prev = adj
        holm_adj[rank] = adj
        out[orig_i] = {
            "raw_p": p,
            "holm_p": float(adj),
            "reject": bool(adj <= alpha),
            "rank": rank + 1,
            "holm_threshold": alpha / (m - rank),
        }
    return out


def paired_permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 20000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Two-sided paired permutation: randomly flip signs of (a-b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    obs = float(np.mean(d))
    if n == 0:
        return {"pvalue": None, "n_perm": 0, "obs_mean": obs, "status": "empty"}
    if np.allclose(d, 0.0):
        return {"pvalue": 1.0, "n_perm": n_perm, "obs_mean": 0.0, "status": "identical"}
    rng = np.random.default_rng(seed)
    # signs: +/- 1
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = (signs * d).mean(axis=1)
    p = float((np.abs(null_means) >= abs(obs)).mean())
    # add +1 correction optional; use raw proportion with small floor
    p = max(p, 1.0 / (n_perm + 1))
    return {
        "pvalue": p,
        "n_perm": n_perm,
        "obs_mean": obs,
        "status": "ok",
    }


def paired_ci_student(a: np.ndarray, b: np.ndarray, alpha: float, stats) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    mean = float(d.mean()) if n else None
    if n < 2:
        return {"mean": mean, "ci_low": None, "ci_high": None, "se": None, "df": None, "method": "student"}
    se = float(d.std(ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return {
        "mean": mean,
        "ci_low": mean - tcrit * se,
        "ci_high": mean + tcrit * se,
        "se": se,
        "df": n - 1,
        "method": "student",
        "alpha": alpha,
    }


def paired_ci_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    mean = float(d.mean()) if n else None
    if n == 0:
        return {"mean": mean, "ci_low": None, "ci_high": None, "method": "bootstrap"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    return {
        "mean": mean,
        "ci_low": float(np.quantile(boots, alpha / 2)),
        "ci_high": float(np.quantile(boots, 1 - alpha / 2)),
        "method": "bootstrap",
        "n_boot": n_boot,
        "alpha": alpha,
    }


def tost_paired(
    a: np.ndarray,
    b: np.ndarray,
    delta: float,
    alpha: float,
    stats,
) -> Dict[str, Any]:
    """
    TOST equivalence for mean(a-b) within (-delta, +delta).
    Reject H0_lower: mean <= -delta  (test mean > -delta)
    Reject H0_upper: mean >= +delta  (test mean < +delta)
    Equivalence iff both one-sided p <= alpha.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    mean = float(d.mean()) if n else None
    out: Dict[str, Any] = {
        "delta": float(delta),
        "alpha": float(alpha),
        "mean_diff": mean,
        "n": int(n),
        "equivalent": False,
        "status": "ok",
    }
    if n < 2 or mean is None:
        out["status"] = "insufficient_n"
        return out
    if np.allclose(d, 0.0) and delta > 0:
        out.update(
            {
                "t_lower": None,
                "p_lower": 0.0,
                "t_upper": None,
                "p_upper": 0.0,
                "equivalent": True,
                "status": "identical_within_margin",
            }
        )
        return out

    se = float(d.std(ddof=1) / math.sqrt(n))
    if se == 0.0:
        # all equal nonzero — still compare to margin
        inside = abs(mean) < delta
        out.update(
            {
                "t_lower": None,
                "p_lower": 0.0 if mean > -delta else 1.0,
                "t_upper": None,
                "p_upper": 0.0 if mean < delta else 1.0,
                "equivalent": bool(inside),
                "status": "zero_se",
            }
        )
        return out

    # lower: H0 mean_diff <= -delta; Ha mean_diff > -delta
    t_lo = (mean - (-delta)) / se
    p_lo = float(stats.t.sf(t_lo, df=n - 1))  # P(T > t_lo)
    # upper: H0 mean_diff >= +delta; Ha mean_diff < +delta
    t_hi = (mean - delta) / se
    p_hi = float(stats.t.cdf(t_hi, df=n - 1))  # P(T < t_hi)

    out.update(
        {
            "t_lower": float(t_lo),
            "p_lower": p_lo,
            "t_upper": float(t_hi),
            "p_upper": p_hi,
            "equivalent": bool(p_lo <= alpha and p_hi <= alpha),
            "se": se,
            "df": n - 1,
        }
    )
    return out


def _dataset_means_pair(
    per_dataset: List[Dict[str, Any]], cmp_name: str, metric: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    a_means, b_means, labels = [], [], []
    for ds in per_dataset:
        cell = ds["comparisons"][cmp_name][metric]
        if cell.get("status") == "missing" or cell.get("n", 0) == 0:
            continue
        labels.append(ds["dataset"])
        a_means.append(float(cell["a_mean"]))
        b_means.append(float(cell["b_mean"]))
    return np.asarray(a_means, dtype=float), np.asarray(b_means, dtype=float), labels


def enrich_cross(
    per_dataset: List[Dict[str, Any]],
    stats,
    alpha: float,
    n_perm: int,
    n_boot: int,
    tost_delta: float,
) -> Dict[str, Any]:
    cross: Dict[str, Any] = {}
    for left, right, name in COMPARISONS:
        cross[name] = {}
        for metric in METRICS:
            a, b, labels = _dataset_means_pair(per_dataset, name, metric)
            base = {
                "datasets": labels,
                "n": int(a.size),
                "a_mean": float(a.mean()) if a.size else None,
                "b_mean": float(b.mean()) if b.size else None,
                "mean_diff": float((a - b).mean()) if a.size else None,
            }
            if a.size == 0:
                cross[name][metric] = {**base, "status": "missing"}
                continue
            if np.allclose(a, b):
                cross[name][metric] = {
                    **base,
                    "status": "identical",
                    "t_pvalue": None,
                    "wilcoxon_pvalue": None,
                    "wilcoxon_stat": None,
                    "permutation_pvalue": 1.0,
                    "ci_student": paired_ci_student(a, b, alpha, stats),
                    "ci_bootstrap": paired_ci_bootstrap(a, b, alpha, n_boot, seed=_seed_key(name + metric)),
                    "cohens_dz": 0.0,
                }
                continue

            t_res = stats.ttest_rel(a, b)
            perm = paired_permutation_pvalue(a, b, n_perm=n_perm, seed=_seed_key(name + metric))
            ci_t = paired_ci_student(a, b, alpha, stats)
            ci_b = paired_ci_bootstrap(a, b, alpha, n_boot, seed=_seed_key(name + metric + "|boot"))
            diffs = a - b
            nz = diffs[np.abs(diffs) > 1e-15]
            w_stat: Optional[float] = None
            w_p: Optional[float] = None
            if nz.size >= 1:
                try:
                    w_res = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
                    w_stat = float(w_res.statistic)
                    w_p = float(w_res.pvalue)
                except ValueError:
                    pass
            cell: Dict[str, Any] = {
                **base,
                "status": "ok",
                "t_stat": float(t_res.statistic),
                "t_pvalue": float(t_res.pvalue),
                "wilcoxon_stat": w_stat,
                "wilcoxon_pvalue": w_p,
                "permutation": perm,
                "permutation_pvalue": perm.get("pvalue"),
                "ci_student": ci_t,
                "ci_bootstrap": ci_b,
                "cohens_dz": _cohen_dz(diffs),
            }
            if (name, metric) in TOST_TARGETS:
                cell["tost"] = tost_paired(a, b, delta=tost_delta, alpha=alpha, stats=stats)
            cross[name][metric] = cell
    return cross


def apply_holm(cross: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    raw = []
    keys = []
    for cmp_name, metric in HOLM_FAMILY:
        cell = cross.get(cmp_name, {}).get(metric, {})
        raw.append(cell.get("t_pvalue"))
        keys.append((cmp_name, metric))
    adjusted = holm_correction(raw, alpha=alpha)
    family = []
    for (cmp_name, metric), adj in zip(keys, adjusted):
        family.append(
            {
                "comparison": cmp_name,
                "metric": metric,
                **adj,
                "mean_diff": cross.get(cmp_name, {}).get(metric, {}).get("mean_diff"),
                "permutation_pvalue": cross.get(cmp_name, {}).get(metric, {}).get("permutation_pvalue"),
            }
        )
        # attach onto cell
        if cmp_name in cross and metric in cross[cmp_name]:
            cross[cmp_name][metric]["holm"] = adj
    return {"alpha": alpha, "family": family, "m": len(keys)}


def summarize_mcnemar(mcnemar_path: Path) -> Optional[Dict[str, Any]]:
    if not mcnemar_path.exists():
        return None
    blob = json.loads(mcnemar_path.read_text(encoding="utf-8"))
    rows = []
    for ds in blob.get("results", []):
        name = ds.get("dataset")
        s = ds.get("summary", {}).get("noise_0.2", {})
        for key in (
            "spc_vs_base_pcp",
            "spc_vs_base_accuracy",
            "spc_vs_soft_pcp",
            "spc_vs_soft_accuracy",
        ):
            block = s.get(key)
            if not block:
                continue
            rows.append(
                {
                    "dataset": name,
                    "test": key,
                    "median_exact_p": block.get("median_exact_p"),
                    "mean_delta": block.get("mean_delta"),
                    "mean_ci_low": block.get("mean_ci_low"),
                    "mean_ci_high": block.get("mean_ci_high"),
                }
            )
    return {"source": str(mcnemar_path), "noise_0.2": rows}


def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return "-"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt(x: Optional[float], d: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{d}f}"


def render_txt(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append("SPC stats suite (cross-dataset)")
    lines.append(
        f"alpha={payload['alpha']}  n_perm={payload['n_perm']}  "
        f"n_boot={payload['n_boot']}  tost_delta={payload['tost_delta']}"
    )
    lines.append("Pipeline: paired t + CI + Cohen dz + Wilcoxon -> permutation -> Holm(superiority) -> TOST(Acc)")
    lines.append("")

    lines.append("=== Holm family (noise PCP superiority) ===")
    lines.append(
        f"{'comparison':18s} {'metric':22s} {'mean_d':>8s} {'dz':>6s} {'p_t':>10s} "
        f"{'p_W':>10s} {'p_holm':>10s} {'rej':>4s}"
    )
    for row in payload["holm"]["family"]:
        rej = "Y" if row.get("reject") else "N"
        cell = payload["cross_dataset"][row["comparison"]][row["metric"]]
        lines.append(
            f"{row['comparison']:18s} {row['metric']:22s} {_fmt(row.get('mean_diff')):>8s} "
            f"{_fmt(cell.get('cohens_dz'), 2):>6s} "
            f"{_fmt_p(row.get('raw_p')):>10s} {_fmt_p(cell.get('wilcoxon_pvalue')):>10s} "
            f"{_fmt_p(row.get('holm_p')):>10s} {rej:>4s}"
        )

    lines.append("")
    lines.append("=== TOST accuracy preservation (SPC vs Base, |d| < delta) ===")
    for cmp_name, metric in TOST_TARGETS:
        cell = payload["cross_dataset"][cmp_name][metric]
        tost = cell.get("tost") or {}
        eq = "YES" if tost.get("equivalent") else "NO"
        ci = cell.get("ci_student") or {}
        lo = ci.get("ci_low", ci.get("low"))
        hi = ci.get("ci_high", ci.get("high"))
        lines.append(
            f"{metric:22s} mean_d={_fmt(cell.get('mean_diff'))}  "
            f"p_lo={_fmt_p(tost.get('p_lower'))} p_hi={_fmt_p(tost.get('p_upper'))}  "
            f"equiv={eq}  "
            f"CI_t=[{_fmt(lo)},{_fmt(hi)}]"
        )

    lines.append("")
    lines.append("=== Cross-dataset detail (spc_vs_base) ===")
    lines.append(
        f"{'metric':28s} {'mean_d':>8s} {'dz':>6s} {'p_t':>10s} {'p_W':>10s} "
        f"{'CI_t_lo':>8s} {'CI_t_hi':>8s}"
    )
    for metric in METRICS:
        cell = payload["cross_dataset"]["spc_vs_base"][metric]
        ci = cell.get("ci_student") or {}
        lo = ci.get("ci_low", ci.get("low"))
        hi = ci.get("ci_high", ci.get("high"))
        lines.append(
            f"{metric:28s} {_fmt(cell.get('mean_diff')):>8s} {_fmt(cell.get('cohens_dz'), 2):>6s} "
            f"{_fmt_p(cell.get('t_pvalue')):>10s} {_fmt_p(cell.get('wilcoxon_pvalue')):>10s} "
            f"{_fmt(lo):>8s} {_fmt(hi):>8s}"
        )

    mc = payload.get("mcnemar_summary")
    lines.append("")
    if mc:
        lines.append("=== Sample-level McNemar summary (noise_0.2; from run_spc_mcnemar.py) ===")
        for row in mc.get("noise_0.2", []):
            if not str(row["test"]).endswith("_pcp") and "accuracy" not in str(row["test"]):
                continue
            if "spc_vs_base" not in row["test"] and "spc_vs_soft" not in row["test"]:
                continue
            lines.append(
                f"{row['dataset']:16s} {row['test']:28s} "
                f"median_p={_fmt_p(row.get('median_exact_p'))}  "
                f"d={_fmt(row.get('mean_delta'))}  "
                f"CI~[{_fmt(row.get('mean_ci_low'))},{_fmt(row.get('mean_ci_high'))}]"
            )
    else:
        lines.append("=== Sample-level McNemar: not found; run scripts/run_spc_mcnemar.py ===")

    lines.append("")
    lines.append("Notes:")
    lines.append("- Holm applied only to predeclared PCP superiority family (see HOLM_FAMILY).")
    lines.append("- TOST claims Acc equivalence within +/- tost_delta (default 1pp).")
    lines.append("- Permutation: sign-flip of paired dataset-level differences.")
    lines.append("- McNemar is sample-level within each seed; reported as median p across seeds.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="SPC Holm + CI + permutation + TOST + McNemar summary")
    ap.add_argument("--outputs-root", type=str, default=str(ROOT / "outputs"))
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "spc_stats_suite"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument(
        "--tost_delta",
        type=float,
        default=0.01,
        help="TOST equivalence margin for Acc (absolute rate), default 0.01 = 1pp",
    )
    ap.add_argument(
        "--mcnemar-json",
        type=str,
        default=str(ROOT / "outputs" / "spc_mcnemar" / "spc_mcnemar.json"),
    )
    args = ap.parse_args()
    stats = _try_scipy()

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

    cross = enrich_cross(
        per_dataset,
        stats,
        alpha=float(args.alpha),
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        tost_delta=float(args.tost_delta),
    )
    holm = apply_holm(cross, alpha=float(args.alpha))
    mcnemar_summary = summarize_mcnemar(Path(args.mcnemar_json))

    payload = {
        "n_datasets": len(per_dataset),
        "datasets": [d["dataset"] for d in per_dataset],
        "alpha": float(args.alpha),
        "n_perm": int(args.n_perm),
        "n_boot": int(args.n_boot),
        "tost_delta": float(args.tost_delta),
        "holm_family_keys": [{"comparison": c, "metric": m} for c, m in HOLM_FAMILY],
        "tost_targets": [{"comparison": c, "metric": m} for c, m in TOST_TARGETS],
        "per_dataset": per_dataset,
        "cross_dataset": cross,
        "holm": holm,
        "mcnemar_summary": mcnemar_summary,
        "notes": [
            "Cross-dataset unit = per-log 3-seed mean.",
            "Holm on predeclared noise-PCP superiority tests only.",
            "TOST for Acc preservation vs Base within +/- tost_delta.",
            "Sample-level McNemar from separate script if available.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "spc_stats_suite.json", payload)
    txt = render_txt(payload)
    # ASCII-safe for Windows consoles
    txt_path = out_dir / "spc_stats_suite.txt"
    txt_path.write_text(txt, encoding="utf-8")
    try:
        print(txt, flush=True)
    except UnicodeEncodeError:
        print(txt.encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
