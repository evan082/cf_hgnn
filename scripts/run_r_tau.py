# -*- coding: utf-8 -*-
"""Compute risk function R(τ)=α(1-PCP)+β(1-Acc)+γ·Coverage from tau_pareto outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "outputs" / "tau_pareto"
OUT_DIR = ROOT / "outputs" / "r_tau"
FIG_DIR = ROOT / "docs" / "figures"


def _mean_rows(per_run, setting_prefix="p"):
    """Aggregate seed rows by percentile setting (p10..p95)."""
    buckets = {}
    for r in per_run:
        s = r.get("setting")
        if not s or not str(s).startswith(setting_prefix):
            continue
        if r.get("noise_0.2_gate_fire_rate") is None:
            continue
        buckets.setdefault(s, []).append(r)
    out = []
    for s, rows in sorted(buckets.items(), key=lambda kv: float(kv[0][1:]) / 100.0):
        acc = np.mean([x["noise_0.2_accuracy"] for x in rows])
        pcp = np.mean([x["noise_0.2_legal_rate"] for x in rows])
        cov = np.mean([x["noise_0.2_gate_fire_rate"] for x in rows])
        pct = rows[0].get("percentile")
        out.append(
            {
                "setting": s,
                "percentile": pct,
                "acc": float(acc),
                "pcp": float(pcp),
                "coverage": float(cov),
            }
        )
    return out


def _anchors(per_run):
    """Soft / HardMask anchors (Coverage 0 / 1 by definition for R)."""
    soft, hard = [], []
    for r in per_run:
        if r.get("setting") == "soft":
            soft.append(r)
        elif r.get("setting") == "hardmask":
            hard.append(r)
    def agg(rows, cov):
        if not rows:
            return None
        return {
            "acc": float(np.mean([x["noise_0.2_accuracy"] for x in rows])),
            "pcp": float(np.mean([x["noise_0.2_legal_rate"] for x in rows])),
            "coverage": float(cov),
        }
    return agg(soft, 0.0), agg(hard, 1.0)


def risk(acc, pcp, cov, alpha, beta, gamma):
    return alpha * (1.0 - pcp) + beta * (1.0 - acc) + gamma * cov


def process_file(path: Path, weights):
    data = json.loads(path.read_text(encoding="utf-8"))
    curve = _mean_rows(data["per_run"])
    soft, hard = _anchors(data["per_run"])
    results = {"dataset": data.get("dataset", path.stem), "curve": curve, "weights": {}}
    for name, (a, b, g) in weights.items():
        scored = []
        for row in curve:
            r = risk(row["acc"], row["pcp"], row["coverage"], a, b, g)
            scored.append({**row, "R": float(r)})
        best = min(scored, key=lambda x: x["R"]) if scored else None
        soft_r = risk(soft["acc"], soft["pcp"], soft["coverage"], a, b, g) if soft else None
        hard_r = risk(hard["acc"], hard["pcp"], hard["coverage"], a, b, g) if hard else None
        p70 = next((x for x in scored if x["setting"] == "p70"), None)
        results["weights"][name] = {
            "alpha": a,
            "beta": b,
            "gamma": g,
            "scored": scored,
            "best": best,
            "soft_R": soft_r,
            "hard_R": hard_r,
            "p70_R": p70["R"] if p70 else None,
        }
    return results, soft, hard


def plot_dataset(res, soft, hard, out_png: Path, weight_key="balanced"):
    from matplotlib import rcParams

    rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    w = res["weights"][weight_key]
    xs = [x["percentile"] for x in w["scored"]]
    ys = [x["R"] for x in w["scored"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=160)
    ax.plot(xs, ys, "o-", color="#E45756", label=f"SPC R(τ) [{weight_key}]")
    if soft and w["soft_R"] is not None:
        ax.axhline(w["soft_R"], color="#72B7B2", ls="--", label=f"SoftLegal R={w['soft_R']:.4f}")
    if hard and w["hard_R"] is not None:
        ax.axhline(w["hard_R"], color="#F58518", ls="--", label=f"HardMask R={w['hard_R']:.4f}")
    if w["best"]:
        ax.scatter([w["best"]["percentile"]], [w["best"]["R"]], s=120, c="#4C78A8", zorder=5, label=f"argmin {w['best']['setting']}")
    ax.set_xlabel("验证熵分位 (τ)")
    ax.set_ylabel("R(τ)")
    ax.set_title(f"{res['dataset']}: 风险函数随 τ")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto_dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    weights = {
        "balanced": (1.0, 1.0, 0.5),
        "compliance_first": (2.0, 0.5, 0.5),
        "accuracy_first": (0.5, 2.0, 0.5),
        "low_intervention": (1.0, 1.0, 2.0),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_res = []
    summary_rows = []
    for path in sorted(args.pareto_dir.glob("pareto_*_spc.json")):
        res, soft, hard = process_file(path, weights)
        all_res.append(res)
        plot_dataset(res, soft, hard, FIG_DIR / f"fig_r_tau_{res['dataset'].replace('_spc','')}.png")
        plot_dataset(res, soft, hard, args.out_dir / f"r_tau_{res['dataset']}.png")
        b = res["weights"]["balanced"]["best"]
        summary_rows.append(
            {
                "dataset": res["dataset"],
                "best_setting": b["setting"] if b else None,
                "best_R": b["R"] if b else None,
                "best_acc": b["acc"] if b else None,
                "best_pcp": b["pcp"] if b else None,
                "best_coverage": b["coverage"] if b else None,
                "p70_R": res["weights"]["balanced"]["p70_R"],
                "soft_R": res["weights"]["balanced"]["soft_R"],
                "hard_R": res["weights"]["balanced"]["hard_R"],
            }
        )
        print(res["dataset"], "best", b["setting"] if b else None, "R", b["R"] if b else None)

    # aggregate plot: mean R curve across datasets for balanced
    from matplotlib import rcParams

    rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    # align by setting
    settings = sorted(
        {row["setting"] for r in all_res for row in r["weights"]["balanced"]["scored"]},
        key=lambda s: float(s[1:]) / 100.0,
    )
    mean_r, mean_cov = [], []
    for s in settings:
        vals = []
        covs = []
        for r in all_res:
            hit = next((x for x in r["weights"]["balanced"]["scored"] if x["setting"] == s), None)
            if hit:
                vals.append(hit["R"])
                covs.append(hit["coverage"])
        mean_r.append(float(np.mean(vals)) if vals else np.nan)
        mean_cov.append(float(np.mean(covs)) if covs else np.nan)
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8), dpi=170)
    ax1.plot([float(s[1:]) / 100 for s in settings], mean_r, "o-", color="#E45756", label="跨日志平均 R(τ)")
    ax1.set_xlabel("验证熵分位 τ")
    ax1.set_ylabel("平均 R(τ)  (α=β=1, γ=0.5)")
    ax2 = ax1.twinx()
    ax2.plot([float(s[1:]) / 100 for s in settings], mean_cov, "s--", color="#4C78A8", alpha=0.7, label="平均 Coverage")
    ax2.set_ylabel("平均 Coverage")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.set_title("风险函数实验：跨日志平均 R(τ) 与 Coverage")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_r_tau.png", bbox_inches="tight")
    fig.savefig(args.out_dir / "fig_r_tau.png", bbox_inches="tight")
    plt.close(fig)

    payload = {"weights": weights, "per_dataset": all_res, "summary_balanced": summary_rows}
    (args.out_dir / "r_tau.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # compact table for paper
    lines = ["dataset\tbest\tR\tAcc\tPCP\tCov\tp70_R\tsoft_R\thard_R"]
    for r in summary_rows:
        lines.append(
            f"{r['dataset']}\t{r['best_setting']}\t{r['best_R']:.4f}\t{r['best_acc']:.3f}\t"
            f"{r['best_pcp']:.3f}\t{r['best_coverage']:.3f}\t{r['p70_R']:.4f}\t{r['soft_R']:.4f}\t{r['hard_R']:.4f}"
        )
    (args.out_dir / "r_tau_summary.tsv").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", args.out_dir / "r_tau.json")
    print("Wrote", FIG_DIR / "fig_r_tau.png")


if __name__ == "__main__":
    main()
