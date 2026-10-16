#!/usr/bin/env python
"""
τ–Pareto frontier: Accuracy vs PCP (LegalRate) trajectory.

Dense val-entropy percentiles (p10,p20,...,p90,p95) on SoftLegal weights
(inference-only gate). Anchors: Base (own ckpt) and HardMask (own ckpt).

Claim upgrade:
  SPC does not provide a single fixed compromise; it exposes a controllable
  accuracy–legality frontier between SoftLegal and HardMask.

Primary plot: x=noise Acc, y=noise PCP; connect
  Base → SoftLegal → SPC(p95…p10) → HardMask
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spc_study import (  # noqa: E402
    _build_model,
    _build_noise_prefixes_fast,
    eval_transformer_on_samples,
)
from scripts.utils import load_config, save_json  # noqa: E402
from src.data.prefixes import PrefixDataset, collate_pad  # noqa: E402
from src.train.trainer import eval_transformer  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/helpdesk_spc.yaml",
    "configs/traffic_spc.yaml",
    "configs/bpic2012_spc.yaml",
    "configs/sepsis_spc.yaml",
    "configs/bpic2019_spc.yaml",
]

# Dense grid: p10..p90 step 10, plus p95 (high-τ / rare gate near Soft)
DEFAULT_PERCENTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# Soft → Hard along frontier: high percentile (rare gate) → low (frequent gate)
TRAJECTORY_ORDER_P = sorted(DEFAULT_PERCENTILES, reverse=True)

COLORS = {
    "base": "#4C72B0",
    "soft": "#55A868",
    "spc": "#C44E52",
    "hardmask": "#8C8C8C",
    "p70": "#2A9D8F",
}


def _val_loader(datasets, legal_masks, num_activities, batch_size):
    ds = PrefixDataset(datasets["val"], legal_masks=legal_masks, num_activities=num_activities)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)


def _val_entropy_quantile(model, loader, device, q: float) -> float:
    from src.conformance.legal_next import prediction_entropy

    model.eval()
    ents: List[float] = []
    with torch.no_grad():
        for batch in loader:
            prefix = batch["prefix_ids"].to(device)
            lengths = batch["lengths"].to(device)
            key_pad = ~batch["mask"].to(device)
            time_feat = batch.get("time_feat")
            if time_feat is not None:
                time_feat = time_feat.to(device)
            resource_ids = batch.get("resource_ids")
            if resource_ids is not None:
                resource_ids = resource_ids.to(device)
            attr_ids = batch.get("attr_ids")
            if attr_ids is not None:
                attr_ids = attr_ids.to(device)
            logits = model(
                prefix,
                lengths,
                key_padding_mask=key_pad,
                time_feat=time_feat,
                resource_ids=resource_ids,
                attr_ids=attr_ids,
            )
            ents.extend(prediction_entropy(logits).cpu().tolist())
    if not ents:
        return 1.0
    return float(np.quantile(ents, q))


def _load_ckpt(out_dir: Path, tag: str, seed: int, cfg, bundle, datasets, device):
    pt = out_dir / f"{tag}_seed{seed}.pt"
    if not pt.exists():
        raise FileNotFoundError(pt)
    model = _build_model(cfg, bundle, datasets, out_dir, device)
    model.load_state_dict(torch.load(pt, map_location=device))
    model.eval()
    return model


def _lab(p: float) -> str:
    # p95 -> "p95"; p10 -> "p10"
    pct = int(round(p * 100))
    return f"p{pct}"


def scan_one_dataset(
    cfg_path: str,
    seeds: List[int],
    percentiles: List[float],
    noise_rate: float,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    bs = int(cfg.get("batch_size", 32))

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)
    with open(out_dir / "legal_next.pkl", "rb") as f:
        legal_masks = pickle.load(f)["masks"]

    noise_path = out_dir / f"noise_{noise_rate}" / "prefixes_test.pkl"
    if not noise_path.exists():
        _build_noise_prefixes_fast(cfg, out_dir, float(noise_rate))
    with open(noise_path, "rb") as f:
        noisy = pickle.load(f)
    test_samples = datasets["test"]

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"[{cfg.get('dataset')}] seed={seed}", flush=True)

        # Anchors from own checkpoints
        for tag, use_mask, gate in (
            ("base", False, None),
            ("soft", False, None),
            ("hardmask", True, None),
        ):
            try:
                model = _load_ckpt(out_dir, tag, seed, cfg, bundle, datasets, device)
            except FileNotFoundError as e:
                print(f"  skip {e}", flush=True)
                continue
            clean = eval_transformer_on_samples(
                model, test_samples, out_dir, bundle.num_activities, bs, device,
                use_legal_mask=use_mask, gate_tau=gate,
            )
            nm = eval_transformer_on_samples(
                model, noisy, out_dir, bundle.num_activities, bs, device,
                use_legal_mask=use_mask, gate_tau=gate,
            )
            row = {
                "seed": seed,
                "setting": tag,
                "percentile": None,
                "gate_tau": None,
                "clean_accuracy": clean.get("accuracy"),
                "clean_legal_rate": clean.get("legal_rate"),
                f"noise_{noise_rate}_accuracy": nm.get("accuracy"),
                f"noise_{noise_rate}_legal_rate": nm.get("legal_rate"),
                f"noise_{noise_rate}_gate_fire_rate": nm.get("gate_fire_rate"),
            }
            rows.append(row)
            print(
                f"  {tag}: nAcc={row[f'noise_{noise_rate}_accuracy']:.4f} "
                f"nPCP={row[f'noise_{noise_rate}_legal_rate']:.4f}",
                flush=True,
            )

        # SPC curve on SoftLegal weights
        try:
            soft_model = _load_ckpt(out_dir, "soft", seed, cfg, bundle, datasets, device)
        except FileNotFoundError:
            print("  skip SPC curve: no soft ckpt", flush=True)
            continue
        vloader = _val_loader(datasets, legal_masks, bundle.num_activities, bs)
        tau_map = {q: _val_entropy_quantile(soft_model, vloader, device, q) for q in percentiles}

        for q in percentiles:
            name = _lab(q)
            tau = float(tau_map[q])
            clean = eval_transformer_on_samples(
                soft_model, test_samples, out_dir, bundle.num_activities, bs, device,
                use_legal_mask=False, gate_tau=tau,
            )
            nm = eval_transformer_on_samples(
                soft_model, noisy, out_dir, bundle.num_activities, bs, device,
                use_legal_mask=False, gate_tau=tau,
            )
            row = {
                "seed": seed,
                "setting": name,
                "percentile": float(q),
                "gate_tau": tau,
                "clean_accuracy": clean.get("accuracy"),
                "clean_legal_rate": clean.get("legal_rate"),
                "clean_gate_fire_rate": clean.get("gate_fire_rate"),
                f"noise_{noise_rate}_accuracy": nm.get("accuracy"),
                f"noise_{noise_rate}_legal_rate": nm.get("legal_rate"),
                f"noise_{noise_rate}_gate_fire_rate": nm.get("gate_fire_rate"),
            }
            rows.append(row)

        # brief log endpoints
        p95 = next(r for r in rows if r["seed"] == seed and r["setting"] == _lab(max(percentiles)))
        p10 = next(r for r in rows if r["seed"] == seed and r["setting"] == _lab(min(percentiles)))
        print(
            f"  SPC {_lab(max(percentiles))}→{_lab(min(percentiles))}: "
            f"nPCP {p95[f'noise_{noise_rate}_legal_rate']:.4f}→"
            f"{p10[f'noise_{noise_rate}_legal_rate']:.4f}  "
            f"nAcc {p95[f'noise_{noise_rate}_accuracy']:.4f}→"
            f"{p10[f'noise_{noise_rate}_accuracy']:.4f}",
            flush=True,
        )

    by_set: Dict[str, List[Dict]] = {}
    for r in rows:
        by_set.setdefault(r["setting"], []).append(r)

    def _agg(key: str) -> Dict[str, Any]:
        out = {}
        for setting, rs in by_set.items():
            vals = [float(r[key]) for r in rs if r.get(key) is not None]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out[setting] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "values": vals,
            }
        return out

    keys = [
        "clean_accuracy",
        "clean_legal_rate",
        f"noise_{noise_rate}_accuracy",
        f"noise_{noise_rate}_legal_rate",
        f"noise_{noise_rate}_gate_fire_rate",
    ]
    summary = {k: _agg(k) for k in keys}
    return {
        "dataset": cfg.get("dataset"),
        "config": cfg_path,
        "seeds": seeds,
        "percentiles": percentiles,
        "noise_rate": noise_rate,
        "per_run": rows,
        "summary": summary,
    }


def _point(summary: Dict, setting: str, noise_rate: float) -> Optional[Tuple[float, float]]:
    acc = summary.get(f"noise_{noise_rate}_accuracy", {}).get(setting, {}).get("mean")
    pcp = summary.get(f"noise_{noise_rate}_legal_rate", {}).get(setting, {}).get("mean")
    if acc is None or pcp is None:
        return None
    return float(acc), float(pcp)


def _plot_dataset(ds: Dict[str, Any], out_png: Path) -> None:
    noise_rate = ds["noise_rate"]
    s = ds["summary"]
    percentiles = list(ds["percentiles"])
    # Soft → Hard: high p → low p
    order_p = sorted(percentiles, reverse=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)

    # Trajectory polyline: Soft → SPC(p95…p10) → HardMask
    traj_settings = ["soft"] + [_lab(p) for p in order_p] + ["hardmask"]
    traj_xy = []
    for name in traj_settings:
        pt = _point(s, name, noise_rate)
        if pt is not None:
            traj_xy.append((name, pt[0], pt[1]))

    if len(traj_xy) >= 2:
        xs = [t[1] for t in traj_xy]
        ys = [t[2] for t in traj_xy]
        ax.plot(xs, ys, "-", color=COLORS["spc"], lw=1.6, alpha=0.85, zorder=2)

    # Soft / HardMask anchors
    for name, color, marker, ms in (
        ("soft", COLORS["soft"], "o", 90),
        ("hardmask", COLORS["hardmask"], "s", 80),
    ):
        pt = _point(s, name, noise_rate)
        if pt:
            ax.scatter(
                [pt[0]], [pt[1]], c=color, marker=marker, s=ms,
                zorder=5, edgecolors="#222", linewidths=0.6,
                label={"soft": "SoftLegal", "hardmask": "HardMask"}[name],
            )

    # Base (may sit off Soft–Hard curve: different weights)
    pt_base = _point(s, "base", noise_rate)
    if pt_base:
        ax.scatter(
            [pt_base[0]], [pt_base[1]], c=COLORS["base"], marker="D", s=80,
            zorder=5, edgecolors="#222", linewidths=0.6, label="Base",
        )

    # SPC percentile markers
    for p in order_p:
        name = _lab(p)
        pt = _point(s, name, noise_rate)
        if not pt:
            continue
        is_p70 = abs(p - 0.70) < 1e-9
        ax.scatter(
            [pt[0]], [pt[1]],
            c=COLORS["p70"] if is_p70 else COLORS["spc"],
            marker="o",
            s=70 if is_p70 else 36,
            zorder=4,
            edgecolors="#222" if is_p70 else "none",
            linewidths=0.8 if is_p70 else 0,
            label="SPC p70" if is_p70 else None,
        )
        # annotate endpoints + p70
        if p in (order_p[0], order_p[-1]) or is_p70:
            ax.annotate(
                name,
                (pt[0], pt[1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="#333",
            )

    ax.set_xlabel(f"Accuracy (noise {noise_rate})")
    ax.set_ylabel(f"PCP / LegalRate (noise {noise_rate})")
    ax.set_title(f"{ds['dataset']}: τ–Pareto frontier")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")

    # Prefer north-east (high Acc, high PCP) readability: slight padding
    ax.margins(0.04)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(all_ds: List[Dict[str, Any]], out_png: Path) -> None:
    usable = [d for d in all_ds if d.get("summary")]
    if not usable:
        return
    n = len(usable)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 3.6 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, ds in zip(axes_flat, usable):
        noise_rate = ds["noise_rate"]
        s = ds["summary"]
        order_p = sorted(ds["percentiles"], reverse=True)
        traj = ["soft"] + [_lab(p) for p in order_p] + ["hardmask"]
        xy = []
        for name in traj:
            pt = _point(s, name, noise_rate)
            if pt:
                xy.append(pt)
        if len(xy) >= 2:
            ax.plot([p[0] for p in xy], [p[1] for p in xy], "-", color=COLORS["spc"], lw=1.5, alpha=0.9)

        for name, color, marker in (
            ("base", COLORS["base"], "D"),
            ("soft", COLORS["soft"], "o"),
            ("hardmask", COLORS["hardmask"], "s"),
        ):
            pt = _point(s, name, noise_rate)
            if pt:
                ax.scatter([pt[0]], [pt[1]], c=color, marker=marker, s=55, zorder=5, edgecolors="#222", linewidths=0.5)

        for p in order_p:
            pt = _point(s, _lab(p), noise_rate)
            if not pt:
                continue
            is_p70 = abs(p - 0.70) < 1e-9
            ax.scatter(
                [pt[0]], [pt[1]],
                c=COLORS["p70"] if is_p70 else COLORS["spc"],
                s=48 if is_p70 else 22,
                zorder=4,
                edgecolors="#222" if is_p70 else "none",
                linewidths=0.6 if is_p70 else 0,
            )

        ax.set_title(str(ds["dataset"]).replace("_spc", ""), fontsize=10)
        ax.set_xlabel("Acc")
        ax.set_ylabel("PCP")
        ax.grid(True, alpha=0.25)
        ax.margins(0.05)

    for ax in axes_flat[len(usable):]:
        ax.axis("off")

    # Shared legend on first axis
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["base"], markersize=7, label="Base"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["soft"], markersize=7, label="SoftLegal"),
        Line2D([0], [0], color=COLORS["spc"], lw=1.5, marker="o", markersize=4, label="SPC (p95…p10)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["p70"], markeredgecolor="#222", markersize=7, label="SPC p70"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["hardmask"], markersize=7, label="HardMask"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"τ–Pareto frontier (noise {usable[0]['noise_rate']}): Acc vs PCP",
        fontsize=12,
        y=1.06,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fmt_txt(all_ds: List[Dict[str, Any]]) -> str:
    lines = [
        "tau-Pareto frontier (noise Acc vs PCP)",
        "Trajectory: SoftLegal -> SPC(p95...p10) -> HardMask; Base as separate anchor.",
        "Lower percentile => lower tau => more gating => closer to HardMask.",
        "",
    ]
    for ds in all_ds:
        nr = ds["noise_rate"]
        s = ds["summary"]
        lines.append(f"=== {ds['dataset']} ===")
        lines.append(
            f"{'setting':10s} {'nAcc':>8s} {'nPCP':>8s} {'gate':>8s}"
        )
        order = ["base", "soft"] + [_lab(p) for p in sorted(ds["percentiles"], reverse=True)] + ["hardmask"]
        for name in order:
            acc = s.get(f"noise_{nr}_accuracy", {}).get(name, {}).get("mean")
            pcp = s.get(f"noise_{nr}_legal_rate", {}).get(name, {}).get("mean")
            gate = s.get(f"noise_{nr}_gate_fire_rate", {}).get(name, {}).get("mean")
            if acc is None:
                continue
            gtxt = f"{gate:8.3f}" if gate is not None else f"{'-':>8s}"
            lines.append(f"{name:10s} {acc:8.4f} {pcp:8.4f} {gtxt}")
        soft = _point(s, "soft", nr)
        hard = _point(s, "hardmask", nr)
        p70 = _point(s, "p70", nr)
        if soft and hard and p70:
            # span along PCP
            lines.append(
                f"  span Soft->Hard nPCP: {soft[1]:.4f} -> {hard[1]:.4f} "
                f"(d={hard[1]-soft[1]:+.4f}); p70 at Acc={p70[0]:.4f} PCP={p70[1]:.4f}"
            )
        lines.append("")
    lines.append(
        "Claim: SPC exposes a controllable Acc-PCP frontier; p70 is one operating point, not the only story."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="SPC tau-Pareto Acc vs PCP frontier")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--percentiles", nargs="+", type=float, default=DEFAULT_PERCENTILES)
    ap.add_argument("--noise_rate", type=float, default=0.2)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs" / "tau_pareto"),
    )
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    all_results: List[Dict[str, Any]] = []
    for cfg_path in args.configs:
        cfg_full = str(ROOT / cfg_path) if not Path(cfg_path).is_absolute() else cfg_path
        if not Path(cfg_full).exists():
            cfg_full = cfg_path
        print(f"== tau-pareto {cfg_full} ==", flush=True)
        try:
            ds = scan_one_dataset(
                cfg_full,
                list(args.seeds),
                list(args.percentiles),
                float(args.noise_rate),
                device,
            )
        except FileNotFoundError as e:
            print(f"skip: {e}", flush=True)
            continue
        all_results.append(ds)
        save_json(out_dir / f"pareto_{ds['dataset']}.json", ds)
        _plot_dataset(ds, out_dir / f"pareto_{ds['dataset']}.png")

    if all_results:
        _plot_overview(all_results, out_dir / "pareto_overview.png")
        fig_dir = ROOT / "docs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "pareto_overview.png", fig_dir / "fig_tau_pareto.png")

    payload = {
        "percentiles": args.percentiles,
        "seeds": args.seeds,
        "noise_rate": args.noise_rate,
        "datasets": [d["dataset"] for d in all_results],
        "claim": (
            "SPC does not provide a single fixed compromise; "
            "it exposes a controllable accuracy-legality frontier."
        ),
        "results": all_results,
    }
    save_json(out_dir / "tau_pareto.json", payload)
    txt = _fmt_txt(all_results)
    (out_dir / "tau_pareto.txt").write_text(txt, encoding="utf-8")
    try:
        print(txt, flush=True)
    except UnicodeEncodeError:
        print(txt.encode("ascii", errors="replace").decode("ascii"), flush=True)
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
