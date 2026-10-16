#!/usr/bin/env python
"""
Entropy-decile mechanism experiment for SPC.

Hypothesis: SPC's compliance gain concentrates in high-entropy prefixes
(where the gate fires), while low-entropy bins stay close to Soft/Base.

Protocol:
  - SoftLegal unconstrained entropy H(z) defines decile bins on each split
    (Soft and SPC share weights; H is pre-gate).
  - Within each decile, report Acc / PCP / illegal rate for
    Base / Soft / HardMask / SPC, plus SPC gate fire rate.
  - Aggregate across seeds (mean ± std); plot Acc & illegal-rate curves.

Default: noise0.2 on Helpdesk / Traffic / BPIC2012 / Sepsis / BPI2019.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spc_mcnemar import (  # noqa: E402
    _eval_variant_on_split,
    _legal_masks_for_samples,
    _load_variant,
)
from scripts.run_spc_study import _build_noise_prefixes_fast  # noqa: E402
from scripts.utils import load_config, save_json  # noqa: E402
from src.conformance.legal_next import prediction_entropy  # noqa: E402
from src.data.prefixes import PrefixDataset, collate_pad  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/helpdesk_spc.yaml",
    "configs/traffic_spc.yaml",
    "configs/bpic2012_spc.yaml",
    "configs/sepsis_spc.yaml",
    "configs/bpic2019_spc.yaml",
]

VARIANTS = ("base", "soft", "hardmask", "spc")
COLORS = {
    "base": "#4C72B0",
    "soft": "#55A868",
    "hardmask": "#8C8C8C",
    "spc": "#C44E52",
}
LABELS = {
    "base": "Base",
    "soft": "SoftLegal",
    "hardmask": "HardMask",
    "spc": "SPC",
}


@torch.no_grad()
def _soft_entropies(
    model,
    samples,
    legal_masks,
    num_activities: int,
    batch_size: int,
    device,
) -> np.ndarray:
    """Unconstrained Soft entropy H(z) per prefix (pre-gate)."""
    ds = PrefixDataset(samples, legal_masks=legal_masks, num_activities=num_activities)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    model.eval()
    ents: List[float] = []
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
    return np.asarray(ents, dtype=np.float64)


def _decile_edges(entropy: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(entropy, qs)
    # Ensure strictly increasing edges for digitize (duplicates → nudge)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def _assign_deciles(entropy: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # bins 0..n-1; rightmost closed
    idx = np.digitize(entropy, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)


def _bin_stats(
    outcomes: Dict[str, Dict[str, np.ndarray]],
    soft_ent: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    gate_tau: Optional[float],
) -> List[Dict[str, Any]]:
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        row: Dict[str, Any] = {
            "decile": b + 1,
            "n": n,
            "entropy_mean": float(soft_ent[mask].mean()) if n else None,
            "entropy_lo": float(soft_ent[mask].min()) if n else None,
            "entropy_hi": float(soft_ent[mask].max()) if n else None,
            "variants": {},
        }
        if gate_tau is not None and n:
            row["spc_gate_fire_rate"] = float((soft_ent[mask] >= float(gate_tau)).mean())
        else:
            row["spc_gate_fire_rate"] = None
        for var in VARIANTS:
            if var not in outcomes:
                continue
            o = outcomes[var]
            acc = float(o["correct"][mask].mean()) if n else None
            pcp = float(o["legal"][mask].mean()) if n else None
            row["variants"][var] = {
                "accuracy": acc,
                "legal_rate": pcp,
                "illegal_rate": (1.0 - pcp) if pcp is not None else None,
            }
        if "spc" in outcomes and "base" in outcomes and n:
            row["delta_spc_minus_base"] = {
                "accuracy": float(
                    outcomes["spc"]["correct"][mask].mean()
                    - outcomes["base"]["correct"][mask].mean()
                ),
                "legal_rate": float(
                    outcomes["spc"]["legal"][mask].mean()
                    - outcomes["base"]["legal"][mask].mean()
                ),
                "illegal_rate": float(
                    (1.0 - outcomes["spc"]["legal"][mask].mean())
                    - (1.0 - outcomes["base"]["legal"][mask].mean())
                ),
            }
        rows.append(row)
    return rows


def run_dataset(
    cfg_path: str,
    seeds: List[int],
    noise_rates: List[float],
    n_bins: int,
    device: torch.device,
    include_clean: bool,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    bs = int(cfg.get("batch_size", 32))

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    splits: Dict[str, Any] = {}
    if include_clean:
        splits["clean"] = datasets["test"]
    for rate in noise_rates:
        noise_path = out_dir / f"noise_{rate}" / "prefixes_test.pkl"
        if not noise_path.exists():
            _build_noise_prefixes_fast(cfg, out_dir, float(rate))
        with open(noise_path, "rb") as f:
            splits[f"noise_{rate}"] = pickle.load(f)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"[{cfg.get('dataset')}] seed={seed}", flush=True)
        loaded = {}
        for var in VARIANTS:
            try:
                model, _, ekw = _load_variant(out_dir, var, seed, cfg, bundle, datasets, device)
                loaded[var] = (model, ekw)
            except FileNotFoundError as e:
                print(f"  skip missing {e}", flush=True)

        if "soft" not in loaded:
            print("  skip seed: no Soft checkpoint", flush=True)
            continue

        soft_model, soft_ekw = loaded["soft"]
        seed_row: Dict[str, Any] = {"seed": seed, "splits": {}}
        gate_tau = None
        if "spc" in loaded:
            gate_tau = loaded["spc"][1].get("gate_tau")

        for split_name, samples in splits.items():
            masks = _legal_masks_for_samples(samples, out_dir, bundle.num_activities)
            soft_ent = _soft_entropies(
                soft_model, samples, masks, bundle.num_activities, bs, device
            )
            edges = _decile_edges(soft_ent, n_bins)
            bins = _assign_deciles(soft_ent, edges)

            outcomes = {}
            for var, (model, ekw) in loaded.items():
                outcomes[var] = _eval_variant_on_split(
                    model, ekw, samples, out_dir, bundle.num_activities, bs, device
                )

            bin_rows = _bin_stats(outcomes, soft_ent, bins, n_bins, gate_tau)
            overall = {
                var: {
                    "accuracy": float(outcomes[var]["correct"].mean()),
                    "legal_rate": float(outcomes[var]["legal"].mean()),
                }
                for var in outcomes
            }
            seed_row["splits"][split_name] = {
                "n": int(len(soft_ent)),
                "entropy_edges": edges.tolist(),
                "gate_tau": float(gate_tau) if gate_tau is not None else None,
                "overall": overall,
                "deciles": bin_rows,
            }
            # quick log: top vs bottom decile illegal Δ
            d1 = bin_rows[0]["variants"].get("base", {}).get("illegal_rate")
            d10 = bin_rows[-1]["variants"].get("base", {}).get("illegal_rate")
            g1 = bin_rows[0].get("spc_gate_fire_rate")
            g10 = bin_rows[-1].get("spc_gate_fire_rate")
            print(
                f"  {split_name}: n={len(soft_ent)}  "
                f"Base illegal d1={d1} d{n_bins}={d10}  "
                f"SPC gate d1={g1} d{n_bins}={g10}",
                flush=True,
            )

        per_seed.append(seed_row)

    summary = _aggregate(per_seed, n_bins)
    return {
        "dataset": cfg.get("dataset"),
        "config": cfg_path,
        "n_bins": n_bins,
        "seeds": seeds,
        "per_seed": per_seed,
        "summary": summary,
    }


def _aggregate(per_seed: List[Dict[str, Any]], n_bins: int) -> Dict[str, Any]:
    if not per_seed:
        return {}
    split_names = sorted({s for r in per_seed for s in r["splits"]})
    out: Dict[str, Any] = {}
    for split in split_names:
        blocks = [r["splits"][split] for r in per_seed if split in r["splits"]]
        if not blocks:
            continue
        dec_agg = []
        for b in range(n_bins):
            cell: Dict[str, Any] = {"decile": b + 1}
            ns = [blk["deciles"][b]["n"] for blk in blocks]
            cell["n_mean"] = float(np.mean(ns))
            ents = [
                blk["deciles"][b]["entropy_mean"]
                for blk in blocks
                if blk["deciles"][b]["entropy_mean"] is not None
            ]
            cell["entropy_mean"] = float(np.mean(ents)) if ents else None
            fires = [
                blk["deciles"][b]["spc_gate_fire_rate"]
                for blk in blocks
                if blk["deciles"][b].get("spc_gate_fire_rate") is not None
            ]
            if fires:
                cell["spc_gate_fire_rate_mean"] = float(np.mean(fires))
                cell["spc_gate_fire_rate_std"] = float(np.std(fires, ddof=1)) if len(fires) > 1 else 0.0
            else:
                cell["spc_gate_fire_rate_mean"] = None
                cell["spc_gate_fire_rate_std"] = None

            cell["variants"] = {}
            for var in VARIANTS:
                accs, pcps, ills = [], [], []
                for blk in blocks:
                    v = blk["deciles"][b]["variants"].get(var)
                    if not v or v["accuracy"] is None:
                        continue
                    accs.append(v["accuracy"])
                    pcps.append(v["legal_rate"])
                    ills.append(v["illegal_rate"])
                if not accs:
                    continue
                cell["variants"][var] = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                    "legal_rate_mean": float(np.mean(pcps)),
                    "legal_rate_std": float(np.std(pcps, ddof=1)) if len(pcps) > 1 else 0.0,
                    "illegal_rate_mean": float(np.mean(ills)),
                    "illegal_rate_std": float(np.std(ills, ddof=1)) if len(ills) > 1 else 0.0,
                }

            d_acc, d_pcp, d_ill = [], [], []
            for blk in blocks:
                d = blk["deciles"][b].get("delta_spc_minus_base")
                if not d:
                    continue
                d_acc.append(d["accuracy"])
                d_pcp.append(d["legal_rate"])
                d_ill.append(d["illegal_rate"])
            if d_acc:
                cell["delta_spc_minus_base"] = {
                    "accuracy_mean": float(np.mean(d_acc)),
                    "legal_rate_mean": float(np.mean(d_pcp)),
                    "illegal_rate_mean": float(np.mean(d_ill)),
                }
            dec_agg.append(cell)
        out[split] = {"deciles": dec_agg, "n_seeds": len(blocks)}
    return out


def _plot_dataset(ds: Dict[str, Any], out_path: Path, split: str = "noise_0.2") -> None:
    summary = ds.get("summary", {}).get(split)
    if not summary:
        return
    decs = summary["deciles"]
    xs = [d["decile"] for d in decs]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    # Acc
    ax = axes[0]
    for var in VARIANTS:
        ys = [d["variants"].get(var, {}).get("accuracy_mean") for d in decs]
        if any(y is not None for y in ys):
            ax.plot(xs, ys, marker="o", lw=1.8, color=COLORS[var], label=LABELS[var])
    ax.set_xlabel("Entropy decile (Soft H)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{ds['dataset']} · Acc")
    ax.set_xticks(xs)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)

    # Illegal rate
    ax = axes[1]
    for var in VARIANTS:
        ys = [d["variants"].get(var, {}).get("illegal_rate_mean") for d in decs]
        if any(y is not None for y in ys):
            ax.plot(xs, ys, marker="o", lw=1.8, color=COLORS[var], label=LABELS[var])
    ax.set_xlabel("Entropy decile (Soft H)")
    ax.set_ylabel("Illegal rate (1−PCP)")
    ax.set_title("Illegal rate")
    ax.set_xticks(xs)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)

    # Gate fire + Δ illegal
    ax = axes[2]
    fires = [d.get("spc_gate_fire_rate_mean") for d in decs]
    ax.bar(xs, [f if f is not None else 0 for f in fires], color="#E8A87C", alpha=0.85, label="SPC gate fire")
    ax.set_ylabel("Gate fire rate")
    ax.set_xlabel("Entropy decile (Soft H)")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    # delta illegal = SPC_ill - Base_ill; negate so ↑ means compliance gain
    d_ill = [
        -(d.get("delta_spc_minus_base") or {}).get("illegal_rate_mean", 0.0) for d in decs
    ]
    ax2.plot(xs, d_ill, marker="s", color="#C44E52", lw=1.8, label="Δ illegal↓ (SPC−Base)")
    ax2.set_ylabel("Illegal-rate drop (SPC vs Base)")
    ax.set_title("Gate + PCP gain")
    ax.grid(True, alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")

    fig.suptitle(
        f"Entropy-decile mechanism · {split.replace('_', ' ')}",
        fontsize=11,
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(all_ds: List[Dict[str, Any]], out_path: Path, split: str = "noise_0.2") -> None:
    usable = [d for d in all_ds if d.get("summary", {}).get(split)]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, ds in zip(axes, usable):
        decs = ds["summary"][split]["deciles"]
        xs = [d["decile"] for d in decs]
        for var in ("base", "spc", "hardmask"):
            ys = [d["variants"].get(var, {}).get("illegal_rate_mean") for d in decs]
            ax.plot(xs, ys, marker="o", lw=1.6, color=COLORS[var], label=LABELS[var])
        ax.set_title(ds["dataset"].replace("_spc", ""), fontsize=9)
        ax.set_xlabel("Entropy decile")
        ax.set_xticks(xs[::2] if len(xs) > 6 else xs)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Illegal rate (1−PCP)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Illegal rate by Soft-entropy decile · {split}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fmt_txt(all_ds: List[Dict[str, Any]], split: str = "noise_0.2") -> str:
    lines = [
        "Entropy-decile mechanism experiment (Soft H bins)",
        f"Primary split: {split}",
        "Bins by SoftLegal unconstrained entropy; SPC gate fire uses Soft H vs tau.",
        "",
    ]
    for ds in all_ds:
        summary = ds.get("summary", {}).get(split)
        if not summary:
            continue
        lines.append(f"=== {ds['dataset']} (n_seeds={summary['n_seeds']}) ===")
        lines.append(
            f"{'dec':>3} {'H':>7} {'gate':>6} "
            f"{'Base_ill':>9} {'SPC_ill':>9} {'Hard_ill':>9} "
            f"{'dAcc':>8} {'dPCP':>8}"
        )
        for d in summary["deciles"]:
            bv = d["variants"].get("base", {})
            sv = d["variants"].get("spc", {})
            hv = d["variants"].get("hardmask", {})
            delta = d.get("delta_spc_minus_base") or {}
            lines.append(
                f"{d['decile']:3d} "
                f"{(d.get('entropy_mean') or 0):7.3f} "
                f"{(d.get('spc_gate_fire_rate_mean') or 0):6.3f} "
                f"{(bv.get('illegal_rate_mean') or 0):9.4f} "
                f"{(sv.get('illegal_rate_mean') or 0):9.4f} "
                f"{(hv.get('illegal_rate_mean') or 0):9.4f} "
                f"{(delta.get('accuracy_mean') or 0):+8.4f} "
                f"{(delta.get('legal_rate_mean') or 0):+8.4f}"
            )
        # bottom vs top contrast
        lo, hi = summary["deciles"][0], summary["deciles"][-1]
        lines.append(
            f"  contrast d1->d{hi['decile']}: "
            f"Base_ill {lo['variants'].get('base', {}).get('illegal_rate_mean', 0):.4f}"
            f"->{hi['variants'].get('base', {}).get('illegal_rate_mean', 0):.4f}; "
            f"gate {lo.get('spc_gate_fire_rate_mean', 0):.2f}"
            f"->{hi.get('spc_gate_fire_rate_mean', 0):.2f}; "
            f"dPCP(SPC-Base) "
            f"{(lo.get('delta_spc_minus_base') or {}).get('legal_rate_mean', 0):+.4f}"
            f"->{(hi.get('delta_spc_minus_base') or {}).get('legal_rate_mean', 0):+.4f}"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="SPC entropy-decile mechanism analysis")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--noise_rates", nargs="+", type=float, default=[0.2])
    ap.add_argument("--n_bins", type=int, default=10, help="Deciles by default")
    ap.add_argument("--include_clean", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs" / "entropy_decile"),
    )
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    primary_split = f"noise_{args.noise_rates[0]}" if args.noise_rates else "clean"

    all_results: List[Dict[str, Any]] = []
    for cfg_path in args.configs:
        cfg_full = str(ROOT / cfg_path) if not Path(cfg_path).is_absolute() else cfg_path
        if not Path(cfg_full).exists():
            cfg_full = cfg_path
        print(f"== entropy decile {cfg_full} ==", flush=True)
        try:
            ds = run_dataset(
                cfg_full,
                list(args.seeds),
                list(args.noise_rates),
                int(args.n_bins),
                device,
                bool(args.include_clean),
            )
        except FileNotFoundError as e:
            print(f"skip: {e}", flush=True)
            continue
        all_results.append(ds)
        save_json(out_dir / f"decile_{ds['dataset']}.json", ds)
        _plot_dataset(ds, out_dir / f"decile_{ds['dataset']}.png", split=primary_split)

    _plot_overview(all_results, out_dir / "decile_overview.png", split=primary_split)
    # paper-ready copy
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    overview = out_dir / "decile_overview.png"
    if overview.exists():
        import shutil

        shutil.copy2(overview, fig_dir / "fig_entropy_decile.png")

    payload = {
        "seeds": args.seeds,
        "noise_rates": args.noise_rates,
        "n_bins": args.n_bins,
        "primary_split": primary_split,
        "datasets": [d["dataset"] for d in all_results],
        "results": all_results,
    }
    save_json(out_dir / "entropy_decile.json", payload)
    txt = _fmt_txt(all_results, split=primary_split)
    (out_dir / "entropy_decile.txt").write_text(txt, encoding="utf-8")
    try:
        print(txt, flush=True)
    except UnicodeEncodeError:
        print(txt.encode("utf-8", errors="replace").decode("ascii", errors="replace"), flush=True)
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
