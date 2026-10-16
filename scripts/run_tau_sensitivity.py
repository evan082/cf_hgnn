#!/usr/bin/env python
"""
Gate threshold τ sensitivity: scan val-entropy percentiles.

Uses SoftLegal checkpoints (soft_seed*.pt); τ is inference-only.
Default: τ = val entropy p70. Scan p50..p90 (+ Soft / HardMask refs)
to show the Acc–PCP trade-off and that p70 is near the knee.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
]

DEFAULT_PERCENTILES = [0.50, 0.60, 0.70, 0.80, 0.90]


def _val_loader(datasets, legal_masks, num_activities, batch_size):
    ds = PrefixDataset(datasets["val"], legal_masks=legal_masks, num_activities=num_activities)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)


def _val_entropy_quantile(model, loader, device, q: float) -> float:
    m = eval_transformer(model, loader, device, use_legal_mask=False, gate_tau=None)
    # Recompute exact quantile from a dedicated pass for arbitrary q
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
        # fallback
        return float(m.get("entropy_p70", 1.0))
    return float(np.quantile(ents, q))


def _pick_checkpoint(out_dir: Path, seed: int) -> Path:
    soft = out_dir / f"soft_seed{seed}.pt"
    spc = out_dir / f"spc_seed{seed}.pt"
    if soft.exists():
        return soft
    if spc.exists():
        return spc
    raise FileNotFoundError(f"No soft/spc checkpoint for seed={seed} in {out_dir}")


def scan_one_dataset(
    cfg_path: str,
    seeds: List[int],
    percentiles: List[float],
    noise_rates: List[float],
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

    for rate in noise_rates:
        noise_path = out_dir / f"noise_{rate}" / "prefixes_test.pkl"
        if not noise_path.exists():
            _build_noise_prefixes_fast(cfg, out_dir, float(rate))

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        pt_path = _pick_checkpoint(out_dir, seed)
        print(f"[{cfg.get('dataset')}] seed={seed} ckpt={pt_path.name}", flush=True)
        model = _build_model(cfg, bundle, datasets, out_dir, device)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        model.eval()

        vloader = _val_loader(datasets, legal_masks, bundle.num_activities, bs)
        tau_map = {q: _val_entropy_quantile(model, vloader, device, q) for q in percentiles}

        # Reference points
        settings: List[tuple] = [
            ("soft", None, False),  # no gate
            ("hardmask", None, True),  # always mask
        ]
        for q in percentiles:
            settings.append((f"p{int(round(q * 100))}", float(tau_map[q]), False))

        test_samples = datasets["test"]
        for name, gate_tau, use_legal in settings:
            clean = eval_transformer_on_samples(
                model,
                test_samples,
                out_dir,
                bundle.num_activities,
                bs,
                device,
                use_legal_mask=use_legal,
                gate_tau=gate_tau,
            )
            row: Dict[str, Any] = {
                "seed": seed,
                "setting": name,
                "percentile": None
                if name in ("soft", "hardmask")
                else float(name[1:]) / 100.0,
                "gate_tau": None if use_legal else gate_tau,
                "clean_accuracy": clean.get("accuracy"),
                "clean_legal_rate": clean.get("legal_rate"),
                "clean_gate_fire_rate": clean.get("gate_fire_rate"),
                "ckpt": pt_path.name,
            }
            for rate in noise_rates:
                noise_path = out_dir / f"noise_{rate}" / "prefixes_test.pkl"
                with open(noise_path, "rb") as f:
                    noisy = pickle.load(f)
                nm = eval_transformer_on_samples(
                    model,
                    noisy,
                    out_dir,
                    bundle.num_activities,
                    bs,
                    device,
                    use_legal_mask=use_legal,
                    gate_tau=gate_tau,
                )
                row[f"noise_{rate}_accuracy"] = nm.get("accuracy")
                row[f"noise_{rate}_legal_rate"] = nm.get("legal_rate")
                row[f"noise_{rate}_gate_fire_rate"] = nm.get("gate_fire_rate")
            rows.append(row)
            print(
                f"  {name}: clean_acc={row['clean_accuracy']:.4f} "
                f"legal={row.get('clean_legal_rate')} "
                f"n0.2_legal={row.get('noise_0.2_legal_rate')}",
                flush=True,
            )

    # aggregate by setting across seeds
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

    keys = ["clean_accuracy", "clean_legal_rate", "clean_gate_fire_rate"]
    for rate in noise_rates:
        keys.extend(
            [
                f"noise_{rate}_accuracy",
                f"noise_{rate}_legal_rate",
                f"noise_{rate}_gate_fire_rate",
            ]
        )
    summary = {k: _agg(k) for k in keys}
    return {
        "dataset": cfg.get("dataset"),
        "config": cfg_path,
        "seeds": seeds,
        "percentiles": percentiles,
        "tau_by_seed_p70": {
            str(s): next(
                (
                    r["gate_tau"]
                    for r in rows
                    if r["seed"] == s and r["setting"] == "p70"
                ),
                None,
            )
            for s in seeds
        },
        "per_run": rows,
        "summary": summary,
    }


def _plot_dataset(ds: Dict[str, Any], out_png: Path, noise_rate: float = 0.2) -> None:
    import matplotlib.pyplot as plt

    summary = ds["summary"]
    percentiles = [p for p in ds["percentiles"]]
    labels = [f"p{int(round(p * 100))}" for p in percentiles]

    def means(metric: str) -> List[float]:
        return [float(summary[metric][lab]["mean"]) for lab in labels]

    acc_c = means("clean_accuracy")
    acc_n = means(f"noise_{noise_rate}_accuracy")
    leg_n = means(f"noise_{noise_rate}_legal_rate")
    soft_acc = summary["clean_accuracy"].get("soft", {}).get("mean")
    soft_leg = summary[f"noise_{noise_rate}_legal_rate"].get("soft", {}).get("mean")
    hard_acc = summary["clean_accuracy"].get("hardmask", {}).get("mean")
    hard_leg = summary[f"noise_{noise_rate}_legal_rate"].get("hardmask", {}).get("mean")

    xs = [p * 100 for p in percentiles]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(xs, acc_c, "o-", color="#1f4e79", label="clean Acc")
    ax.plot(xs, acc_n, "s--", color="#c45c26", label=f"noise{noise_rate} Acc")
    if soft_acc is not None:
        ax.axhline(soft_acc, color="#1f4e79", ls=":", lw=1, alpha=0.7, label="Soft (no gate)")
    if hard_acc is not None:
        ax.axhline(hard_acc, color="#666666", ls=":", lw=1, alpha=0.7, label="HardMask")
    ax.axvline(70, color="#2a9d8f", ls="--", lw=1.2, label="p70 (default)")
    ax.set_xlabel("val entropy percentile for τ")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{ds['dataset']}: Acc vs τ percentile")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, leg_n, "D-", color="#2a9d8f", label=f"noise{noise_rate} PCP")
    if soft_leg is not None:
        ax.axhline(soft_leg, color="#c45c26", ls=":", lw=1, alpha=0.8, label="Soft PCP")
    if hard_leg is not None:
        ax.axhline(hard_leg, color="#666666", ls=":", lw=1, alpha=0.8, label="HardMask PCP")
    ax.axvline(70, color="#2a9d8f", ls="--", lw=1.2, label="p70 (default)")
    ax.set_xlabel("val entropy percentile for τ")
    ax.set_ylabel("PCP")
    ax.set_title(f"{ds['dataset']}: PCP vs τ percentile")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_overview(all_ds: List[Dict[str, Any]], out_png: Path, noise_rate: float = 0.2) -> None:
    """Relative to Soft: ΔAcc / ΔPCP across datasets at each percentile."""
    import matplotlib.pyplot as plt

    percentiles = all_ds[0]["percentiles"]
    xs = [p * 100 for p in percentiles]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)

    for ds in all_ds:
        s = ds["summary"]
        soft_acc = s["clean_accuracy"]["soft"]["mean"]
        soft_leg = s[f"noise_{noise_rate}_legal_rate"]["soft"]["mean"]
        d_acc = []
        d_leg = []
        for p in percentiles:
            lab = f"p{int(round(p * 100))}"
            d_acc.append(s["clean_accuracy"][lab]["mean"] - soft_acc)
            d_leg.append(s[f"noise_{noise_rate}_legal_rate"][lab]["mean"] - soft_leg)
        axes[0].plot(xs, d_acc, "o-", label=ds["dataset"])
        axes[1].plot(xs, d_leg, "D-", label=ds["dataset"])

    for ax, title, ylab in zip(
        axes,
        [
            f"clean Acc − Soft vs τ percentile",
            f"noise{noise_rate} PCP − Soft vs τ percentile",
        ],
        ["Δ Acc", "Δ PCP"],
    ):
        ax.axvline(70, color="#2a9d8f", ls="--", lw=1.2, label="p70")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel("val entropy percentile for τ")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _score_table(ds: Dict[str, Any], noise_rate: float = 0.2) -> str:
    """
    Rank percentiles by noise PCP lift vs Soft, under a noise-Acc budget.
    Clean Acc is often flat (Soft already legal on clean); the real trade-off is noise.
    """
    s = ds["summary"]
    soft_acc = s["clean_accuracy"]["soft"]["mean"]
    soft_nacc = s[f"noise_{noise_rate}_accuracy"]["soft"]["mean"]
    soft_leg = s[f"noise_{noise_rate}_legal_rate"]["soft"]["mean"]
    lines = [
        f"dataset={ds['dataset']}",
        f"{'setting':8s} {'cleanAcc':>8s} {'nAcc':>8s} {'nPCP':>8s} "
        f"{'dNAcc':>8s} {'dPCP':>8s}",
    ]
    best_budget = None  # max dPCP with noise Acc drop <= 0.5pp
    best_balanced = None  # max dPCP + 0.5*dNAcc (simple scalarization)
    for p in ds["percentiles"]:
        lab = f"p{int(round(p * 100))}"
        ca = s["clean_accuracy"][lab]["mean"]
        na = s[f"noise_{noise_rate}_accuracy"][lab]["mean"]
        nl = s[f"noise_{noise_rate}_legal_rate"][lab]["mean"]
        d_nacc = na - soft_nacc
        d_leg = nl - soft_leg
        lines.append(
            f"{lab:8s} {ca:8.4f} {na:8.4f} {nl:8.4f} {d_nacc:+8.4f} {d_leg:+8.4f}"
        )
        if d_nacc >= -0.005:
            if best_budget is None or d_leg > best_budget[1]:
                best_budget = (lab, d_leg, d_nacc)
        score = d_leg + 0.5 * d_nacc
        if best_balanced is None or score > best_balanced[1]:
            best_balanced = (lab, score, d_leg, d_nacc)
    if best_budget:
        lines.append(
            f"best under noiseAcc drop<=0.5pp vs Soft: {best_budget[0]} "
            f"(dPCP={best_budget[1]:+.4f}, dNAcc={best_budget[2]:+.4f})"
        )
    if best_balanced:
        lines.append(
            f"best balanced (dPCP+0.5*dNAcc): {best_balanced[0]} "
            f"(score={best_balanced[1]:+.4f}); default=p70"
        )
    _ = soft_acc  # clean Acc usually flat; kept for table
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="SPC gate τ percentile sensitivity")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=DEFAULT_PERCENTILES,
    )
    ap.add_argument("--noise_rates", nargs="+", type=float, default=[0.1, 0.2])
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs" / "tau_sensitivity"),
    )
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    all_results = []
    text_blocks = []
    for cfg_path in args.configs:
        cfg_full = str(ROOT / cfg_path) if not Path(cfg_path).is_absolute() else cfg_path
        if not Path(cfg_full).exists():
            # allow relative to CWD
            cfg_full = cfg_path
        print(f"== scan {cfg_full} ==", flush=True)
        ds = scan_one_dataset(
            cfg_full,
            list(args.seeds),
            list(args.percentiles),
            list(args.noise_rates),
            device,
        )
        all_results.append(ds)
        name = ds["dataset"]
        _plot_dataset(ds, out_dir / f"tau_scan_{name}.png", noise_rate=0.2)
        text_blocks.append(_score_table(ds, noise_rate=0.2))
        save_json(out_dir / f"tau_scan_{name}.json", ds)

    if all_results:
        _plot_overview(all_results, out_dir / "tau_scan_overview.png", noise_rate=0.2)

    payload = {
        "percentiles": args.percentiles,
        "seeds": args.seeds,
        "noise_rates": args.noise_rates,
        "datasets": [d["dataset"] for d in all_results],
        "note": (
            "τ = val-set prediction-entropy quantile; SoftLegal weights; "
            "p70 is the paper default."
        ),
        "results": all_results,
    }
    save_json(out_dir / "tau_sensitivity.json", payload)
    txt = (
        "SPC gate τ sensitivity (val entropy percentiles)\n"
        "Default SPC uses p70. Soft = no gate; HardMask = always mask.\n\n"
        + "\n\n".join(text_blocks)
        + "\n"
    )
    (out_dir / "tau_sensitivity.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
