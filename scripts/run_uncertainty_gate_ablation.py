#!/usr/bin/env python
"""
Uncertainty-proxy ablation for SPC gating (inference-only).

Proxies (higher = more uncertain):
  entropy: H(p)
  conf:    1 - max_a p_a
  margin:  1 - (p_(1) - p_(2))

Two matched protocols (SoftLegal weights):
  1) Matched gate fire-rate: tau = quantile(U_val, 1 - target_fire)
  2) Acc-budget on val noise0.2: among fire rates, max PCP s.t. Acc drop vs Soft <= budget

Primary claim options:
  - entropy wins on PCP at matched fire / budget  -> support default H
  - three proxies similar                         -> SPC insensitive to proxy
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from scripts.tau_gate import build_noise_prefixes_for_split  # noqa: E402
from scripts.utils import load_config, save_json  # noqa: E402
from src.conformance.legal_next import prediction_uncertainty  # noqa: E402
from src.data.prefixes import PrefixDataset, collate_pad  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/helpdesk_spc.yaml",
    "configs/traffic_spc.yaml",
    "configs/bpic2012_spc.yaml",
    "configs/sepsis_spc.yaml",
    "configs/bpic2019_spc.yaml",
]

PROXIES = ("entropy", "conf", "margin")
PROXY_LABEL = {
    "entropy": "Entropy H(p)",
    "conf": "1 − max p",
    "margin": "1 − (p1−p2)",
}
PROXY_COLOR = {
    "entropy": "#C44E52",
    "conf": "#4C72B0",
    "margin": "#55A868",
}

# Target fire rates; 0.30 ≈ entropy p70 default
DEFAULT_FIRE_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]
DEFAULT_ACC_BUDGET = 0.005


def _val_loader(datasets, legal_masks, num_activities, batch_size):
    ds = PrefixDataset(datasets["val"], legal_masks=legal_masks, num_activities=num_activities)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)


@torch.no_grad()
def _collect_uncertainty(model, loader, device, kind: str) -> np.ndarray:
    model.eval()
    vals: List[float] = []
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
        vals.extend(prediction_uncertainty(logits, kind=kind).cpu().tolist())
    return np.asarray(vals, dtype=np.float64)


def _tau_for_fire_rate(u_val: np.ndarray, fire: float) -> float:
    """Gate when U >= tau; want P(U>=tau) ≈ fire => tau = quantile(U, 1-fire)."""
    fire = float(np.clip(fire, 0.0, 1.0))
    q = 1.0 - fire
    if fire <= 0:
        return float(np.max(u_val) + 1.0)  # never fire
    if fire >= 1:
        return float(np.min(u_val) - 1e-9)  # always fire
    return float(np.quantile(u_val, q))


def _eval(
    model,
    samples,
    out_dir,
    num_activities,
    bs,
    device,
    gate_tau: Optional[float],
    uncertainty: str,
    use_legal_mask: bool = False,
) -> Dict[str, float]:
    return eval_transformer_on_samples(
        model,
        samples,
        out_dir,
        num_activities,
        bs,
        device,
        use_legal_mask=use_legal_mask,
        gate_tau=gate_tau,
        uncertainty=uncertainty,
    )


def run_dataset(
    cfg_path: str,
    seeds: List[int],
    fire_rates: List[float],
    noise_rate: float,
    acc_budget: float,
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
        test_noisy = pickle.load(f)

    val_noise_path = out_dir / f"noise_{noise_rate}" / "prefixes_val.pkl"
    if not val_noise_path.exists():
        build_noise_prefixes_for_split(cfg, out_dir, float(noise_rate), "val")
    with open(val_noise_path, "rb") as f:
        val_noisy = pickle.load(f)

    test_clean = datasets["test"]
    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        soft_pt = out_dir / f"soft_seed{seed}.pt"
        if not soft_pt.exists():
            print(f"  skip seed={seed}: no soft ckpt", flush=True)
            continue
        print(f"[{cfg.get('dataset')}] seed={seed}", flush=True)
        model = _build_model(cfg, bundle, datasets, out_dir, device)
        model.load_state_dict(torch.load(soft_pt, map_location=device))
        model.eval()

        vloader = _val_loader(datasets, legal_masks, bundle.num_activities, bs)
        soft_test = _eval(
            model, test_noisy, out_dir, bundle.num_activities, bs, device, None, "entropy"
        )
        soft_val = _eval(
            model, val_noisy, out_dir, bundle.num_activities, bs, device, None, "entropy"
        )
        soft_nacc_test = float(soft_test["accuracy"])
        soft_npcp_test = float(soft_test["legal_rate"])
        soft_nacc_val = float(soft_val["accuracy"])

        seed_row: Dict[str, Any] = {
            "seed": seed,
            "soft": {
                "noise_accuracy": soft_nacc_test,
                "noise_legal_rate": soft_npcp_test,
                "val_noise_accuracy": soft_nacc_val,
            },
            "matched_fire": {},
            "acc_budget": {},
        }

        u_val = {k: _collect_uncertainty(model, vloader, device, k) for k in PROXIES}

        # --- Protocol 1: matched fire-rate ---
        for fire in fire_rates:
            fire_key = f"fire_{int(round(fire * 100))}"
            seed_row["matched_fire"][fire_key] = {"target_fire": fire, "proxies": {}}
            for kind in PROXIES:
                tau = _tau_for_fire_rate(u_val[kind], fire)
                m = _eval(
                    model,
                    test_noisy,
                    out_dir,
                    bundle.num_activities,
                    bs,
                    device,
                    tau,
                    kind,
                )
                cell = {
                    "tau": tau,
                    "accuracy": float(m["accuracy"]),
                    "legal_rate": float(m["legal_rate"]),
                    "gate_fire_rate": float(m.get("gate_fire_rate") or 0.0),
                    "d_acc_vs_soft": float(m["accuracy"]) - soft_nacc_test,
                    "d_pcp_vs_soft": float(m["legal_rate"]) - soft_npcp_test,
                }
                seed_row["matched_fire"][fire_key]["proxies"][kind] = cell

            # log primary fire=0.30
            if abs(fire - 0.30) < 1e-9:
                bits = [
                    f"{k} PCP={seed_row['matched_fire'][fire_key]['proxies'][k]['legal_rate']:.4f}"
                    for k in PROXIES
                ]
                print(f"  matched fire=0.30: " + " | ".join(bits), flush=True)

        # --- Protocol 2: Acc budget on val ---
        # Scan same fire grid on val; pick max PCP under Acc drop <= budget
        for kind in PROXIES:
            candidates = []
            for fire in fire_rates:
                tau = _tau_for_fire_rate(u_val[kind], fire)
                mv = _eval(
                    model, val_noisy, out_dir, bundle.num_activities, bs, device, tau, kind
                )
                d_acc = float(mv["accuracy"]) - soft_nacc_val
                candidates.append(
                    {
                        "target_fire": fire,
                        "tau": tau,
                        "val_accuracy": float(mv["accuracy"]),
                        "val_legal_rate": float(mv["legal_rate"]),
                        "val_d_acc": d_acc,
                        "val_gate_fire_rate": float(mv.get("gate_fire_rate") or 0.0),
                        "feasible": d_acc >= -float(acc_budget),
                    }
                )
            feasible = [c for c in candidates if c["feasible"]]
            pool = feasible if feasible else candidates  # fallback: least Acc damage
            if feasible:
                best = max(pool, key=lambda c: (c["val_legal_rate"], -c["target_fire"]))
                sel_rule = "max_val_pcp_under_acc_budget"
            else:
                best = max(pool, key=lambda c: c["val_d_acc"])
                sel_rule = "fallback_min_acc_drop"

            mt = _eval(
                model,
                test_noisy,
                out_dir,
                bundle.num_activities,
                bs,
                device,
                float(best["tau"]),
                kind,
            )
            seed_row["acc_budget"][kind] = {
                "selected": best,
                "selection_rule": sel_rule,
                "test_accuracy": float(mt["accuracy"]),
                "test_legal_rate": float(mt["legal_rate"]),
                "test_gate_fire_rate": float(mt.get("gate_fire_rate") or 0.0),
                "d_acc_vs_soft": float(mt["accuracy"]) - soft_nacc_test,
                "d_pcp_vs_soft": float(mt["legal_rate"]) - soft_npcp_test,
                "candidates": candidates,
            }

        # Soft / HardMask anchors (hard from own ckpt if present)
        seed_row["anchors"] = {
            "soft": {
                "accuracy": soft_nacc_test,
                "legal_rate": soft_npcp_test,
            }
        }
        hard_pt = out_dir / f"hardmask_seed{seed}.pt"
        if hard_pt.exists():
            hard = _build_model(cfg, bundle, datasets, out_dir, device)
            hard.load_state_dict(torch.load(hard_pt, map_location=device))
            hard.eval()
            hm = _eval(
                hard,
                test_noisy,
                out_dir,
                bundle.num_activities,
                bs,
                device,
                None,
                "entropy",
                use_legal_mask=True,
            )
            seed_row["anchors"]["hardmask"] = {
                "accuracy": float(hm["accuracy"]),
                "legal_rate": float(hm["legal_rate"]),
            }

        # clean reference at fire=0.30 entropy (optional)
        tau_e = _tau_for_fire_rate(u_val["entropy"], 0.30)
        mc = _eval(
            model, test_clean, out_dir, bundle.num_activities, bs, device, tau_e, "entropy"
        )
        seed_row["clean_entropy_fire30"] = {
            "accuracy": float(mc["accuracy"]),
            "legal_rate": float(mc.get("legal_rate") or 0.0),
        }

        per_seed.append(seed_row)

    summary = _aggregate(per_seed, fire_rates)
    return {
        "dataset": cfg.get("dataset"),
        "config": cfg_path,
        "seeds": seeds,
        "proxies": list(PROXIES),
        "fire_rates": fire_rates,
        "noise_rate": noise_rate,
        "acc_budget": acc_budget,
        "per_seed": per_seed,
        "summary": summary,
    }


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in vals],
    }


def _aggregate(per_seed: List[Dict[str, Any]], fire_rates: List[float]) -> Dict[str, Any]:
    if not per_seed:
        return {}
    out: Dict[str, Any] = {"matched_fire": {}, "acc_budget": {}, "soft": {}}

    soft_acc = [r["soft"]["noise_accuracy"] for r in per_seed]
    soft_pcp = [r["soft"]["noise_legal_rate"] for r in per_seed]
    out["soft"] = {
        "noise_accuracy": _mean_std(soft_acc),
        "noise_legal_rate": _mean_std(soft_pcp),
    }

    for fire in fire_rates:
        fire_key = f"fire_{int(round(fire * 100))}"
        out["matched_fire"][fire_key] = {"target_fire": fire, "proxies": {}}
        for kind in PROXIES:
            accs, pcps, fires, d_acc, d_pcp = [], [], [], [], []
            for r in per_seed:
                cell = r["matched_fire"][fire_key]["proxies"][kind]
                accs.append(cell["accuracy"])
                pcps.append(cell["legal_rate"])
                fires.append(cell["gate_fire_rate"])
                d_acc.append(cell["d_acc_vs_soft"])
                d_pcp.append(cell["d_pcp_vs_soft"])
            out["matched_fire"][fire_key]["proxies"][kind] = {
                "accuracy": _mean_std(accs),
                "legal_rate": _mean_std(pcps),
                "gate_fire_rate": _mean_std(fires),
                "d_acc_vs_soft": _mean_std(d_acc),
                "d_pcp_vs_soft": _mean_std(d_pcp),
            }

    for kind in PROXIES:
        accs, pcps, fires, d_acc, d_pcp, sel_f = [], [], [], [], [], []
        for r in per_seed:
            cell = r["acc_budget"][kind]
            accs.append(cell["test_accuracy"])
            pcps.append(cell["test_legal_rate"])
            fires.append(cell["test_gate_fire_rate"])
            d_acc.append(cell["d_acc_vs_soft"])
            d_pcp.append(cell["d_pcp_vs_soft"])
            sel_f.append(cell["selected"]["target_fire"])
        out["acc_budget"][kind] = {
            "accuracy": _mean_std(accs),
            "legal_rate": _mean_std(pcps),
            "gate_fire_rate": _mean_std(fires),
            "d_acc_vs_soft": _mean_std(d_acc),
            "d_pcp_vs_soft": _mean_std(d_pcp),
            "selected_fire": _mean_std(sel_f),
        }
    return out


def _fire_series(ds: Dict[str, Any], kind: str, metric: str) -> tuple:
    """Return (fire_pct_list, values) for matched-fire metric mean."""
    fires = list(ds["fire_rates"])
    mf = ds["summary"]["matched_fire"]
    xs = [100.0 * f for f in fires]
    ys = [
        mf[f"fire_{int(round(f * 100))}"]["proxies"][kind][metric]["mean"]
        for f in fires
    ]
    return xs, ys


def _plot_dataset(ds: Dict[str, Any], out_png: Path) -> None:
    """Per-dataset: ΔPCP/ΔAcc vs fire + Acc–PCP delta Pareto."""
    summary = ds.get("summary") or {}
    if not summary.get("matched_fire"):
        return
    fires = list(ds["fire_rates"])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)

    ax = axes[0]
    for kind in PROXIES:
        xs, ys = _fire_series(ds, kind, "d_pcp_vs_soft")
        ax.plot(xs, ys, "o-", color=PROXY_COLOR[kind], label=PROXY_LABEL[kind], lw=1.8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(30, color="#2A9D8F", ls="--", lw=1.0, alpha=0.75)
    ax.set_xlabel("Matched fire rate (%)")
    ax.set_ylabel("ΔPCP vs Soft")
    ax.set_title(f"{ds['dataset']}: ΔPCP vs fire")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for kind in PROXIES:
        xs, ys = _fire_series(ds, kind, "d_acc_vs_soft")
        ax.plot(xs, ys, "s-", color=PROXY_COLOR[kind], label=PROXY_LABEL[kind], lw=1.8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(30, color="#2A9D8F", ls="--", lw=1.0, alpha=0.75)
    ax.set_xlabel("Matched fire rate (%)")
    ax.set_ylabel("ΔAcc vs Soft")
    ax.set_title(f"{ds['dataset']}: ΔAcc vs fire")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    for kind in PROXIES:
        _, d_acc = _fire_series(ds, kind, "d_acc_vs_soft")
        _, d_pcp = _fire_series(ds, kind, "d_pcp_vs_soft")
        ax.plot(
            d_acc,
            d_pcp,
            "o-",
            color=PROXY_COLOR[kind],
            label=PROXY_LABEL[kind],
            lw=1.8,
        )
        # mark fire=30%
        if 0.30 in fires:
            i = fires.index(0.30)
            ax.scatter(
                [d_acc[i]],
                [d_pcp[i]],
                s=70,
                facecolors="none",
                edgecolors=PROXY_COLOR[kind],
                linewidths=1.4,
                zorder=5,
            )
    ax.axhline(0, color="#999", lw=0.7)
    ax.axvline(0, color="#999", lw=0.7)
    ax.set_xlabel("ΔAcc vs Soft")
    ax.set_ylabel("ΔPCP vs Soft")
    ax.set_title("Trade-off (○ = fire 30%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(all_ds: List[Dict[str, Any]], out_png: Path, fire: float = 0.30) -> None:
    """
    Cross-dataset mean curves over fire grid + delta Pareto.
    (fire arg kept for API compat; primary story is full budget curves.)
    """
    usable = [d for d in all_ds if d.get("summary", {}).get("matched_fire")]
    if not usable:
        return
    fires = list(usable[0]["fire_rates"])

    def mean_over_ds(kind: str, metric: str) -> List[float]:
        out = []
        for f in fires:
            key = f"fire_{int(round(f * 100))}"
            vals = [
                d["summary"]["matched_fire"][key]["proxies"][kind][metric]["mean"]
                for d in usable
            ]
            out.append(float(np.mean(vals)))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    xs = [100.0 * f for f in fires]

    ax = axes[0]
    for kind in PROXIES:
        ax.plot(
            xs,
            mean_over_ds(kind, "d_pcp_vs_soft"),
            "o-",
            color=PROXY_COLOR[kind],
            label=PROXY_LABEL[kind],
            lw=2.0,
        )
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(100 * fire, color="#2A9D8F", ls="--", lw=1.0, alpha=0.75, label=f"fire={int(fire*100)}%")
    ax.set_xlabel("Matched fire rate (%)")
    ax.set_ylabel("ΔPCP vs Soft (mean over logs)")
    ax.set_title("ΔPCP vs intervention budget")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for kind in PROXIES:
        ax.plot(
            xs,
            mean_over_ds(kind, "d_acc_vs_soft"),
            "s-",
            color=PROXY_COLOR[kind],
            label=PROXY_LABEL[kind],
            lw=2.0,
        )
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(100 * fire, color="#2A9D8F", ls="--", lw=1.0, alpha=0.75)
    ax.set_xlabel("Matched fire rate (%)")
    ax.set_ylabel("ΔAcc vs Soft (mean over logs)")
    ax.set_title("ΔAcc vs intervention budget")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    for kind in PROXIES:
        d_acc = mean_over_ds(kind, "d_acc_vs_soft")
        d_pcp = mean_over_ds(kind, "d_pcp_vs_soft")
        ax.plot(d_acc, d_pcp, "o-", color=PROXY_COLOR[kind], label=PROXY_LABEL[kind], lw=2.0)
        if fire in fires:
            i = fires.index(fire)
            ax.scatter(
                [d_acc[i]],
                [d_pcp[i]],
                s=80,
                facecolors="none",
                edgecolors=PROXY_COLOR[kind],
                linewidths=1.6,
                zorder=5,
            )
    ax.axhline(0, color="#999", lw=0.7)
    ax.axvline(0, color="#999", lw=0.7)
    ax.set_xlabel("ΔAcc vs Soft")
    ax.set_ylabel("ΔPCP vs Soft")
    ax.set_title("Proxy trade-off curve (○ = 30%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        "Uncertainty proxies across fire rates 10–50% (SoftLegal weights)",
        fontsize=11,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto_only(all_ds: List[Dict[str, Any]], out_png: Path) -> None:
    """Standalone Acc–PCP delta Pareto (paper figure focus)."""
    usable = [d for d in all_ds if d.get("summary", {}).get("matched_fire")]
    if not usable:
        return
    fires = list(usable[0]["fire_rates"])

    def mean_over_ds(kind: str, metric: str) -> List[float]:
        out = []
        for f in fires:
            key = f"fire_{int(round(f * 100))}"
            vals = [
                d["summary"]["matched_fire"][key]["proxies"][kind][metric]["mean"]
                for d in usable
            ]
            out.append(float(np.mean(vals)))
        return out

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    for kind in PROXIES:
        d_acc = mean_over_ds(kind, "d_acc_vs_soft")
        d_pcp = mean_over_ds(kind, "d_pcp_vs_soft")
        ax.plot(d_acc, d_pcp, "o-", color=PROXY_COLOR[kind], label=PROXY_LABEL[kind], lw=2.0)
        # annotate endpoints
        ax.annotate("10%", (d_acc[0], d_pcp[0]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.annotate("50%", (d_acc[-1], d_pcp[-1]), textcoords="offset points", xytext=(4, -10), fontsize=7)
        if 0.30 in fires:
            i = fires.index(0.30)
            ax.scatter(
                [d_acc[i]],
                [d_pcp[i]],
                s=90,
                facecolors="none",
                edgecolors=PROXY_COLOR[kind],
                linewidths=1.8,
                zorder=5,
            )
    ax.axhline(0, color="#999", lw=0.7)
    ax.axvline(0, color="#999", lw=0.7)
    ax.set_xlabel("ΔAcc vs SoftLegal (noise)")
    ax.set_ylabel("ΔPCP vs SoftLegal (noise)")
    ax.set_title("Proxy-agnostic? Acc–PCP trade-off over fire 10–50%")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fmt_txt(all_ds: List[Dict[str, Any]], primary_fire: float = 0.30) -> str:
    fire_key = f"fire_{int(round(primary_fire * 100))}"
    lines = [
        "Uncertainty-proxy gate ablation (SoftLegal weights)",
        "U_entropy=H(p); U_conf=1-max p; U_margin=1-(p1-p2)",
        "Matched fire grid: 10%,20%,30%,40%,50% (full intervention budget).",
        f"Primary snapshot fire={primary_fire:.2f} (entropy p70 equivalent).",
        f"Acc-budget protocol: val noise Acc drop <= budget vs Soft, then max PCP.",
        "",
    ]
    for ds in all_ds:
        s = ds.get("summary") or {}
        if not s:
            continue
        lines.append(f"=== {ds['dataset']} ===")
        lines.append(
            f"Soft noise Acc/PCP: "
            f"{s['soft']['noise_accuracy']['mean']:.4f} / "
            f"{s['soft']['noise_legal_rate']['mean']:.4f}"
        )
        # full fire grid compact
        lines.append(f"{'fire':>5} " + " ".join(f"{k:>14s}" for k in PROXIES) + "   (dPCP)")
        for f in ds["fire_rates"]:
            fk = f"fire_{int(round(f * 100))}"
            cells = []
            for kind in PROXIES:
                d = s["matched_fire"][fk]["proxies"][kind]["d_pcp_vs_soft"]["mean"]
                cells.append(f"{d:+14.4f}")
            lines.append(f"{int(round(100*f)):5d} " + " ".join(cells))
        # Acc grid
        lines.append(f"{'fire':>5} " + " ".join(f"{k:>14s}" for k in PROXIES) + "   (dAcc)")
        for f in ds["fire_rates"]:
            fk = f"fire_{int(round(f * 100))}"
            cells = []
            for kind in PROXIES:
                d = s["matched_fire"][fk]["proxies"][kind]["d_acc_vs_soft"]["mean"]
                cells.append(f"{d:+14.4f}")
            lines.append(f"{int(round(100*f)):5d} " + " ".join(cells))

        lines.append(f"-- snapshot fire={primary_fire:.2f} --")
        block = s["matched_fire"][fire_key]["proxies"]
        ranked = sorted(PROXIES, key=lambda k: block[k]["legal_rate"]["mean"], reverse=True)
        spread = block[ranked[0]]["legal_rate"]["mean"] - block[ranked[-1]]["legal_rate"]["mean"]
        lines.append(f"  PCP rank: {' > '.join(ranked)}  (max-min={spread:.4f})")
        lines.append("")

    # Cross-dataset mean curves
    lines.append("=== Cross-dataset mean ΔPCP vs fire ===")
    lines.append(f"{'fire':>5} " + " ".join(f"{k:>10s}" for k in PROXIES))
    fires = list(all_ds[0]["fire_rates"]) if all_ds else []
    curve = {k: [] for k in PROXIES}
    acc_curve = {k: [] for k in PROXIES}
    for f in fires:
        fk = f"fire_{int(round(f * 100))}"
        row = [f"{int(round(100*f)):5d}"]
        for kind in PROXIES:
            vals = [
                d["summary"]["matched_fire"][fk]["proxies"][kind]["d_pcp_vs_soft"]["mean"]
                for d in all_ds
                if d.get("summary")
            ]
            m = float(np.mean(vals)) if vals else 0.0
            curve[kind].append(m)
            row.append(f"{m:+10.4f}")
            avals = [
                d["summary"]["matched_fire"][fk]["proxies"][kind]["d_acc_vs_soft"]["mean"]
                for d in all_ds
                if d.get("summary")
            ]
            acc_curve[kind].append(float(np.mean(avals)) if avals else 0.0)
        lines.append(" ".join(row))

    lines.append("=== Cross-dataset mean ΔAcc vs fire ===")
    lines.append(f"{'fire':>5} " + " ".join(f"{k:>10s}" for k in PROXIES))
    for i, f in enumerate(fires):
        row = [f"{int(round(100*f)):5d}"] + [f"{acc_curve[k][i]:+10.4f}" for k in PROXIES]
        lines.append(" ".join(row))

    # curve separation: mean |entropy - other| over fire grid
    if fires:
        sep_ec = float(np.mean([abs(curve["entropy"][i] - curve["conf"][i]) for i in range(len(fires))]))
        sep_em = float(np.mean([abs(curve["entropy"][i] - curve["margin"][i]) for i in range(len(fires))]))
        # dominance: how often entropy has highest ΔPCP
        wins = sum(
            1
            for i in range(len(fires))
            if curve["entropy"][i] >= max(curve["conf"][i], curve["margin"][i]) - 1e-12
        )
        lines.append("")
        lines.append(
            f"Mean |ΔPCP entropy-conf| over fire grid: {sep_ec:.4f}; "
            f"|entropy-margin|: {sep_em:.4f}"
        )
        lines.append(f"Entropy has best/tied mean ΔPCP at {wins}/{len(fires)} fire rates.")
        if max(sep_ec, sep_em) < 0.003:
            lines.append(
                "Verdict: curves nearly coincide across 10-50% fire -> "
                "SPC is largely proxy-agnostic."
            )
        elif wins >= len(fires) - 1 and curve["entropy"][-1] >= max(curve["conf"][-1], curve["margin"][-1]):
            lines.append(
                "Verdict: entropy frontier weakly dominates across budget -> "
                "Entropy offers the best overall trade-off."
            )
        else:
            lines.append(
                "Verdict: mixed; entropy competitive over the full budget, not only at 30%."
            )
    return "\n".join(lines)


def _replot_from_json(json_path: Path, out_dir: Path) -> int:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    all_results = payload.get("results") or []
    if not all_results:
        print(f"No results in {json_path}", flush=True)
        return 1
    for ds in all_results:
        _plot_dataset(ds, out_dir / f"uablate_{ds['dataset']}.png")
    _plot_overview(all_results, out_dir / "uablate_overview.png", fire=0.30)
    _plot_pareto_only(all_results, out_dir / "uablate_pareto.png")
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "uablate_overview.png", fig_dir / "fig_uncertainty_gate.png")
    shutil.copy2(out_dir / "uablate_pareto.png", fig_dir / "fig_uncertainty_pareto.png")
    txt = _fmt_txt(all_results, primary_fire=0.30)
    (out_dir / "uncertainty_gate_ablation.txt").write_text(txt, encoding="utf-8")
    try:
        print(txt, flush=True)
    except UnicodeEncodeError:
        print(txt.encode("ascii", errors="replace").decode("ascii"), flush=True)
    print(f"Replotted -> {out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SPC uncertainty-proxy gate ablation")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--fire_rates", nargs="+", type=float, default=DEFAULT_FIRE_RATES)
    ap.add_argument("--noise_rate", type=float, default=0.2)
    ap.add_argument("--acc_budget", type=float, default=DEFAULT_ACC_BUDGET)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs" / "uncertainty_gate_ablation"),
    )
    ap.add_argument(
        "--replot",
        action="store_true",
        help="Only regenerate curves/Pareto from saved uncertainty_gate_ablation.json",
    )
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        jp = out_dir / "uncertainty_gate_ablation.json"
        return _replot_from_json(jp, out_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    all_results: List[Dict[str, Any]] = []
    for cfg_path in args.configs:
        cfg_full = str(ROOT / cfg_path) if not Path(cfg_path).is_absolute() else cfg_path
        if not Path(cfg_full).exists():
            cfg_full = cfg_path
        print(f"== uncertainty ablation {cfg_full} ==", flush=True)
        try:
            ds = run_dataset(
                cfg_full,
                list(args.seeds),
                list(args.fire_rates),
                float(args.noise_rate),
                float(args.acc_budget),
                device,
            )
        except FileNotFoundError as e:
            print(f"skip: {e}", flush=True)
            continue
        all_results.append(ds)
        save_json(out_dir / f"uablate_{ds['dataset']}.json", ds)
        _plot_dataset(ds, out_dir / f"uablate_{ds['dataset']}.png")

    if all_results:
        _plot_overview(all_results, out_dir / "uablate_overview.png", fire=0.30)
        _plot_pareto_only(all_results, out_dir / "uablate_pareto.png")
        fig_dir = ROOT / "docs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "uablate_overview.png", fig_dir / "fig_uncertainty_gate.png")
        shutil.copy2(out_dir / "uablate_pareto.png", fig_dir / "fig_uncertainty_pareto.png")

    payload = {
        "proxies": list(PROXIES),
        "fire_rates": args.fire_rates,
        "noise_rate": args.noise_rate,
        "acc_budget": args.acc_budget,
        "seeds": args.seeds,
        "datasets": [d["dataset"] for d in all_results],
        "results": all_results,
    }
    save_json(out_dir / "uncertainty_gate_ablation.json", payload)
    txt = _fmt_txt(all_results, primary_fire=0.30)
    (out_dir / "uncertainty_gate_ablation.txt").write_text(txt, encoding="utf-8")
    try:
        print(txt, flush=True)
    except UnicodeEncodeError:
        print(txt.encode("ascii", errors="replace").decode("ascii"), flush=True)
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
