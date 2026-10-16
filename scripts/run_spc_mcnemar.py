#!/usr/bin/env python
"""
Sample-level McNemar + bootstrap CI for SPC study.

Reloads Base / Soft / HardMask / SPC checkpoints, evaluates the same test
(and noise) prefixes, then for Acc and PCP (legal):
  - McNemar exact (binomial) + mid-p style chi-square with continuity correction
  - Bootstrap 95% CI for paired rate difference (SPC − reference)

Per (dataset, seed, split) — do not pool seeds for McNemar (same samples × 3).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spc_study import (  # noqa: E402
    _build_model,
    _build_noise_prefixes_fast,
)
from scripts.utils import load_config, save_json  # noqa: E402
from src.conformance.legal_next import (  # noqa: E402
    apply_gated_legal_mask,
    apply_legal_mask_to_logits,
    build_successor_index,
    cache_legal_masks,
    prediction_entropy,
)
from src.data.prefixes import PrefixDataset, collate_pad  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/helpdesk_spc.yaml",
    "configs/traffic_spc.yaml",
    "configs/bpic2012_spc.yaml",
    "configs/sepsis_spc.yaml",
    "configs/bpic2019_spc.yaml",
]

COMPARISONS = [
    ("spc", "base", "spc_vs_base"),
    ("spc", "soft", "spc_vs_soft"),
    ("hardmask", "base", "hardmask_vs_base"),
]


def _mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> Dict[str, Any]:
    """
    McNemar on paired binary outcomes (True=success).
    Tests H0: P(a success) = P(b success).
    b01: a False, b True; b10: a True, b False.
    """
    a_ok = np.asarray(a_ok, dtype=bool)
    b_ok = np.asarray(b_ok, dtype=bool)
    assert a_ok.shape == b_ok.shape
    n01 = int((~a_ok & b_ok).sum())  # a fail, b success
    n10 = int((a_ok & ~b_ok).sum())  # a success, b fail
    n11 = int((a_ok & b_ok).sum())
    n00 = int((~a_ok & ~b_ok).sum())
    discordant = n01 + n10
    out: Dict[str, Any] = {
        "n": int(a_ok.size),
        "n11": n11,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "discordant": discordant,
        "a_rate": float(a_ok.mean()),
        "b_rate": float(b_ok.mean()),
        "delta_a_minus_b": float(a_ok.mean() - b_ok.mean()),
    }
    if discordant == 0:
        out.update(
            {
                "exact_pvalue": 1.0,
                "chi2_corr": 0.0,
                "chi2_pvalue": 1.0,
                "status": "no_discordant",
            }
        )
        return out

    # Exact binomial test: under H0, n10 ~ Bin(discordant, 0.5)
    from scipy.stats import binomtest, chi2

    exact = binomtest(n10, n=discordant, p=0.5, alternative="two-sided")
    out["exact_pvalue"] = float(exact.pvalue)
    # Edwards continuity-corrected chi-square
    chi2_stat = (abs(n10 - n01) - 1) ** 2 / discordant
    out["chi2_corr"] = float(chi2_stat)
    out["chi2_pvalue"] = float(chi2.sf(chi2_stat, df=1))
    out["status"] = "ok"
    return out


def _bootstrap_delta_ci(
    a_ok: np.ndarray,
    b_ok: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile bootstrap CI for mean(a) - mean(b) on paired samples."""
    a_ok = np.asarray(a_ok, dtype=float)
    b_ok = np.asarray(b_ok, dtype=float)
    n = a_ok.size
    rng = np.random.default_rng(seed)
    point = float(a_ok.mean() - b_ok.mean())
    if n == 0:
        return {"point": point, "ci_low": None, "ci_high": None, "n_boot": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a_ok[idx].mean(axis=1) - b_ok[idx].mean(axis=1)
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": n_boot,
        "alpha": alpha,
        "mean_boot": float(deltas.mean()),
        "std_boot": float(deltas.std(ddof=1)),
    }


@torch.no_grad()
def _predict_outcomes(
    model,
    samples,
    legal_masks,
    num_activities: int,
    batch_size: int,
    device,
    use_legal_mask: bool,
    gate_tau: Optional[float],
) -> Dict[str, np.ndarray]:
    ds = PrefixDataset(samples, legal_masks=legal_masks, num_activities=num_activities)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    model.eval()
    ys: List[int] = []
    preds: List[int] = []
    legal_flags: List[bool] = []

    for batch in loader:
        prefix = batch["prefix_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["label"].to(device)
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
        legal_mask = batch.get("legal_mask")
        if legal_mask is not None:
            legal_mask = legal_mask.to(device)

        logits = model(
            prefix,
            lengths,
            key_padding_mask=key_pad,
            time_feat=time_feat,
            resource_ids=resource_ids,
            attr_ids=attr_ids,
        )
        if legal_mask is not None:
            if use_legal_mask:
                logits = apply_legal_mask_to_logits(logits, legal_mask)
            elif gate_tau is not None and gate_tau >= 0:
                logits = apply_gated_legal_mask(logits, legal_mask, tau=float(gate_tau))

        pred = logits.argmax(dim=-1)
        ys.extend(labels.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        if legal_mask is not None:
            hit = legal_mask[torch.arange(legal_mask.size(0), device=device), pred]
            legal_flags.extend(hit.bool().cpu().tolist())
        else:
            legal_flags.extend([True] * pred.size(0))

    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    legal = np.asarray(legal_flags, dtype=bool)
    correct = y == p
    return {"y": y, "pred": p, "correct": correct, "legal": legal}


def _legal_masks_for_samples(samples, out_dir: Path, num_activities: int):
    meta_path = out_dir / "legal_next.pkl"
    with open(meta_path, "rb") as f:
        blob = pickle.load(f)
    succ = blob.get("succ_index")
    if succ is None and blob.get("pairs") is not None:
        succ = build_successor_index(blob["pairs"], num_activities)
    if succ is None:
        return blob["masks"]
    return cache_legal_masks(samples, succ, num_activities)


def _load_variant(
    out_dir: Path,
    variant: str,
    seed: int,
    cfg,
    bundle,
    datasets,
    device,
) -> Tuple[torch.nn.Module, Optional[float], Dict[str, Any]]:
    tag = f"{variant}_seed{seed}"
    pt = out_dir / f"{tag}.pt"
    metrics_path = out_dir / f"{tag}_metrics.json"
    if not pt.exists():
        raise FileNotFoundError(pt)
    meta = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    model = _build_model(cfg, bundle, datasets, out_dir, device)
    model.load_state_dict(torch.load(pt, map_location=device))
    model.eval()
    gate_tau = meta.get("gate_tau")
    if variant == "hardmask":
        return model, None, {"use_legal_mask": True, "gate_tau": None, "meta": meta}
    if variant == "spc":
        tau = float(gate_tau) if gate_tau is not None else None
        return model, tau, {"use_legal_mask": False, "gate_tau": tau, "meta": meta}
    return model, None, {"use_legal_mask": False, "gate_tau": None, "meta": meta}


def _eval_variant_on_split(
    model,
    eval_kw: Dict[str, Any],
    samples,
    out_dir: Path,
    num_activities: int,
    bs: int,
    device,
) -> Dict[str, np.ndarray]:
    masks = _legal_masks_for_samples(samples, out_dir, num_activities)
    return _predict_outcomes(
        model,
        samples,
        masks,
        num_activities,
        bs,
        device,
        use_legal_mask=bool(eval_kw.get("use_legal_mask")),
        gate_tau=eval_kw.get("gate_tau"),
    )


def run_dataset(
    cfg_path: str,
    seeds: List[int],
    noise_rates: List[float],
    n_boot: int,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    bs = int(cfg.get("batch_size", 32))

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    splits: Dict[str, Any] = {"clean": datasets["test"]}
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
        for var in ("base", "soft", "spc", "hardmask"):
            try:
                model, _, ekw = _load_variant(out_dir, var, seed, cfg, bundle, datasets, device)
                loaded[var] = (model, ekw)
            except FileNotFoundError as e:
                print(f"  skip missing {e}", flush=True)

        seed_row: Dict[str, Any] = {"seed": seed, "splits": {}}
        for split_name, samples in splits.items():
            outcomes = {}
            for var, (model, ekw) in loaded.items():
                outcomes[var] = _eval_variant_on_split(
                    model, ekw, samples, out_dir, bundle.num_activities, bs, device
                )
                print(
                    f"  {split_name}/{var}: "
                    f"acc={outcomes[var]['correct'].mean():.4f} "
                    f"pcp={outcomes[var]['legal'].mean():.4f} n={len(outcomes[var]['correct'])}",
                    flush=True,
                )

            comps = {}
            for a_name, b_name, key in COMPARISONS:
                if a_name not in outcomes or b_name not in outcomes:
                    continue
                oa, ob = outcomes[a_name], outcomes[b_name]
                # Acc McNemar: success = correct
                m_acc = _mcnemar(oa["correct"], ob["correct"])
                ci_acc = _bootstrap_delta_ci(
                    oa["correct"], ob["correct"], n_boot=n_boot, seed=seed + hash(split_name) % 10007
                )
                # PCP McNemar: success = legal
                m_pcp = _mcnemar(oa["legal"], ob["legal"])
                ci_pcp = _bootstrap_delta_ci(
                    oa["legal"],
                    ob["legal"],
                    n_boot=n_boot,
                    seed=seed + 17 + hash(split_name) % 10007,
                )
                comps[key] = {
                    "accuracy": {"mcnemar": m_acc, "bootstrap_delta": ci_acc},
                    "pcp": {"mcnemar": m_pcp, "bootstrap_delta": ci_pcp},
                }
            seed_row["splits"][split_name] = {
                "n": int(len(next(iter(outcomes.values()))["correct"])),
                "rates": {
                    v: {
                        "accuracy": float(o["correct"].mean()),
                        "pcp": float(o["legal"].mean()),
                    }
                    for v, o in outcomes.items()
                },
                "comparisons": comps,
            }
        per_seed.append(seed_row)

    # Aggregate: median exact p across seeds; mean delta + pooled narrative
    summary: Dict[str, Any] = {}
    for split_name in splits:
        summary[split_name] = {}
        for _, _, key in COMPARISONS:
            for metric in ("accuracy", "pcp"):
                ps, deltas, ci_lows, ci_highs = [], [], [], []
                for row in per_seed:
                    block = (
                        row["splits"]
                        .get(split_name, {})
                        .get("comparisons", {})
                        .get(key, {})
                        .get(metric)
                    )
                    if not block:
                        continue
                    ps.append(block["mcnemar"]["exact_pvalue"])
                    deltas.append(block["bootstrap_delta"]["point"])
                    ci_lows.append(block["bootstrap_delta"]["ci_low"])
                    ci_highs.append(block["bootstrap_delta"]["ci_high"])
                if not ps:
                    continue
                summary[split_name][f"{key}_{metric}"] = {
                    "n_seeds": len(ps),
                    "exact_p_values": ps,
                    "median_exact_p": float(np.median(ps)),
                    "min_exact_p": float(np.min(ps)),
                    "max_exact_p": float(np.max(ps)),
                    "mean_delta": float(np.mean(deltas)),
                    "delta_per_seed": deltas,
                    "ci_low_per_seed": ci_lows,
                    "ci_high_per_seed": ci_highs,
                    # conservative: CI that covers all seed point estimates' mean via averaging bounds
                    "mean_ci_low": float(np.mean(ci_lows)),
                    "mean_ci_high": float(np.mean(ci_highs)),
                }

    return {
        "dataset": cfg.get("dataset"),
        "config": cfg_path,
        "seeds": seeds,
        "noise_rates": noise_rates,
        "n_boot": n_boot,
        "per_seed": per_seed,
        "summary": summary,
        "note": (
            "McNemar exact binomial on discordant pairs; "
            "bootstrap percentile CI for paired rate difference (A−B). "
            "PCP success = prediction in allowed successor set."
        ),
    }


def _fmt_summary(all_ds: List[Dict[str, Any]]) -> str:
    lines = [
        "Sample-level McNemar + bootstrap CI (SPC study)",
        "McNemar exact p on Acc / PCP; bootstrap 95% CI for delta = A-B (paired).",
        "",
    ]
    for ds in all_ds:
        lines.append(f"=== {ds['dataset']} ===")
        s = ds["summary"]
        for split in sorted(s.keys()):
            lines.append(f"  [{split}]")
            for key, block in sorted(s[split].items()):
                pmed = block["median_exact_p"]
                d = block["mean_delta"]
                lo, hi = block["mean_ci_low"], block["mean_ci_high"]
                lines.append(
                    f"    {key}: median_p={pmed:.4g}  "
                    f"mean_d={d:+.4f}  mean_CI~[{lo:+.4f},{hi:+.4f}]  "
                    f"p_seeds={['%.3g' % p for p in block['exact_p_values']]}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="SPC sample-level McNemar + bootstrap CI")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--noise_rates", nargs="+", type=float, default=[0.2])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs" / "spc_mcnemar"),
    )
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    all_results = []
    for cfg_path in args.configs:
        cfg_full = str(ROOT / cfg_path) if not Path(cfg_path).is_absolute() else cfg_path
        if not Path(cfg_full).exists():
            cfg_full = cfg_path
        print(f"== mcnemar {cfg_full} ==", flush=True)
        try:
            ds = run_dataset(
                cfg_full,
                list(args.seeds),
                list(args.noise_rates),
                int(args.n_boot),
                device,
            )
        except FileNotFoundError as e:
            print(f"skip: {e}", flush=True)
            continue
        all_results.append(ds)
        save_json(out_dir / f"mcnemar_{ds['dataset']}.json", ds)

    payload = {
        "seeds": args.seeds,
        "noise_rates": args.noise_rates,
        "n_boot": args.n_boot,
        "datasets": [d["dataset"] for d in all_results],
        "results": all_results,
    }
    save_json(out_dir / "spc_mcnemar.json", payload)
    txt = _fmt_summary(all_results)
    (out_dir / "spc_mcnemar.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
