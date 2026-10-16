"""Adaptive τ selection for SPC gate (val-set Acc budget → max PCP)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.conformance.legal_next import prediction_entropy

DEFAULT_PERCENTILES = [0.50, 0.60, 0.70, 0.80, 0.90]
DEFAULT_ACC_BUDGET = 0.005  # max allowed noise Acc drop vs Soft on val


@torch.no_grad()
def val_entropy_quantile(model, loader, device, q: float) -> float:
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


def build_noise_prefixes_for_split(cfg, out_dir, rate: float, split_name: str):
    """Inject CF noise into train/val/test cases and cache prefixes."""
    import pickle
    from pathlib import Path

    from src.data.prefixes import build_prefix_dataset
    from src.experiments import inject_control_flow_noise

    out_dir = Path(out_dir)
    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)

    id_attr = {"train": "train_ids", "val": "val_ids", "test": "test_ids"}[split_name]
    case_ids = list(getattr(split, id_attr))
    cases = {cid: bundle.cases[cid] for cid in case_ids}
    activities = list(bundle.activity2id.keys())
    noisy = inject_control_flow_noise(
        cases,
        activities,
        rate=rate,
        seed=int(cfg.get("seed", 42)) + int(rate * 1000) + hash(split_name) % 997,
        per_event=True,
    )
    samples = build_prefix_dataset(
        noisy,
        case_ids,
        bundle.activity2id,
        cfg.get("min_prefix_len", 1),
        max_prefix_len=cfg.get("max_prefix_len"),
    )
    noise_dir = out_dir / f"noise_{rate}"
    noise_dir.mkdir(parents=True, exist_ok=True)
    path = noise_dir / f"prefixes_{split_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    return samples, path


def _eval_samples(model, samples, out_dir, num_activities, batch_size, device, gate_tau):
    from scripts.run_spc_study import eval_transformer_on_samples

    return eval_transformer_on_samples(
        model,
        samples,
        out_dir,
        num_activities,
        batch_size,
        device,
        use_legal_mask=False,
        gate_tau=gate_tau,
    )


def select_tau_adaptive(
    model,
    *,
    val_loader,
    noisy_val_samples,
    out_dir,
    num_activities: int,
    batch_size: int,
    device,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    acc_budget: float = DEFAULT_ACC_BUDGET,
) -> Dict[str, Any]:
    """
    On validation noisy set: among τ = val-entropy percentiles,
    maximize PCP subject to Acc drop vs Soft ≤ acc_budget.
    Tie-break: milder gate (higher percentile), then higher Acc.
    """
    soft = _eval_samples(
        model,
        noisy_val_samples,
        out_dir,
        num_activities,
        batch_size,
        device,
        gate_tau=None,
    )
    soft_acc = float(soft["accuracy"])
    soft_leg = float(soft.get("legal_rate") or 0.0)

    candidates: List[Dict[str, Any]] = []
    for q in percentiles:
        tau = val_entropy_quantile(model, val_loader, device, float(q))
        m = _eval_samples(
            model,
            noisy_val_samples,
            out_dir,
            num_activities,
            batch_size,
            device,
            gate_tau=float(tau),
        )
        na = float(m["accuracy"])
        nl = float(m.get("legal_rate") or 0.0)
        d_nacc = na - soft_acc
        d_leg = nl - soft_leg
        candidates.append(
            {
                "percentile": float(q),
                "label": f"p{int(round(q * 100))}",
                "gate_tau": float(tau),
                "val_noise_accuracy": na,
                "val_noise_legal_rate": nl,
                "d_nacc": d_nacc,
                "d_leg": d_leg,
                "feasible": bool(d_nacc >= -float(acc_budget)),
            }
        )

    feasible = [c for c in candidates if c["feasible"]]
    if feasible:
        best = max(
            feasible,
            key=lambda c: (c["val_noise_legal_rate"], c["percentile"], c["d_nacc"]),
        )
        rule = f"max_val_legal_under_nAcc_drop<={acc_budget}"
    else:
        best = max(
            candidates,
            key=lambda c: (c["d_nacc"], c["val_noise_legal_rate"], c["percentile"]),
        )
        rule = "fallback_min_acc_violation"

    return {
        "selected_percentile": best["percentile"],
        "selected_label": best["label"],
        "gate_tau": best["gate_tau"],
        "acc_budget": float(acc_budget),
        "rule": rule,
        "soft_val_noise_accuracy": soft_acc,
        "soft_val_noise_legal_rate": soft_leg,
        "selected": best,
        "candidates": candidates,
    }
