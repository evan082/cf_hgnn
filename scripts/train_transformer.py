#!/usr/bin/env python
"""Train next-activity Transformer (± hard legal mask / soft legal / SPC gate)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json, set_seed
from src.data.prefixes import PrefixDataset, collate_pad
from src.models.transformer import TransformerNextActivity
from src.train.trainer import eval_transformer, fit_transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--use_legal_mask", action="store_true")
    ap.add_argument("--soft_legal_lambda", type=float, default=None)
    ap.add_argument("--gate_tau", type=float, default=None, help="Uncertainty gate; <0 disables")
    ap.add_argument(
        "--calibrate_gate",
        action="store_true",
        help="Calibrate gate_tau on val (adaptive Acc-budget by default)",
    )
    ap.add_argument(
        "--gate_mode",
        choices=["adaptive", "p70"],
        default=None,
        help="Gate calibration: adaptive (val Acc budget) or fixed p70",
    )
    ap.add_argument("--gate_acc_budget", type=float, default=None, help="Max val noise Acc drop vs Soft")
    ap.add_argument("--gate_noise_rate", type=float, default=None, help="Val noise rate for τ selection")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--embed_dim", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--label_smoothing", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.nhead is None:
        args.nhead = int(cfg.get("tf_nhead", 4))
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    device = torch.device(cfg.get("device", "cpu"))

    soft_lambda = (
        float(args.soft_legal_lambda)
        if args.soft_legal_lambda is not None
        else float(cfg.get("soft_legal_lambda", 0.0))
    )
    gate_tau = args.gate_tau
    if gate_tau is None and "gate_tau" in cfg:
        gate_tau = float(cfg["gate_tau"])
    calibrate_gate = bool(args.calibrate_gate)
    if (not args.calibrate_gate) and ("calibrate_gate" in cfg) and args.gate_tau is None:
        # only honor config when CLI did not force gate_tau
        calibrate_gate = bool(cfg.get("calibrate_gate", False))
    # Explicit --gate_tau < 0 means disable calibration
    if args.gate_tau is not None and float(args.gate_tau) < 0:
        calibrate_gate = False
        gate_tau = -1.0

    # label smoothing hurts legal-mask CE; default off when masking / soft-legal
    if args.label_smoothing is None:
        if "label_smoothing" in cfg:
            args.label_smoothing = float(cfg["label_smoothing"])
        else:
            args.label_smoothing = (
                0.0 if (args.use_legal_mask or soft_lambda > 0) else 0.05
            )

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    need_legal = True  # always load masks when present (for LegalRate metric)
    legal_masks = None
    ln_path = out_dir / "legal_next.pkl"
    if ln_path.exists():
        with open(ln_path, "rb") as f:
            legal_masks = pickle.load(f)["masks"]
    elif args.use_legal_mask or soft_lambda > 0 or (
        gate_tau is not None and gate_tau >= 0
    ) or calibrate_gate:
        raise FileNotFoundError(f"Missing {ln_path}; run scripts/cache_legal_next.py first")

    train_ds = PrefixDataset(
        datasets["train"], legal_masks=legal_masks, num_activities=bundle.num_activities
    )
    val_ds = PrefixDataset(
        datasets["val"], legal_masks=legal_masks, num_activities=bundle.num_activities
    )
    test_ds = PrefixDataset(
        datasets["test"], legal_masks=legal_masks, num_activities=bundle.num_activities
    )

    bs = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_pad)

    embed_dim = args.embed_dim or cfg.get("tf_embed_dim", 128)
    num_layers = args.num_layers or cfg.get("tf_num_layers", 4)
    hidden_dim = cfg.get("tf_hidden_dim", embed_dim)
    use_time = bool(cfg.get("use_time_feat", True))
    use_resource = bool(cfg.get("use_resource_feat", True))
    use_extra = bool(cfg.get("use_extra_attrs", True))
    has_time = bool(datasets["train"] and datasets["train"][0].prefix_times is not None)
    has_res = bool(datasets["train"] and datasets["train"][0].prefix_resource_ids is not None)
    has_attr = bool(datasets["train"] and datasets["train"][0].prefix_attr_ids is not None)
    time_dim = 2 if (use_time and has_time) else 0
    num_resources = 0
    attr_sizes = []
    if (out_dir / "meta.json").exists():
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        if use_resource and has_res:
            num_resources = int(meta.get("num_resources", 0))
        if use_extra and has_attr:
            attr_sizes = [int(x) for x in meta.get("attr_sizes", [])]

    model = TransformerNextActivity(
        num_activities=bundle.num_activities,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nhead=args.nhead,
        dropout=cfg.get("tf_dropout", 0.1),
        max_len=cfg.get("max_prefix_len", 512) or 512,
        time_feat_dim=time_dim,
        num_resources=num_resources,
        attr_sizes=attr_sizes,
        pool_mode=str(cfg.get("tf_pool_mode", "mean_last")),
        learned_pos=bool(cfg.get("tf_learned_pos", False)),
    ).to(device)

    if len(val_ds) == 0:
        val_loader = train_loader

    result = fit_transformer(
        model,
        train_loader,
        val_loader,
        device,
        epochs=cfg.get("epochs", 40),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
        patience=cfg.get("patience", 10),
        use_legal_mask=args.use_legal_mask,
        soft_legal_lambda=soft_lambda,
        gate_tau=None,  # calibrate after fit; don't gate during checkpointing
        label_smoothing=args.label_smoothing,
        early_stop_metric=cfg.get("early_stop_metric", "accuracy"),
    )

    # Calibrate gate on validation (SPC)
    gate_select_info = None
    gate_mode = args.gate_mode or str(cfg.get("gate_mode", "adaptive"))
    if calibrate_gate and not args.use_legal_mask:
        if gate_mode == "p70":
            val_m0 = eval_transformer(model, val_loader, device, use_legal_mask=False, gate_tau=None)
            gate_tau = float(val_m0.get("entropy_p70", 1.0))
            gate_select_info = {"mode": "p70", "gate_tau": gate_tau, "selected_label": "p70"}
            print(f"calibrated gate_tau={gate_tau:.4f} (val entropy p70)", flush=True)
        else:
            from scripts.tau_gate import (
                DEFAULT_ACC_BUDGET,
                DEFAULT_PERCENTILES,
                build_noise_prefixes_for_split,
                select_tau_adaptive,
            )

            acc_budget = (
                float(args.gate_acc_budget)
                if args.gate_acc_budget is not None
                else float(cfg.get("gate_acc_budget", DEFAULT_ACC_BUDGET))
            )
            noise_rate = (
                float(args.gate_noise_rate)
                if args.gate_noise_rate is not None
                else float(cfg.get("gate_noise_rate", 0.2))
            )
            percentiles = [float(x) for x in cfg.get("gate_percentiles", DEFAULT_PERCENTILES)]
            noise_path = out_dir / f"noise_{noise_rate}" / "prefixes_val.pkl"
            if noise_path.exists():
                with open(noise_path, "rb") as f:
                    noisy_val = pickle.load(f)
            else:
                noisy_val, _ = build_noise_prefixes_for_split(cfg, out_dir, noise_rate, "val")
            gate_select_info = select_tau_adaptive(
                model,
                val_loader=val_loader,
                noisy_val_samples=noisy_val,
                out_dir=out_dir,
                num_activities=bundle.num_activities,
                batch_size=bs,
                device=device,
                percentiles=percentiles,
                acc_budget=acc_budget,
            )
            gate_select_info["mode"] = "adaptive"
            gate_select_info["noise_rate"] = noise_rate
            gate_tau = float(gate_select_info["gate_tau"])
            print(
                f"adaptive gate_tau={gate_tau:.4f} "
                f"({gate_select_info['selected_label']}, rule={gate_select_info['rule']}, "
                f"budget={acc_budget})",
                flush=True,
            )

    test_gate = None
    if not args.use_legal_mask and gate_tau is not None and float(gate_tau) >= 0:
        test_gate = float(gate_tau)

    test_m = eval_transformer(
        model,
        test_loader,
        device,
        use_legal_mask=args.use_legal_mask,
        gate_tau=test_gate,
    )
    print("TEST", test_m, flush=True)

    if args.tag:
        tag = args.tag
    elif args.use_legal_mask:
        tag = "transformer_mask"
    elif soft_lambda > 0 and test_gate is not None:
        tag = "transformer_spc"
    elif soft_lambda > 0:
        tag = "transformer_soft"
    else:
        tag = "transformer"
    if args.seed is not None:
        tag = f"{tag}_seed{seed}"
    torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    save_json(
        out_dir / f"{tag}_metrics.json",
        {
            "val_best": result.best_val_acc,
            "test": test_m,
            "history": result.history,
            "tag": tag,
            "seed": seed,
            "use_legal_mask": args.use_legal_mask,
            "soft_legal_lambda": soft_lambda,
            "gate_tau": test_gate,
            "calibrate_gate": calibrate_gate,
            "gate_mode": gate_mode if calibrate_gate else None,
            "gate_select": gate_select_info,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "nhead": args.nhead,
        },
    )


if __name__ == "__main__":
    main()
