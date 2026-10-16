#!/usr/bin/env python
"""Train HGNN with optional conformance features."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json, set_seed
from src.data.prefixes import PrefixDataset, collate_pad
from src.models.hgnn import HGNNNextActivity
from src.train.trainer import eval_hgnn, fit_hgnn
from src.conformance.features import GRAPH_FEAT_DIM, NODE_FEAT_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--use_graph_feat", action="store_true")
    ap.add_argument("--use_node_feat", action="store_true")
    ap.add_argument("--use_extra_rels", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train_noise_rate", type=float, default=0.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    device = torch.device(cfg.get("device", "cpu"))

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    gmap, nmap, diagnoses, allowed = {}, {}, {}, None
    if args.use_graph_feat and (out_dir / "graph_feats.pkl").exists():
        with open(out_dir / "graph_feats.pkl", "rb") as f:
            gmap = pickle.load(f)
    if args.use_node_feat and (out_dir / "node_feats.pkl").exists():
        with open(out_dir / "node_feats.pkl", "rb") as f:
            nmap = pickle.load(f)
    if (out_dir / "diagnoses.pkl").exists():
        with open(out_dir / "diagnoses.pkl", "rb") as f:
            diagnoses = pickle.load(f)
    if args.use_extra_rels and (out_dir / "allowed_pairs.pkl").exists():
        with open(out_dir / "allowed_pairs.pkl", "rb") as f:
            allowed = pickle.load(f)

    train_ds = PrefixDataset(
        datasets["train"], graph_feats=gmap, node_feats=nmap, num_activities=bundle.num_activities
    )
    val_ds = PrefixDataset(
        datasets["val"], graph_feats=gmap, node_feats=nmap, num_activities=bundle.num_activities
    )
    test_ds = PrefixDataset(
        datasets["test"], graph_feats=gmap, node_feats=nmap, num_activities=bundle.num_activities
    )

    bs = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_pad)

    rels = ["forward", "backward", "repeat"]
    if args.use_extra_rels:
        rels += ["allowed", "deviation"]

    model = HGNNNextActivity(
        num_activities=bundle.num_activities,
        embed_dim=cfg.get("embed_dim", 64),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        relation_types=rels,
        graph_feat_dim=GRAPH_FEAT_DIM if args.use_graph_feat else 0,
        node_conf_dim=NODE_FEAT_DIM if args.use_node_feat else 0,
    ).to(device)

    result = fit_hgnn(
        model,
        train_loader,
        val_loader,
        device,
        epochs=cfg.get("epochs", 30),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
        patience=cfg.get("patience", 8),
        allowed_pairs=allowed,
        diagnoses=diagnoses,
        activity2id=bundle.activity2id,
        use_extra_rels=args.use_extra_rels,
        aug_rate=args.train_noise_rate,
        seed=seed,
    )
    test_m = eval_hgnn(
        model,
        test_loader,
        device,
        allowed_pairs=allowed,
        diagnoses=diagnoses,
        activity2id=bundle.activity2id,
        use_extra_rels=args.use_extra_rels,
    )
    print("TEST", test_m)

    tag = args.tag or (
        "hgnn"
        + ("_g" if args.use_graph_feat else "")
        + ("_n" if args.use_node_feat else "")
        + ("_rel" if args.use_extra_rels else "")
    )
    if args.seed is not None:
        tag = f"{tag}_seed{seed}"
    if args.train_noise_rate > 0:
        tag = f"{tag}_aug{args.train_noise_rate:g}"
    torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    save_json(
        out_dir / f"{tag}_metrics.json",
        {
            "val_best": result.best_val_acc,
            "test": test_m,
            "history": result.history,
            "tag": tag,
            "seed": seed,
            "train_noise_rate": args.train_noise_rate,
        },
    )


if __name__ == "__main__":
    main()
