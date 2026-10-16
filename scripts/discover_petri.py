#!/usr/bin/env python
"""Step 6: discover Petri net from training cases."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config
from src.conformance.discovery import discover_petri_net, petri_allowed_df_pairs, save_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)

    train_cases = {cid: bundle.cases[cid] for cid in split.train_ids}
    net, im, fm = discover_petri_net(
        train_cases, noise_threshold=cfg.get("discovery_noise_threshold", 0.2)
    )
    petri_path = out_dir / "petri.pkl"
    save_petri(petri_path, net, im, fm)

    allowed = petri_allowed_df_pairs(net, bundle.activity2id, train_cases=train_cases)
    with open(out_dir / "allowed_pairs.pkl", "wb") as f:
        pickle.dump(allowed, f)

    print(f"Discovered Petri net -> {petri_path}")
    print(f"transitions={len(net.transitions)} places={len(net.places)} allowed_pairs={len(allowed)}")


if __name__ == "__main__":
    main()
