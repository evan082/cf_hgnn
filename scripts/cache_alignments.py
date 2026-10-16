#!/usr/bin/env python
"""Step 7: align all prefixes and write offline cache + feature maps."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config
from src.conformance.alignment import AlignmentCache
from src.conformance.discovery import load_petri
from src.conformance.features import build_feature_maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    net, im, fm = load_petri(out_dir / "petri.pkl")
    cache_dir = out_dir / "alignment_cache"
    cache = AlignmentCache(cache_dir)

    all_samples = []
    for sp in args.splits:
        all_samples.extend(datasets[sp])

    diagnoses = cache.build_for_samples(all_samples, net, im, fm, show_progress=True)
    gmap, nmap = build_feature_maps(diagnoses, bundle.activity2id)

    with open(out_dir / "diagnoses.pkl", "wb") as f:
        pickle.dump(diagnoses, f)
    with open(out_dir / "graph_feats.pkl", "wb") as f:
        pickle.dump(gmap, f)
    with open(out_dir / "node_feats.pkl", "wb") as f:
        pickle.dump(nmap, f)

    # quick stats
    fits = [d.fitness for d in diagnoses.values()]
    print(f"Aligned {len(diagnoses)} prefixes -> {cache_dir}")
    if fits:
        print(
            f"fitness mean={sum(fits)/len(fits):.4f} "
            f"min={min(fits):.4f} max={max(fits):.4f}"
        )


if __name__ == "__main__":
    main()
