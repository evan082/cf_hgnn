#!/usr/bin/env python
"""Build noisy test prefix sets for multiple noise rates (Bernoulli per-event)."""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json, set_seed
from src.conformance.alignment import AlignmentCache
from src.conformance.discovery import load_petri
from src.conformance.features import build_feature_maps
from src.data.prefixes import build_prefix_dataset
from src.experiments import inject_control_flow_noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "helpdesk.yaml"))
    ap.add_argument("--rates", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--rebuild", action="store_true", help="Delete existing noise dirs first")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)
    net, im, fm = load_petri(out_dir / "petri.pkl")
    activities = list(bundle.activity2id.keys())
    test_only = {cid: bundle.cases[cid] for cid in split.test_ids}

    summary = {}
    for rate in args.rates:
        noise_dir = out_dir / f"noise_{rate}"
        cache_dir = out_dir / f"alignment_cache_noise_{rate}"
        if args.rebuild:
            if noise_dir.exists():
                shutil.rmtree(noise_dir)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        noisy_test = inject_control_flow_noise(
            test_only,
            activities,
            rate=rate,
            seed=cfg.get("seed", 42) + int(rate * 1000),
            per_event=True,
        )
        samples = build_prefix_dataset(
            noisy_test, split.test_ids, bundle.activity2id, cfg.get("min_prefix_len", 1)
        )
        cache = AlignmentCache(cache_dir)
        diagnoses = cache.build_for_samples(samples, net, im, fm, show_progress=True)
        gmap, nmap = build_feature_maps(diagnoses, bundle.activity2id)

        noise_dir.mkdir(exist_ok=True)
        with open(noise_dir / "prefixes_test.pkl", "wb") as f:
            pickle.dump(samples, f)
        with open(noise_dir / "graph_feats.pkl", "wb") as f:
            pickle.dump(gmap, f)
        with open(noise_dir / "node_feats.pkl", "wb") as f:
            pickle.dump(nmap, f)
        with open(noise_dir / "diagnoses.pkl", "wb") as f:
            pickle.dump(diagnoses, f)

        fits = [getattr(d, "prefix_fitness", d.fitness) for d in diagnoses.values()]
        n_log = sum(1 for d in diagnoses.values() if sum(d.log_moves.values()) > 0)
        summary[str(rate)] = {
            "n_prefixes": len(samples),
            "prefix_fitness_mean": float(sum(fits) / len(fits)) if fits else None,
            "n_with_log_move": int(n_log),
            "frac_with_log_move": float(n_log / len(diagnoses)) if diagnoses else 0.0,
        }
        print(f"[noise={rate}] {summary[str(rate)]}")

    save_json(out_dir / "noise_ladder_summary.json", summary)
    print(f"Wrote {out_dir / 'noise_ladder_summary.json'}")


if __name__ == "__main__":
    main()
