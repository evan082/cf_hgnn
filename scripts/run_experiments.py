#!/usr/bin/env python
"""Step 10: run ablation suite, noise robustness, and explainability dumps."""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json, set_seed
from src.experiments import (
    ablation_configs,
    explain_prefix_diagnosis,
    inject_control_flow_noise,
)
from src.data.prefixes import build_prefix_dataset
from src.conformance.alignment import AlignmentCache
from src.conformance.discovery import load_petri
from src.conformance.features import build_feature_maps


def _run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def run_ablation(cfg_path: str, python: str):
    cfgs = ablation_configs()
    summary = []
    for c in cfgs:
        if c["model"] == "lstm":
            cmd = [python, "scripts/train_lstm.py", "--config", cfg_path]
            if c["graph_feat"]:
                cmd.append("--use_graph_feat")
        else:
            cmd = [python, "scripts/train_hgnn.py", "--config", cfg_path, "--tag", c["name"]]
            if c["graph_feat"]:
                cmd.append("--use_graph_feat")
            if c["node_feat"]:
                cmd.append("--use_node_feat")
            if c["extra_rels"]:
                cmd.append("--use_extra_rels")
        _run(cmd)
        summary.append(c["name"])
    return summary


def run_noise_experiment(cfg, out_dir: Path, rates=(0.05, 0.1, 0.2)):
    """Rebuild noisy test prefixes, re-align, and leave artifacts for re-eval."""
    set_seed(cfg.get("seed", 42))
    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)
    net, im, fm = load_petri(out_dir / "petri.pkl")

    results = {}
    activities = list(bundle.activity2id.keys())
    for rate in rates:
        noisy_cases = dict(bundle.cases)
        test_only = {cid: bundle.cases[cid] for cid in split.test_ids}
        noisy_test = inject_control_flow_noise(
            test_only, activities, rate=rate, seed=cfg.get("seed", 42), per_event=True
        )
        noisy_cases.update(noisy_test)

        samples = build_prefix_dataset(
            noisy_cases, split.test_ids, bundle.activity2id, cfg.get("min_prefix_len", 1)
        )
        cache = AlignmentCache(out_dir / f"alignment_cache_noise_{rate}")
        diagnoses = cache.build_for_samples(samples, net, im, fm, show_progress=True)
        gmap, nmap = build_feature_maps(diagnoses, bundle.activity2id)

        noise_dir = out_dir / f"noise_{rate}"
        noise_dir.mkdir(exist_ok=True)
        with open(noise_dir / "prefixes_test.pkl", "wb") as f:
            pickle.dump(samples, f)
        with open(noise_dir / "graph_feats.pkl", "wb") as f:
            pickle.dump(gmap, f)
        with open(noise_dir / "node_feats.pkl", "wb") as f:
            pickle.dump(nmap, f)
        with open(noise_dir / "diagnoses.pkl", "wb") as f:
            pickle.dump(diagnoses, f)

        fits = [d.fitness for d in diagnoses.values()]
        results[str(rate)] = {
            "n_prefixes": len(samples),
            "fitness_mean": float(sum(fits) / len(fits)) if fits else None,
            "n_low_fitness": int(sum(1 for x in fits if x < 0.9)),
        }
        print(f"[noise={rate}] prefixes={len(samples)} fitness_mean={results[str(rate)]['fitness_mean']}")
    save_json(out_dir / "noise_summary.json", results)
    return results


def run_explainability(out_dir: Path, top_k: int = 20):
    """Dump worst-fitness prefixes with diagnosis summaries for qualitative analysis."""
    with open(out_dir / "diagnoses.pkl", "rb") as f:
        diagnoses = pickle.load(f)
    items = sorted(diagnoses.values(), key=lambda d: d.fitness)
    explanations = [explain_prefix_diagnosis(d) for d in items[:top_k]]
    save_json(out_dir / "explanations_low_fitness.json", explanations)
    print(f"Wrote {len(explanations)} explanations -> explanations_low_fitness.json")
    return explanations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip_ablation", action="store_true")
    ap.add_argument("--skip_noise", action="store_true")
    ap.add_argument("--skip_explain", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")

    if not args.skip_ablation:
        names = run_ablation(args.config, args.python)
        print("Ablation finished:", names)

    if not args.skip_noise:
        run_noise_experiment(cfg, out_dir)

    if not args.skip_explain:
        run_explainability(out_dir)


if __name__ == "__main__":
    main()
