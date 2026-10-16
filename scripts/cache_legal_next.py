#!/usr/bin/env python
"""Cache DFG ∪ Petri legal-next masks for all prefixes."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json
from src.conformance.discovery import load_petri
from src.conformance.legal_next import (
    build_successor_index,
    cache_legal_masks,
    union_allowed_pairs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    # train cases for DFG
    with open(out_dir / "meta.json", "r", encoding="utf-8") as f:
        import json

        meta = json.load(f)

    # recover train cases from prefixes if cases.pkl exists
    cases_path = out_dir / "cases.pkl"
    if cases_path.exists():
        with open(cases_path, "rb") as f:
            all_cases = pickle.load(f)
        train_cases = {cid: all_cases[cid] for cid in split["train"] if cid in all_cases}
    else:
        # rebuild approximate train DFG from train prefix samples (last→label edges)
        train_cases = {}
        for s in datasets["train"]:
            train_cases.setdefault(s.case_id, [])
            # keep longest seen sequence per case via prefix+label
            seq = list(s.prefix_acts) + [s.label_act]
            if len(seq) > len(train_cases[s.case_id]):
                train_cases[s.case_id] = seq

    net, im, fm = load_petri(out_dir / "petri.pkl")
    pairs = union_allowed_pairs(net, bundle.activity2id, train_cases)
    succ = build_successor_index(pairs, bundle.num_activities)

    all_samples = list(datasets["train"]) + list(datasets["val"]) + list(datasets["test"])
    masks = cache_legal_masks(all_samples, succ, bundle.num_activities)

    # coverage: how often gold label is in mask
    hit = 0
    for s in datasets["test"]:
        m = masks[(s.case_id, s.prefix_len)]
        if m[s.label_id] > 0.5:
            hit += 1
    cover = hit / max(len(datasets["test"]), 1)

    with open(out_dir / "legal_next.pkl", "wb") as f:
        pickle.dump(
            {
                "pairs": pairs,
                "succ_index": succ,
                "masks": masks,
            },
            f,
        )
    save_json(
        out_dir / "legal_next_meta.json",
        {
            "n_pairs": len(pairs),
            "n_masks": len(masks),
            "test_label_coverage": cover,
            "avg_legal_per_prefix": float(
                sum(float(m.sum()) for m in masks.values()) / max(len(masks), 1)
            ),
        },
    )
    print(f"pairs={len(pairs)} masks={len(masks)} test_label_coverage={cover:.4f}")
    print(f"Wrote {out_dir / 'legal_next.pkl'}")


if __name__ == "__main__":
    main()
