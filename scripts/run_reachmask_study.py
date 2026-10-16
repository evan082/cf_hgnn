# -*- coding: utf-8 -*-
"""ReachMask baseline: SoftLegal weights + always Petri enabled-transition mask."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spc_study import (  # noqa: E402
    _build_model,
    _build_noise_prefixes_fast,
    _eval_split,
)
from scripts.utils import load_config, set_seed  # noqa: E402
from src.conformance.discovery import load_petri  # noqa: E402
from src.conformance.reachability_mask import cache_reachability_masks  # noqa: E402
from src.data.prefixes import PrefixDataset, collate_pad  # noqa: E402
from src.train.trainer import eval_transformer  # noqa: E402


def _id2act(activity2id):
    return {int(v): k for k, v in activity2id.items()}


def _eval_reach(model, samples, reach_masks, num_activities, batch_size, device):
    return _eval_split(
        model,
        samples,
        reach_masks,
        num_activities,
        batch_size,
        device,
        use_legal_mask=True,
        gate_tau=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/helpdesk_spc.yaml",
            "configs/traffic_spc.yaml",
            "configs/sepsis_spc.yaml",
            "configs/bpic2012_spc.yaml",
        ],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--noise_rate", type=float, default=0.2)
    ap.add_argument("--force_cache", action="store_true")
    args = ap.parse_args()

    out_root = ROOT / "outputs" / "reachmask_study"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        ds = cfg.get("dataset", "run")
        out_dir = ROOT / cfg.get("output_dir", "outputs") / ds
        device = torch.device(cfg.get("device", "cpu"))
        print(f"=== ReachMask {ds} ===", flush=True)

        with open(out_dir / "bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        with open(out_dir / "prefixes.pkl", "rb") as f:
            datasets = pickle.load(f)
        with open(out_dir / "legal_next.pkl", "rb") as f:
            legal_obj = pickle.load(f)
        succ_index = legal_obj.get("succ_index")
        net, im, _fm = load_petri(out_dir / "petri.pkl")
        activity2id = bundle.activity2id if hasattr(bundle, "activity2id") else bundle["activity2id"]
        id2activity = _id2act(activity2id)
        num_act = len(activity2id)
        bs = int(cfg.get("batch_size", 32))

        cache_path = out_dir / "legal_next_reach.pkl"
        if cache_path.exists() and not args.force_cache:
            blob = pickle.load(open(cache_path, "rb"))
            reach_clean = blob["masks"]
            print("reuse reach cache", blob.get("stats"), flush=True)
        else:
            print("building clean reachability masks...", flush=True)
            all_s = list(datasets["train"]) + list(datasets["val"]) + list(datasets["test"])
            reach_clean, stats = cache_reachability_masks(
                all_s, net, im, activity2id, id2activity, succ_index=succ_index
            )
            pickle.dump({"masks": reach_clean, "stats": stats}, open(cache_path, "wb"))
            print("reach stats", stats, flush=True)

        # noisy prefixes + reach masks
        noise_dir = out_dir / f"noise_{args.noise_rate}"
        noise_pref = noise_dir / "prefixes_test.pkl"
        if not noise_pref.exists():
            _build_noise_prefixes_fast(cfg, out_dir, args.noise_rate)
        with open(noise_pref, "rb") as f:
            noisy_test = pickle.load(f)
        print(f"building noisy reach masks n={len(noisy_test)}...", flush=True)
        reach_noise, noise_stats = cache_reachability_masks(
            noisy_test, net, im, activity2id, id2activity, succ_index=succ_index
        )
        print("noise reach stats", noise_stats, flush=True)

        study_path = out_dir / "spc_study" / "spc_study.json"
        study = json.loads(study_path.read_text(encoding="utf-8")) if study_path.exists() else None

        for seed in args.seeds:
            soft_pt = out_dir / f"soft_seed{seed}.pt"
            if not soft_pt.exists():
                print("skip", soft_pt, flush=True)
                continue
            set_seed(seed)
            model = _build_model(cfg, bundle, datasets, out_dir, device)
            ckpt = torch.load(soft_pt, map_location=device)
            state = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
            model.load_state_dict(state)
            model.eval()

            clean_m = _eval_reach(model, datasets["test"], reach_clean, num_act, bs, device)
            noise_m = _eval_reach(model, noisy_test, reach_noise, num_act, bs, device)
            row = {
                "dataset": ds,
                "seed": seed,
                "variant": "reachmask",
                "clean_accuracy": clean_m.get("accuracy"),
                "clean_legal_rate": clean_m.get("legal_rate"),
                "noise_accuracy": noise_m.get("accuracy"),
                "noise_legal_rate": noise_m.get("legal_rate"),
                "noise_reach_fail_rate": noise_stats.get("replay_fail_rate"),
            }
            if study:
                for pr in study.get("per_run", []):
                    if pr.get("seed") != seed:
                        continue
                    v = pr.get("variant")
                    if v in ("base", "soft", "hardmask", "spc"):
                        row[f"{v}_noise_acc"] = pr.get("noise_0.2_accuracy")
                        row[f"{v}_noise_pcp"] = pr.get("noise_0.2_legal_rate")
            rows.append(row)
            print(
                f"{ds} seed={seed} ReachMask noise Acc={row['noise_accuracy']:.3f} "
                f"PCP={row['noise_legal_rate']:.3f}",
                flush=True,
            )

    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {}
    for ds, rs in by.items():
        summary[ds] = {
            "reach_noise_acc": ms([r["noise_accuracy"] for r in rs]),
            "reach_noise_pcp": ms([r["noise_legal_rate"] for r in rs]),
            "hard_noise_acc": ms([r.get("hardmask_noise_acc") for r in rs]),
            "hard_noise_pcp": ms([r.get("hardmask_noise_pcp") for r in rs]),
            "spc_noise_acc": ms([r.get("spc_noise_acc") for r in rs]),
            "spc_noise_pcp": ms([r.get("spc_noise_pcp") for r in rs]),
            "soft_noise_pcp": ms([r.get("soft_noise_pcp") for r in rs]),
            "n": len(rs),
        }

    payload = {"rows": rows, "summary": summary, "noise_rate": args.noise_rate}
    (out_root / "reachmask_study.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # tsv
    lines = ["dataset\treach_acc\treach_pcp\thard_acc\thard_pcp\tspc_acc\tspc_pcp\tsoft_pcp"]
    for ds, s in summary.items():
        def g(k, field="mean"):
            return "" if s[k] is None else f"{s[k][field]:.3f}"

        lines.append(
            "\t".join(
                [
                    ds,
                    g("reach_noise_acc"),
                    g("reach_noise_pcp"),
                    g("hard_noise_acc"),
                    g("hard_noise_pcp"),
                    g("spc_noise_acc"),
                    g("spc_noise_pcp"),
                    g("soft_noise_pcp"),
                ]
            )
        )
    (out_root / "reachmask_summary.tsv").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_root / "reachmask_study.json")


if __name__ == "__main__":
    main()
