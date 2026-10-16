#!/usr/bin/env python
"""Multi-seed ProcessTransformer-protocol Acc report (helpdesk_pt)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json


def _run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "helpdesk_pt.yaml"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prepare", action="store_true", help="Re-run prepare_data first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    report_dir = out_dir / "acc_pt_protocol"
    report_dir.mkdir(parents=True, exist_ok=True)

    variant = str(cfg.get("model_variant", "pt_clone"))
    if args.prepare or not (out_dir / "prefixes.pkl").exists():
        _run([args.python, "scripts/prepare_data.py", "--config", args.config])

    rows = []
    for seed in args.seeds:
        if variant == "pt_clone":
            tag = f"pt_clone_seed{seed}"
            train_cmd = [
                args.python,
                "scripts/train_pt_clone.py",
                "--config",
                args.config,
                "--seed",
                str(seed),
            ]
        else:
            tag = f"transformer_seed{seed}"
            train_cmd = [
                args.python,
                "scripts/train_transformer.py",
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--tag",
                "transformer",
            ]

        metrics_path = out_dir / f"{tag}_metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"reuse {metrics_path.name}", flush=True)
        else:
            _run(train_cmd)
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = m.get("test", {})
        rows.append(
            {
                "seed": seed,
                "accuracy": test.get("accuracy"),
                "accuracy_macro_k": test.get("accuracy_macro_k"),
                "accuracy_pt_quirk": test.get("accuracy_pt_quirk"),
                "f1_weighted": test.get("f1_weighted"),
                "val_best": m.get("val_best", m.get("best_train_acc")),
                "embed_dim": m.get("embed_dim", cfg.get("tf_embed_dim")),
                "num_layers": m.get("num_layers", cfg.get("tf_num_layers")),
            }
        )
        print(
            f"[seed={seed}] micro={test.get('accuracy')} "
            f"macro_k={test.get('accuracy_macro_k')} "
            f"pt_quirk={test.get('accuracy_pt_quirk')}",
            flush=True,
        )

    def _stats(key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "values": []}
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": vals,
        }

    out = {
        "dataset": cfg.get("dataset"),
        "protocol": {
            "split_mode": cfg.get("split_mode"),
            "train_ratio": cfg.get("train_ratio"),
            "val_ratio": cfg.get("val_ratio"),
            "test_ratio": cfg.get("test_ratio"),
            "normalize_activity_names": cfg.get("normalize_activity_names", False),
            "features": "activity_only",
            "padding": "pre-pad + GAP includes PAD (PT)",
            "metric": "accuracy_macro_k = mean Acc over prefix lengths",
            "metric_pt_quirk": "PT next_activity.py appends last-k Acc again before mean",
            "checkpoint": "train_accuracy (PT-style)",
            "literature_ProcessTransformer_Helpdesk": 0.8563,
        },
        "model": {
            "variant": variant,
            "tf_embed_dim": cfg.get("tf_embed_dim"),
            "tf_num_layers": cfg.get("tf_num_layers"),
            "tf_nhead": cfg.get("tf_nhead"),
            "tf_ff_dim": cfg.get("tf_ff_dim"),
        },
        "per_run": rows,
        "mean_std_micro": _stats("accuracy"),
        "mean_std_macro_k": _stats("accuracy_macro_k"),
        "mean_std_pt_quirk": _stats("accuracy_pt_quirk"),
    }
    save_json(report_dir / "acc_pt_protocol.json", out)

    def _fmt(stats):
        if stats["mean"] is None:
            return "n/a"
        return f"{stats['mean']:.4f}±{stats['std']:.4f}"

    text = (
        f"helpdesk_pt protocol ({variant})\n"
        f"micro Acc:    {_fmt(out['mean_std_micro'])}\n"
        f"macro_k Acc:  {_fmt(out['mean_std_macro_k'])}\n"
        f"PT quirk Acc: {_fmt(out['mean_std_pt_quirk'])}\n"
        f"literature:   0.8563 (ProcessTransformer Helpdesk)\n"
    )
    (report_dir / "acc_pt_protocol.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Wrote {report_dir / 'acc_pt_protocol.json'}", flush=True)


if __name__ == "__main__":
    main()
