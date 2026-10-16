#!/usr/bin/env python
"""
Acc-first benchmark: LSTM / HGNN / Transformer / Transformer+LegalMask on clean test.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json


def _run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--skip_lstm", action="store_true")
    ap.add_argument("--skip_hgnn", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    report_dir = out_dir / "acc_benchmark"
    report_dir.mkdir(exist_ok=True)

    # legal next cache
    if not (out_dir / "legal_next.pkl").exists() or args.force:
        _run([args.python, "scripts/cache_legal_next.py", "--config", args.config])

    rows = []
    for seed in args.seeds:
        jobs = []
        if not args.skip_lstm:
            jobs.append(("lstm", ["scripts/train_lstm.py", "--config", args.config, "--seed", str(seed), "--tag", "lstm"]))
        if not args.skip_hgnn:
            jobs.append(("hgnn", ["scripts/train_hgnn.py", "--config", args.config, "--seed", str(seed), "--tag", "hgnn"]))
        jobs.append(
            (
                "transformer",
                [
                    "scripts/train_transformer.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--tag",
                    "transformer",
                ],
            )
        )
        jobs.append(
            (
                "transformer_mask",
                [
                    "scripts/train_transformer.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--use_legal_mask",
                    "--tag",
                    "transformer_mask",
                ],
            )
        )

        for name, cmd_tail in jobs:
            tag = f"{name}_seed{seed}"
            metrics_path = out_dir / f"{tag}_metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"reuse {metrics_path.name}")
            else:
                _run([args.python, *cmd_tail])
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            test = m.get("test", {})
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "accuracy": test.get("accuracy"),
                    "accuracy_macro_k": test.get("accuracy_macro_k"),
                    "f1_macro": test.get("f1_macro"),
                    "f1_weighted": test.get("f1_weighted"),
                    "val_best": m.get("val_best"),
                }
            )
            print(
                f"[seed={seed} {name}] acc={test.get('accuracy')} "
                f"acc_macro_k={test.get('accuracy_macro_k')}"
            )

    # aggregate
    def _agg(key: str) -> Dict:
        by_model: Dict[str, List[float]] = {}
        for r in rows:
            if r.get(key) is not None:
                by_model.setdefault(r["model"], []).append(float(r[key]))
        out = {}
        for model, vals in by_model.items():
            arr = np.asarray(vals, dtype=float)
            out[model] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "values": [float(x) for x in arr],
            }
        return out

    agg = _agg("accuracy")
    agg_macro_k = _agg("accuracy_macro_k")

    lit = {
        "ProcessTransformer_Helpdesk": 0.8563,
        "ProcessTransformer_BPIC2012": 0.8520,
        "DFG_GAT_Helpdesk": 0.862,
        "DFG_GAT_BPIC2012": 0.875,
        "note": "Literature numbers from published papers; protocols may differ.",
    }

    out = {
        "dataset": cfg.get("dataset"),
        "per_run": rows,
        "mean_std": agg,
        "mean_std_macro_k": agg_macro_k,
        "literature_reference": lit,
    }
    if (out_dir / "legal_next_meta.json").exists():
        out["legal_next_meta"] = json.loads((out_dir / "legal_next_meta.json").read_text(encoding="utf-8"))

    save_json(report_dir / "acc_benchmark.json", out)

    lines = [f"dataset={cfg.get('dataset')}", "", "model\tacc_micro±std\tacc_macro_k±std"]
    for model in sorted(set(agg) | set(agg_macro_k)):
        a = agg.get(model, {})
        mk = agg_macro_k.get(model, {})
        a_s = f"{a['mean']:.4f}±{a['std']:.4f}" if a else "n/a"
        mk_s = f"{mk['mean']:.4f}±{mk['std']:.4f}" if mk else "n/a"
        lines.append(f"{model}\t{a_s}\t{mk_s}")
    lines.append("")
    lines.append("literature ProcessTransformer Acc is mean-over-prefix-length (≈ acc_macro_k)")
    for k, v in lit.items():
        if k == "note":
            lines.append(f"note\t{v}")
        else:
            lines.append(f"{k}\t{v}")
    text = "\n".join(lines)
    (report_dir / "acc_benchmark.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {report_dir / 'acc_benchmark.json'}")


if __name__ == "__main__":
    main()
