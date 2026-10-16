#!/usr/bin/env python
"""End-to-end pipeline: prepare -> discover -> align -> train variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--quick", action="store_true", help="Skip full ablation; train core models only")
    args = ap.parse_args()
    py = args.python
    cfg = args.config

    run([py, "scripts/prepare_data.py", "--config", cfg])
    run([py, "scripts/discover_petri.py", "--config", cfg])
    run([py, "scripts/cache_alignments.py", "--config", cfg])
    run([py, "scripts/train_lstm.py", "--config", cfg])
    run([py, "scripts/train_hgnn.py", "--config", cfg, "--tag", "hgnn"])
    run(
        [
            py,
            "scripts/train_hgnn.py",
            "--config",
            cfg,
            "--use_graph_feat",
            "--tag",
            "hgnn_graph",
        ]
    )
    run(
        [
            py,
            "scripts/train_hgnn.py",
            "--config",
            cfg,
            "--use_graph_feat",
            "--use_node_feat",
            "--tag",
            "hgnn_graph_node",
        ]
    )
    run(
        [
            py,
            "scripts/train_hgnn.py",
            "--config",
            cfg,
            "--use_graph_feat",
            "--use_node_feat",
            "--use_extra_rels",
            "--tag",
            "hgnn_full",
        ]
    )
    if not args.quick:
        run([py, "scripts/run_experiments.py", "--config", cfg, "--python", py, "--skip_ablation"])


if __name__ == "__main__":
    main()
