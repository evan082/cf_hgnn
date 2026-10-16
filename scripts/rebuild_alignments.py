#!/usr/bin/env python
"""Clear alignment cache and rebuild diagnoses/features with prefix_fitness."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "helpdesk.yaml"))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    # resolve dataset dir from config lightly
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    cache = out_dir / "alignment_cache"
    if cache.exists():
        shutil.rmtree(cache)
        print(f"Removed {cache}")
    for name in ("diagnoses.pkl", "graph_feats.pkl", "node_feats.pkl"):
        p = out_dir / name
        if p.exists():
            p.unlink()
    subprocess.check_call(
        [args.python, str(ROOT / "scripts" / "cache_alignments.py"), "--config", args.config],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
