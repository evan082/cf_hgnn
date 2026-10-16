"""Queue sequential SPC studies. Usage:
  python scripts/run_spc_queue.py configs/a.yaml configs/b.yaml ...
Skips configs whose spc_study.json already exists unless --force.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for cfg in args.configs:
        cfg_path = Path(cfg)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        # infer output dir from yaml dataset field lightly
        text = cfg_path.read_text(encoding="utf-8")
        dataset = None
        for line in text.splitlines():
            if line.strip().startswith("dataset:"):
                dataset = line.split(":", 1)[1].strip()
                break
        report = ROOT / "outputs" / (dataset or "unknown") / "spc_study" / "spc_study.json"
        if report.exists() and not args.force:
            print(f"SKIP done {cfg_path.name} -> {report}", flush=True)
            continue
        cmd = [
            args.python,
            str(ROOT / "scripts" / "run_spc_study.py"),
            "--config",
            str(cfg_path),
            "--seeds",
            *[str(s) for s in args.seeds],
        ]
        if args.force:
            cmd.append("--force")
        print(">>", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"FAILED {cfg_path.name} rc={rc}", flush=True)
            return rc
    print("QUEUE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
