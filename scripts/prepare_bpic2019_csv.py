"""Build a smaller UTF-8 CSV for BPI2019 SPC from the ICPM CSV dump."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "public_logs" / "bpic2019_csv" / "BPI_Challenge_2019.csv"
OUT = ROOT / "data" / "public_logs" / "bpic2019_3k.csv"


def main() -> int:
    print(f"Reading {SRC}", flush=True)
    df = pd.read_csv(SRC, encoding="latin-1", low_memory=False)
    rename = {
        "case concept:name": "case",
        "event concept:name": "activity",
        "event time:timestamp": "timestamp",
        "event org:resource": "resource",
    }
    miss = [k for k in rename if k not in df.columns]
    if miss:
        raise SystemExit(f"missing columns: {miss}; have={list(df.columns)}")
    df = df.rename(columns=rename)[["case", "activity", "timestamp", "resource"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["case", "activity", "timestamp"])
    cases = sorted(df["case"].astype(str).unique().tolist())
    rng = random.Random(42)
    keep = set(rng.sample(cases, min(3000, len(cases))))
    sub = df[df["case"].astype(str).isin(keep)].sort_values(["case", "timestamp"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(
        f"Wrote {OUT}: cases={sub['case'].nunique()} events={len(sub)} "
        f"acts={sub['activity'].nunique()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
