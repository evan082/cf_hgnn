#!/usr/bin/env python
"""Steps 1–2: load event log, split cases, build prefix–label datasets."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json, set_seed
from src.data.log_io import load_event_log
from src.data.prefixes import build_prefix_dataset
from src.data.splits import split_cases

# Helpdesk / generic categorical event attributes (exclude id/time/activity)
DEFAULT_ATTR_COLS = [
    "customer",
    "product",
    "responsible_section",
    "seriousness_2",
    "service_level",
    "service_type",
    "support_section",
    "workgroup",
]


def _build_resource_maps(df):
    resource2id = {"<pad>": 0}
    case_resources = None
    if "resource" not in df.columns and "Resource" in df.columns:
        df = df.rename(columns={"Resource": "resource"})
    if "resource" in df.columns:
        df["resource"] = df["resource"].astype(str).fillna("<unk>")
        for r in sorted(df["resource"].unique()):
            if r not in resource2id:
                resource2id[r] = len(resource2id)
        case_resources = {}
        for case_id, g in df.groupby("case", sort=False):
            case_resources[str(case_id)] = [resource2id[str(x)] for x in g["resource"].tolist()]
    return df, resource2id, case_resources


def _build_attr_maps(df, attr_cols: List[str]):
    """Return (attr_cols_used, attr_vocabs, attr_sizes, case_attrs)."""
    used = [c for c in attr_cols if c in df.columns]
    if not used:
        return [], {}, [], None

    attr_vocabs: Dict[str, Dict[str, int]] = {}
    attr_sizes: List[int] = []
    for col in used:
        df[col] = df[col].astype(str).fillna("<unk>")
        vocab = {"<pad>": 0}
        for v in sorted(df[col].unique()):
            if v not in vocab:
                vocab[v] = len(vocab)
        attr_vocabs[col] = vocab
        attr_sizes.append(len(vocab))

    case_attrs: Dict[str, List[List[int]]] = {}
    for case_id, g in df.groupby("case", sort=False):
        rows = []
        for _, row in g.iterrows():
            rows.append([attr_vocabs[c][str(row[c])] for c in used])
        case_attrs[str(case_id)] = rows
    return used, attr_vocabs, attr_sizes, case_attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data", default=None, help="Override data path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    data_path = Path(args.data or cfg["data_path"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    bundle = load_event_log(
        data_path,
        case_col=cfg.get("case_col"),
        activity_col=cfg.get("activity_col"),
        timestamp_col=cfg.get("timestamp_col"),
        normalize_activity_names=bool(cfg.get("normalize_activity_names", False)),
    )

    # Optional: keep only activities with given prefixes (e.g. BPIC2012 A_/O_/W_ subsets).
    act_prefixes = cfg.get("activity_prefixes")
    if cfg.get("activity_prefix"):
        act_prefixes = [cfg.get("activity_prefix")]
    if act_prefixes:
        prefs = [str(p) for p in act_prefixes]
        mask = bundle.df["activity"].astype(str).map(lambda a: any(a.startswith(p) for p in prefs))
        bundle.df = bundle.df.loc[mask].reset_index(drop=True)
        activities = sorted(bundle.df["activity"].unique().tolist())
        bundle.activity2id = {a: i for i, a in enumerate(activities)}
        bundle.id2activity = {i: a for a, i in bundle.activity2id.items()}
        cases = {}
        for case_id, g in bundle.df.groupby("case", sort=False):
            seq = g["activity"].tolist()
            if seq:
                cases[str(case_id)] = seq
        bundle.cases = cases
        print(
            f"Filtered activity_prefixes={prefs}: "
            f"cases={len(bundle.cases)} events={len(bundle.df)} acts={bundle.num_activities}"
        )

    max_cases = cfg.get("max_cases")
    if max_cases is not None and int(max_cases) < len(bundle.cases):
        import random

        rng = random.Random(cfg.get("seed", 42))
        keep = set(rng.sample(sorted(bundle.cases.keys()), int(max_cases)))
        bundle.cases = {cid: seq for cid, seq in bundle.cases.items() if cid in keep}
        bundle.df = bundle.df[bundle.df["case"].isin(keep)].reset_index(drop=True)
        print(f"Subsampled to max_cases={max_cases} (kept {len(bundle.cases)} cases)")

    split_mode = cfg.get("split_mode", "random")
    case_ids_for_split = list(bundle.case_ids)
    # ProcessTransformer: case order = first appearance in the raw CSV
    if split_mode == "sequential":
        import pandas as pd

        raw = pd.read_csv(data_path)
        case_col = cfg.get("case_col") or "Case ID"
        if case_col not in raw.columns:
            # try common aliases
            for c in raw.columns:
                if c.lower().replace(" ", "") in {"caseid", "case:concept:name"}:
                    case_col = c
                    break
        case_ids_for_split = raw[case_col].astype(str).drop_duplicates().tolist()
        # keep only cases present after load/subsample
        keep = set(bundle.cases.keys())
        case_ids_for_split = [c for c in case_ids_for_split if c in keep]

    split = split_cases(
        case_ids_for_split,
        train_ratio=cfg.get("train_ratio", 0.64),
        val_ratio=cfg.get("val_ratio", 0.16),
        test_ratio=cfg.get("test_ratio", 0.20),
        seed=cfg.get("seed", 42),
        mode=split_mode,
    )

    min_len = cfg.get("min_prefix_len", 1)
    max_len = cfg.get("max_prefix_len")

    bundle.df, resource2id, case_resources = _build_resource_maps(bundle.df)

    attr_cols_cfg = cfg.get("attr_cols", DEFAULT_ATTR_COLS)
    if cfg.get("use_extra_attrs", True) is False:
        attr_cols_cfg = []
    used_attrs, attr_vocabs, attr_sizes, case_attrs = _build_attr_maps(bundle.df, list(attr_cols_cfg))

    case_timestamps = {}
    for case_id, g in bundle.df.groupby("case", sort=False):
        case_timestamps[str(case_id)] = g["timestamp"].tolist()

    datasets = {
        "train": build_prefix_dataset(
            bundle.cases,
            split.train_ids,
            bundle.activity2id,
            min_len,
            max_len,
            case_timestamps=case_timestamps,
            case_resources=case_resources,
            case_attrs=case_attrs,
        ),
        "val": build_prefix_dataset(
            bundle.cases,
            split.val_ids,
            bundle.activity2id,
            min_len,
            max_len,
            case_timestamps=case_timestamps,
            case_resources=case_resources,
            case_attrs=case_attrs,
        ),
        "test": build_prefix_dataset(
            bundle.cases,
            split.test_ids,
            bundle.activity2id,
            min_len,
            max_len,
            case_timestamps=case_timestamps,
            case_resources=case_resources,
            case_attrs=case_attrs,
        ),
    }

    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "data_path": str(data_path),
        "num_cases": len(bundle.cases),
        "num_events": len(bundle.df),
        "num_activities": bundle.num_activities,
        "num_resources": len(resource2id),
        "attr_cols": used_attrs,
        "attr_sizes": attr_sizes,
        "attr_vocabs": attr_vocabs,
        "activity2id": bundle.activity2id,
        "resource2id": resource2id,
        "id2activity": {str(k): v for k, v in bundle.id2activity.items()},
        "split_mode": split_mode,
        "normalize_activity_names": bool(cfg.get("normalize_activity_names", False)),
        "split_sizes": {k: len(v) for k, v in split.as_dict().items()},
        "prefix_counts": {k: len(v) for k, v in datasets.items()},
    }
    save_json(out_dir / "meta.json", meta)

    with open(out_dir / "bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    with open(out_dir / "split.pkl", "wb") as f:
        pickle.dump(split, f)
    with open(out_dir / "prefixes.pkl", "wb") as f:
        pickle.dump(datasets, f)

    print(f"Saved prepared data to {out_dir}")
    print(
        f"cases={meta['num_cases']} activities={meta['num_activities']} "
        f"attrs={used_attrs} prefixes={meta['prefix_counts']}"
    )


if __name__ == "__main__":
    main()
