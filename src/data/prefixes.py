"""Step 2: prefix–label dataset for next-activity prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # ImportError / OSError (broken DLL) — allow non-torch scripts
    torch = None
    Dataset = object  # type: ignore


@dataclass
class PrefixSample:
    case_id: str
    prefix_acts: List[str]
    prefix_ids: List[int]
    label_act: str
    label_id: int
    prefix_len: int
    # optional per-event time features: [dt_from_start_h, dt_from_prev_h] (log1p hours)
    prefix_times: Optional[List[List[float]]] = None
    prefix_resource_ids: Optional[List[int]] = None
    # optional categorical attrs per event: [T, n_attrs]
    prefix_attr_ids: Optional[List[List[int]]] = None


def _log1p_hours(seconds: float) -> float:
    import math

    return float(math.log1p(max(seconds, 0.0) / 3600.0))


def build_prefix_dataset(
    cases: Dict[str, Sequence[str]],
    case_ids: Sequence[str],
    activity2id: Dict[str, int],
    min_prefix_len: int = 1,
    max_prefix_len: int | None = None,
    case_timestamps: Optional[Dict[str, Sequence]] = None,
    case_resources: Optional[Dict[str, Sequence[int]]] = None,
    case_attrs: Optional[Dict[str, Sequence[Sequence[int]]]] = None,
) -> List[PrefixSample]:
    samples: List[PrefixSample] = []
    for cid in case_ids:
        acts = list(cases[cid])
        if len(acts) < min_prefix_len + 1:
            continue
        ids = [activity2id[a] for a in acts]
        ts = None
        if case_timestamps is not None and cid in case_timestamps:
            ts = list(case_timestamps[cid])
            if len(ts) != len(acts):
                ts = None
        res = None
        if case_resources is not None and cid in case_resources:
            res = list(case_resources[cid])
            if len(res) != len(acts):
                res = None
        attrs = None
        if case_attrs is not None and cid in case_attrs:
            attrs = [list(row) for row in case_attrs[cid]]
            if len(attrs) != len(acts):
                attrs = None
        upper = len(acts)
        if max_prefix_len is not None:
            upper = min(upper, max_prefix_len + 1)
        for k in range(min_prefix_len, upper):
            ptimes = None
            if ts is not None:
                t0 = ts[0]
                ptimes = []
                for i in range(k):
                    try:
                        dt0 = (ts[i] - t0).total_seconds()
                        dtp = 0.0 if i == 0 else (ts[i] - ts[i - 1]).total_seconds()
                    except Exception:
                        dt0, dtp = float(i), 1.0
                    ptimes.append([_log1p_hours(dt0), _log1p_hours(dtp)])
            samples.append(
                PrefixSample(
                    case_id=str(cid),
                    prefix_acts=acts[:k],
                    prefix_ids=ids[:k],
                    label_act=acts[k],
                    label_id=ids[k],
                    prefix_len=k,
                    prefix_times=ptimes,
                    prefix_resource_ids=(res[:k] if res is not None else None),
                    prefix_attr_ids=(attrs[:k] if attrs is not None else None),
                )
            )
    return samples


class PrefixDataset(Dataset):
    """Torch dataset over prefix samples with optional conformance caches."""

    def __init__(
        self,
        samples: List[PrefixSample],
        graph_feats: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
        node_feats: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
        legal_masks: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
        num_activities: Optional[int] = None,
    ):
        self.samples = samples
        self.graph_feats = graph_feats or {}
        self.node_feats = node_feats or {}
        self.legal_masks = legal_masks or {}
        self.num_activities = num_activities

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        if torch is None:
            raise ImportError("PyTorch is required for PrefixDataset")
        s = self.samples[idx]
        key = (s.case_id, s.prefix_len)
        item = {
            "case_id": s.case_id,
            "prefix_ids": torch.tensor(s.prefix_ids, dtype=torch.long),
            "label": torch.tensor(s.label_id, dtype=torch.long),
            "prefix_len": s.prefix_len,
            "prefix_acts": s.prefix_acts,
        }
        if s.prefix_times is not None:
            item["time_feat"] = torch.tensor(s.prefix_times, dtype=torch.float32)
        if s.prefix_resource_ids is not None:
            item["resource_ids"] = torch.tensor(s.prefix_resource_ids, dtype=torch.long)
        if s.prefix_attr_ids is not None:
            item["attr_ids"] = torch.tensor(s.prefix_attr_ids, dtype=torch.long)
        if key in self.graph_feats:
            item["graph_feat"] = torch.tensor(self.graph_feats[key], dtype=torch.float32)
        if key in self.node_feats:
            item["node_feat"] = torch.tensor(self.node_feats[key], dtype=torch.float32)
        if key in self.legal_masks:
            item["legal_mask"] = torch.tensor(self.legal_masks[key], dtype=torch.float32)
        elif self.num_activities is not None:
            item["legal_mask"] = torch.ones(self.num_activities, dtype=torch.float32)
        return item


def collate_pad(batch: List[Dict]):
    """Pad variable-length prefixes for LSTM / Transformer / batched HGNN."""
    if torch is None:
        raise ImportError("PyTorch is required for collate_pad")
    lengths = [b["prefix_ids"].numel() for b in batch]
    max_len = max(lengths)
    B = len(batch)
    padded = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.stack([b["label"] for b in batch])
    for i, b in enumerate(batch):
        L = lengths[i]
        padded[i, :L] = b["prefix_ids"]
        mask[i, :L] = True

    out = {
        "prefix_ids": padded,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "label": labels,
        "case_ids": [b["case_id"] for b in batch],
        "prefix_lens": [b["prefix_len"] for b in batch],
        "prefix_acts": [b["prefix_acts"] for b in batch],
    }
    if "graph_feat" in batch[0]:
        out["graph_feat"] = torch.stack([b["graph_feat"] for b in batch])
    if "node_feat" in batch[0]:
        out["node_feat"] = torch.stack([b["node_feat"] for b in batch])
    if "legal_mask" in batch[0]:
        out["legal_mask"] = torch.stack([b["legal_mask"] for b in batch])
    if "time_feat" in batch[0]:
        tdim = batch[0]["time_feat"].size(-1)
        tpad = torch.zeros(B, max_len, tdim, dtype=torch.float32)
        for i, b in enumerate(batch):
            L = lengths[i]
            tpad[i, :L] = b["time_feat"]
        out["time_feat"] = tpad
    if "resource_ids" in batch[0]:
        rpad = torch.zeros(B, max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            L = lengths[i]
            rpad[i, :L] = b["resource_ids"]
        out["resource_ids"] = rpad
    if "attr_ids" in batch[0]:
        n_attr = batch[0]["attr_ids"].size(-1)
        apad = torch.zeros(B, max_len, n_attr, dtype=torch.long)
        for i, b in enumerate(batch):
            L = lengths[i]
            apad[i, :L] = b["attr_ids"]
        out["attr_ids"] = apad
    return out
