"""Case-level train/val/test split (no case leakage across splits)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class CaseSplit:
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "train": self.train_ids,
            "val": self.val_ids,
            "test": self.test_ids,
        }


def split_cases(
    case_ids: Sequence[str],
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
    seed: int = 42,
    mode: str = "random",
) -> CaseSplit:
    """
    Split case IDs. Ratios are normalized to sum to 1.

    mode:
      - random: shuffle then cut (default)
      - sequential: keep input order (ProcessTransformer-style first-N train)
    """
    ids = list(case_ids)
    if not ids:
        raise ValueError("Empty case id list")

    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = (
        train_ratio / total,
        val_ratio / total,
        test_ratio / total,
    )

    if mode == "sequential":
        ordered = ids
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ids))
        ordered = [ids[i] for i in perm]

    n = len(ordered)
    # ProcessTransformer uses int(n * train_ratio) then rest as test (no val).
    # When val_ratio>0 we carve val from the train block's tail.
    n_test = max(1, n - int(n * (train_ratio + val_ratio))) if mode == "sequential" else None
    if mode == "sequential":
        n_train_block = int(n * (1.0 - test_ratio))
        if n_train_block >= n:
            n_train_block = n - 1
        n_val = int(round(n_train_block * (val_ratio / max(train_ratio + val_ratio, 1e-9))))
        if val_ratio <= 0:
            n_val = 0
        n_train = n_train_block - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(0, n_train_block - 1)
        train_ids = ordered[:n_train]
        val_ids = ordered[n_train:n_train + n_val]
        test_ids = ordered[n_train + n_val :]
    else:
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(0, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train_ids = ordered[:n_train]
        val_ids = ordered[n_train : n_train + n_val]
        test_ids = ordered[n_train + n_val :]

    assert len(train_ids) + len(val_ids) + len(test_ids) == n

    return CaseSplit(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
