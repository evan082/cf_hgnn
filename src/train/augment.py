"""Online control-flow noise augmentation for optional training robustness."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


def mutate_prefix_ids(
    prefix_ids: Sequence[int],
    num_activities: int,
    rng: np.random.Generator,
    modes: Sequence[str] = ("skip", "insert", "swap"),
) -> Tuple[List[int], Set[Tuple[int, int]], bool]:
    s = list(prefix_ids)
    if len(s) < 2 or num_activities <= 0:
        return s, set(), False

    mode = modes[int(rng.integers(0, len(modes)))]
    dev: Set[Tuple[int, int]] = set()

    if mode == "skip" and len(s) > 2:
        i = int(rng.integers(1, len(s) - 1))
        if i - 1 >= 0 and i + 1 < len(s):
            dev.add((s[i - 1], s[i + 1]))
        s.pop(i)
        return s, dev, True

    if mode == "insert":
        i = int(rng.integers(0, len(s)))
        a = int(rng.integers(0, num_activities))
        if i > 0:
            dev.add((s[i - 1], a))
        if i < len(s):
            dev.add((a, s[i]))
        s.insert(i, a)
        return s, dev, True

    if mode == "swap" and len(s) > 2:
        i = int(rng.integers(0, len(s) - 1))
        s[i], s[i + 1] = s[i + 1], s[i]
        if i > 0:
            dev.add((s[i - 1], s[i]))
        dev.add((s[i], s[i + 1]))
        if i + 2 < len(s):
            dev.add((s[i + 1], s[i + 2]))
        return s, set(dev), True

    return s, set(), False


def synthetic_graph_feat(device=None, severity: float = 0.45) -> torch.Tensor:
    s = float(max(0.05, min(1.0, severity)))
    return torch.tensor(
        [1.0 - 0.55 * s, 0.35 + 0.55 * s, 0.4 + 1.2 * s, 0.3 + 0.7 * s, 1.0],
        dtype=torch.float32,
        device=device,
    )


def synthetic_node_feat(num_activities: int, affected: Sequence[int], device=None) -> torch.Tensor:
    feat = torch.zeros(num_activities, 3, dtype=torch.float32, device=device)
    for a in affected:
        if 0 <= a < num_activities:
            feat[a, 0] = 1.1
            feat[a, 2] = 1.1
    return feat


def augment_batch_prefixes(
    prefs: List[List[int]],
    num_activities: int,
    aug_rate: float,
    rng: np.random.Generator,
    graph_feat: Optional[torch.Tensor],
    node_table: Optional[torch.Tensor],
    use_graph_feat: bool,
    use_node_feat: bool,
    use_extra_rels: bool,
    base_deviation_maps: Optional[List[set]],
) -> Tuple[List[List[int]], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[set]]]:
    if aug_rate <= 0:
        return prefs, graph_feat, node_table, base_deviation_maps

    device = None
    if graph_feat is not None:
        device = graph_feat.device
    elif node_table is not None:
        device = node_table.device

    new_prefs: List[List[int]] = []
    dev_maps: List[set] = []
    g_list, n_list = [], []

    for i in range(len(prefs)):
        pref = list(prefs[i])
        base_dev = set(base_deviation_maps[i]) if base_deviation_maps is not None else set()

        if rng.random() < aug_rate:
            pref2, mut_dev, ok = mutate_prefix_ids(pref, num_activities, rng)
            if ok and len(pref2) >= 1:
                pref = pref2
                base_dev = base_dev | mut_dev
                if use_graph_feat:
                    g_list.append(synthetic_graph_feat(device=device))
                if use_node_feat:
                    affected = sorted({a for p in mut_dev for a in p} | set(pref[-2:]))
                    n_list.append(synthetic_node_feat(num_activities, affected, device=device))
                elif node_table is not None:
                    n_list.append(node_table[i] if node_table.dim() == 3 else node_table)
            else:
                if use_graph_feat and graph_feat is not None:
                    g_list.append(graph_feat[i])
                if use_node_feat and node_table is not None:
                    n_list.append(node_table[i] if node_table.dim() == 3 else node_table)
        else:
            if use_graph_feat and graph_feat is not None:
                g_list.append(graph_feat[i])
            if use_node_feat and node_table is not None:
                n_list.append(node_table[i] if node_table.dim() == 3 else node_table)

        new_prefs.append(pref)
        if use_extra_rels:
            dev_maps.append(base_dev)

    new_g = torch.stack(g_list, dim=0) if g_list else graph_feat
    new_n = torch.stack(n_list, dim=0) if n_list else node_table
    new_dev = dev_maps if use_extra_rels else base_deviation_maps
    return new_prefs, new_g, new_n, new_dev
