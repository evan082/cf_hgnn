# -*- coding: utf-8 -*-
"""Reachability / enabled-transition masks via Petri token game (ReachMask)."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def _marking_key(marking) -> Tuple:
    # Marking is Counter-like: place -> tokens
    try:
        items = tuple(sorted((str(p), int(marking[p])) for p in marking if marking[p]))
    except Exception:
        items = tuple(sorted((str(p), int(c)) for p, c in dict(marking).items() if c))
    return items


def _silent_enabled(net, marking):
    from pm4py.objects.petri_net.semantics import enabled_transitions

    return [t for t in enabled_transitions(net, marking) if t.label is None]


def _labeled_enabled(net, marking, activity: str):
    from pm4py.objects.petri_net.semantics import enabled_transitions

    return [t for t in enabled_transitions(net, marking) if t.label == activity]


def _all_labeled_enabled(net, marking) -> Set[str]:
    from pm4py.objects.petri_net.semantics import enabled_transitions

    return {t.label for t in enabled_transitions(net, marking) if t.label is not None}


def fire_activity_with_silent_search(net, marking, activity: str, max_states: int = 500):
    """
    Find a marking reachable by silent transitions where `activity` is enabled, then fire it.
    BFS over silent firings; returns new marking or None.
    """
    from pm4py.objects.petri_net.semantics import weak_execute

    # If already enabled, fire directly
    cands = _labeled_enabled(net, marking, activity)
    if cands:
        return weak_execute(cands[0], marking)

    q = deque([marking])
    seen = {_marking_key(marking)}
    while q and len(seen) < max_states:
        m = q.popleft()
        for st in _silent_enabled(net, m):
            m2 = weak_execute(st, m)
            k = _marking_key(m2)
            if k in seen:
                continue
            seen.add(k)
            cands = _labeled_enabled(net, m2, activity)
            if cands:
                return weak_execute(cands[0], m2)
            q.append(m2)
    return None


def replay_prefix_marking(net, im, prefix_acts: Sequence[str]):
    m = im
    for act in prefix_acts:
        m2 = fire_activity_with_silent_search(net, m, act)
        if m2 is None:
            return None
        m = m2
    return m


def enabled_activity_labels_with_silent(net, marking, max_states: int = 400) -> Set[str]:
    """Labels enabled now or after some silent firings (BFS)."""
    from pm4py.objects.petri_net.semantics import weak_execute

    labs = set(_all_labeled_enabled(net, marking))
    q = deque([marking])
    seen = {_marking_key(marking)}
    while q and len(seen) < max_states:
        m = q.popleft()
        labs |= _all_labeled_enabled(net, m)
        for st in _silent_enabled(net, m):
            m2 = weak_execute(st, m)
            k = _marking_key(m2)
            if k in seen:
                continue
            seen.add(k)
            q.append(m2)
    return labs


def reachability_mask_vector(
    prefix_acts: Sequence[str],
    net,
    im,
    activity2id: Dict[str, int],
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = len(activity2id)
    mask = np.zeros(n, dtype=np.float32)
    marking = replay_prefix_marking(net, im, prefix_acts)
    if marking is None:
        if fallback is not None:
            return fallback.astype(np.float32, copy=True)
        mask[:] = 1.0
        return mask
    labs = enabled_activity_labels_with_silent(net, marking)
    for lab, idx in activity2id.items():
        if lab in labs:
            mask[idx] = 1.0
    if mask.sum() <= 0:
        if fallback is not None:
            return fallback.astype(np.float32, copy=True)
        mask[:] = 1.0
    return mask


def cache_reachability_masks(
    samples,
    net,
    im,
    activity2id: Dict[str, int],
    id2activity: Dict[int, str],
    succ_index: Optional[List[Set[int]]] = None,
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[str, float]]:
    from .legal_next import legal_mask_vector

    out: Dict[Tuple[str, int], np.ndarray] = {}
    n_fail = 0
    n_empty = 0
    n_diff = 0
    n = len(activity2id)
    for s in samples:
        key = (s.case_id, s.prefix_len)
        acts = [id2activity[int(i)] for i in s.prefix_ids if int(i) in id2activity]
        fb = legal_mask_vector(s.prefix_ids, succ_index, n) if succ_index is not None else None
        marking = replay_prefix_marking(net, im, acts)
        if marking is None:
            n_fail += 1
            m = fb if fb is not None else np.ones(n, dtype=np.float32)
        else:
            labs = enabled_activity_labels_with_silent(net, marking)
            m = np.zeros(n, dtype=np.float32)
            for lab, idx in activity2id.items():
                if lab in labs:
                    m[idx] = 1.0
            if m.sum() <= 0:
                n_empty += 1
                m = fb if fb is not None else np.ones(n, dtype=np.float32)
            elif fb is not None and not np.array_equal(m > 0, fb > 0):
                n_diff += 1
        out[key] = m
    total = max(len(samples), 1)
    stats = {
        "n_samples": float(len(samples)),
        "replay_fail_rate": n_fail / total,
        "empty_enabled_rate": n_empty / total,
        "diff_from_dfg_rate": n_diff / total,
    }
    return out, stats
