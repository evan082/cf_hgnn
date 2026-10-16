"""Step 6: discover Petri net from training cases."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple

from ..data.log_io import cases_to_pm4py_log


def discover_petri_net(
    train_cases: Dict[str, Sequence[str]],
    noise_threshold: float = 0.2,
):
    """
    Discover an accepting Petri net from training traces using Inductive Miner.
    Returns (net, initial_marking, final_marking).
    """
    import pm4py

    log = cases_to_pm4py_log(train_cases)
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise_threshold)
    return net, im, fm


def save_petri(path: str | Path, net, im, fm) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"net": net, "im": im, "fm": fm}, f)


def load_petri(path: str | Path) -> Tuple:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["net"], obj["im"], obj["fm"]


def petri_allowed_df_pairs(net, activity2id: Dict[str, int], train_cases=None):
    """
    Approximate model-allowed activity pairs.

    1) Structural heuristic on Petri net places/transitions.
    2) Fallback: directly-follows pairs observed in training cases
       (empirical reference when the net is highly silent/complex).
    """
    pairs = set()
    label_of = {}
    for t in net.transitions:
        lab = t.label
        if lab is None:
            continue
        if lab in activity2id:
            label_of[t] = activity2id[lab]

    consumers = {p: [] for p in net.places}
    producers = {p: [] for p in net.places}
    for arc in net.arcs:
        src, tgt = arc.source, arc.target
        if src in net.transitions and tgt in net.places:
            producers[tgt].append(src)
        elif src in net.places and tgt in net.transitions:
            consumers[src].append(tgt)

    for p in net.places:
        for t1 in producers[p]:
            for t2 in consumers[p]:
                if t1 in label_of and t2 in label_of:
                    pairs.add((label_of[t1], label_of[t2]))

    # Inductive nets often insert silent transitions, so structural labeled
    # pairs can be near-empty. Complement with training DF pairs (the net
    # was discovered from train, so these are empirical model-allowed edges).
    if train_cases is not None:
        for acts in train_cases.values():
            ids = [activity2id[a] for a in acts if a in activity2id]
            for i in range(len(ids) - 1):
                pairs.add((ids[i], ids[i + 1]))
    return pairs
