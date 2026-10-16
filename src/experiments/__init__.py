"""Step 10: ablation, noise injection, and explainability helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def inject_control_flow_noise(
    cases: Dict[str, List[str]],
    activities: Sequence[str],
    rate: float = 0.1,
    seed: int = 0,
    modes: Sequence[str] = ("skip", "insert", "swap"),
    per_event: bool = False,
) -> Dict[str, List[str]]:
    """
    Inject control-flow anomalies into cases (for robustness experiments).

    Modes (recording-error style):
      - skip: drop an internal event (missing / incomplete logging)
      - swap: adjacent transposition (out-of-order / timestamp jitter)
      - insert: spurious event (mixed CF ladder only)

    rate: if per_event=False, probability of mutating each case once;
          if per_event=True, probability of mutating each internal event.
    Pass modes=("skip",) or ("swap",) to isolate a single realistic noise type.
    """
    rng = np.random.default_rng(seed)
    acts = list(activities)
    out: Dict[str, List[str]] = {}
    for cid, seq in cases.items():
        s = list(seq)
        if len(s) < 2:
            out[cid] = s
            continue
        if per_event:
            # Bernoulli per internal position so rate ladders are distinguishable.
            # Work from a copy; apply mutations left-to-right on original indices
            # mapped onto the growing/shrinking sequence carefully via rebuild.
            s = list(seq)
            # Decide mutations on original positions 1..len-2 (and inserts anywhere)
            positions = list(range(len(s)))
            ops = []
            for i in positions:
                if i == 0 or i == len(s) - 1:
                    # only allow insert before/after endpoints with rate/2 to keep ends
                    if rng.random() < rate * 0.5:
                        ops.append(("insert", i))
                    continue
                if rng.random() < rate:
                    mode = modes[int(rng.integers(0, len(modes)))]
                    ops.append((mode, i))
            # Apply from right to left so indices stay valid for skip/swap
            for mode, i in sorted(ops, key=lambda x: -x[1]):
                if len(s) < 2:
                    break
                i = min(i, len(s) - 1)
                if mode == "skip" and 0 < i < len(s) - 1 and len(s) > 2:
                    s.pop(i)
                elif mode == "insert":
                    s.insert(i, acts[int(rng.integers(0, len(acts)))])
                elif mode == "swap" and i < len(s) - 1 and len(s) > 2:
                    j = min(i + 1, len(s) - 1)
                    s[i], s[j] = s[j], s[i]
            out[cid] = s
            continue
        if rng.random() > rate:
            out[cid] = s
            continue
        mode = modes[int(rng.integers(0, len(modes)))]
        if mode == "skip" and len(s) > 2:
            i = int(rng.integers(1, len(s) - 1))
            s.pop(i)
        elif mode == "insert":
            i = int(rng.integers(0, len(s)))
            s.insert(i, acts[int(rng.integers(0, len(acts)))])
        elif mode == "swap" and len(s) > 2:
            i = int(rng.integers(0, len(s) - 1))
            s[i], s[i + 1] = s[i + 1], s[i]
        out[cid] = s
    return out


def ablation_configs() -> List[Dict]:
    """Named ablation settings for CF-HGNN variants."""
    return [
        {"name": "lstm", "model": "lstm", "graph_feat": False, "node_feat": False, "extra_rels": False},
        {"name": "hgnn", "model": "hgnn", "graph_feat": False, "node_feat": False, "extra_rels": False},
        {"name": "hgnn_graph", "model": "hgnn", "graph_feat": True, "node_feat": False, "extra_rels": False},
        {"name": "hgnn_node", "model": "hgnn", "graph_feat": False, "node_feat": True, "extra_rels": False},
        {"name": "hgnn_graph_node", "model": "hgnn", "graph_feat": True, "node_feat": True, "extra_rels": False},
        {
            "name": "hgnn_full",
            "model": "hgnn",
            "graph_feat": True,
            "node_feat": True,
            "extra_rels": True,
        },
    ]


def explain_prefix_diagnosis(diag, id2activity: Dict[int, str] | None = None) -> Dict:
    """Human-readable explanation from an alignment diagnosis."""
    return {
        "case_id": diag.case_id,
        "prefix_len": diag.prefix_len,
        "fitness": diag.fitness,
        "total_cost": diag.total_cost,
        "log_moves": diag.log_moves,
        "model_moves": diag.model_moves,
        "per_activity_cost": diag.per_activity_cost,
        "deviation_transitions": diag.deviation_transitions,
        "recent_deviation": diag.recent_deviation,
        "summary": (
            f"fitness={diag.fitness:.3f}, cost={diag.total_cost:.1f}, "
            f"log_moves={sum(diag.log_moves.values())}, "
            f"model_moves={sum(diag.model_moves.values())}, "
            f"recent_dev={diag.recent_deviation}"
        ),
    }


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
