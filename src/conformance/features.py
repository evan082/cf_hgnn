"""Steps 8–9: convert alignment diagnoses into graph-/node-level features."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .alignment import AlignmentDiagnosis


# Graph-level feature layout (fixed dim = 5)
# [prefix_fitness, total_cost_norm, n_log_moves, n_model_moves, recent_deviation]
GRAPH_FEAT_DIM = 5


def diagnosis_to_graph_feat(
    diag: AlignmentDiagnosis,
    cost_scale: float = 10.0,
) -> np.ndarray:
    n_log = float(sum(diag.log_moves.values()))
    n_model = float(sum(diag.model_moves.values()))
    pref_fit = getattr(diag, "prefix_fitness", None)
    if pref_fit is None:
        n_sync = float(sum(diag.sync_moves.values()))
        pref_fit = n_sync / max(n_sync + n_log, 1.0)
    # log1p keeps sparse move counts well-scaled for neural nets
    return np.array(
        [
            float(np.clip(pref_fit, 0.0, 1.0)),
            float(np.log1p(diag.total_cost) / np.log1p(max(cost_scale, 1.0))),
            float(np.log1p(n_log)),
            float(np.log1p(n_model)),
            float(diag.recent_deviation),
        ],
        dtype=np.float32,
    )


def diagnosis_to_node_feat(
    diag: AlignmentDiagnosis,
    activities: Sequence[str],
    activity2id: Dict[str, int],
) -> np.ndarray:
    """
    Per-activity node features, shape [num_activities, 3]:
      [log_move_count, model_move_count, per_activity_cost] (log1p-scaled)
    """
    A = len(activity2id)
    feat = np.zeros((A, 3), dtype=np.float32)
    for act, cnt in diag.log_moves.items():
        if act in activity2id:
            feat[activity2id[act], 0] = float(np.log1p(cnt))
    for act, cnt in diag.model_moves.items():
        if act in activity2id:
            feat[activity2id[act], 1] = float(np.log1p(cnt))
    for act, cost in diag.per_activity_cost.items():
        if act in activity2id:
            feat[activity2id[act], 2] = float(np.log1p(cost))
    return feat


NODE_FEAT_DIM = 3


def build_feature_maps(
    diagnoses: Dict[Tuple[str, int], AlignmentDiagnosis],
    activity2id: Dict[str, int],
    cost_scale: float = 10.0,
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], np.ndarray]]:
    """Return (graph_feat_map, node_feat_map)."""
    activities = list(activity2id.keys())
    gmap: Dict[Tuple[str, int], np.ndarray] = {}
    nmap: Dict[Tuple[str, int], np.ndarray] = {}
    for key, diag in diagnoses.items():
        gmap[key] = diagnosis_to_graph_feat(diag, cost_scale=cost_scale)
        nmap[key] = diagnosis_to_node_feat(diag, activities, activity2id)
    return gmap, nmap


def deviation_pairs_from_diag(
    diag: AlignmentDiagnosis,
    activity2id: Dict[str, int],
) -> set:
    pairs = set()
    for a, b in diag.deviation_transitions:
        if a in activity2id and b in activity2id:
            pairs.add((activity2id[a], activity2id[b]))
    return pairs
