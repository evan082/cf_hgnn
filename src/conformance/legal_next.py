"""Legal next-activity masks from DFG ∪ Petri-allowed pairs."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .discovery import petri_allowed_df_pairs


def train_dfg_pairs(train_cases: Dict[str, Sequence[str]], activity2id: Dict[str, int]) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    for acts in train_cases.values():
        ids = [activity2id[a] for a in acts if a in activity2id]
        for i in range(len(ids) - 1):
            pairs.add((ids[i], ids[i + 1]))
    return pairs


def build_successor_index(
    pairs: Iterable[Tuple[int, int]],
    num_activities: int,
) -> List[Set[int]]:
    succ = [set() for _ in range(num_activities)]
    for a, b in pairs:
        if 0 <= a < num_activities and 0 <= b < num_activities:
            succ[a].add(b)
    return succ


def union_allowed_pairs(net, activity2id: Dict[str, int], train_cases: Dict[str, Sequence[str]]):
    """DFG ∪ Petri structural/empirical allowed pairs."""
    pairs = set(train_dfg_pairs(train_cases, activity2id))
    try:
        pairs |= set(petri_allowed_df_pairs(net, activity2id, train_cases=train_cases))
    except Exception:
        pass
    return pairs


def legal_next_ids(
    prefix_ids: Sequence[int],
    succ_index: List[Set[int]],
    num_activities: int,
) -> Set[int]:
    """Successors of the last activity; empty => unconstrained."""
    if not prefix_ids:
        return set(range(num_activities))
    last = int(prefix_ids[-1])
    if last < 0 or last >= len(succ_index):
        return set()
    return set(succ_index[last])


def legal_mask_vector(
    prefix_ids: Sequence[int],
    succ_index: List[Set[int]],
    num_activities: int,
) -> np.ndarray:
    """
    Binary mask [A]. If no known successors, return all-ones (fallback).
    """
    legal = legal_next_ids(prefix_ids, succ_index, num_activities)
    mask = np.zeros(num_activities, dtype=np.float32)
    if not legal:
        mask[:] = 1.0
    else:
        for a in legal:
            mask[a] = 1.0
    return mask


def cache_legal_masks(
    samples,
    succ_index: List[Set[int]],
    num_activities: int,
) -> Dict[Tuple[str, int], np.ndarray]:
    out: Dict[Tuple[str, int], np.ndarray] = {}
    for s in samples:
        key = (s.case_id, s.prefix_len)
        out[key] = legal_mask_vector(s.prefix_ids, succ_index, num_activities)
    return out


def apply_legal_mask_to_logits(logits, mask, neg_inf: float = -1e2):
    """logits: [B, A], mask: [B, A] with 1=legal. Empty rows stay unconstrained."""
    import torch

    m = mask.to(dtype=logits.dtype)
    row_sum = m.sum(dim=-1, keepdim=True)
    m = torch.where(row_sum > 0, m, torch.ones_like(m))
    return logits + (1.0 - m) * neg_inf


def prediction_entropy(logits):
    """Per-sample entropy H(softmax(z)). logits: [B, A] -> [B]."""
    import torch.nn.functional as F

    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def prediction_uncertainty(logits, kind: str = "entropy"):
    """
    Per-sample uncertainty scores (higher = more uncertain). logits: [B, A] -> [B].

      entropy: H(p)
      conf:    1 - max_a p_a
      margin:  1 - (p_(1) - p_(2))   with p_(2)=0 if |A|<2
    """
    import torch
    import torch.nn.functional as F

    kind = str(kind).lower()
    if kind in ("entropy", "ent", "h"):
        return prediction_entropy(logits)
    p = F.softmax(logits, dim=-1)
    if kind in ("conf", "confidence", "maxprob", "max_prob"):
        return 1.0 - p.max(dim=-1).values
    if kind in ("margin", "top2", "top1_top2"):
        if p.size(-1) < 2:
            return 1.0 - p.max(dim=-1).values
        top2 = torch.topk(p, k=2, dim=-1).values
        return 1.0 - (top2[:, 0] - top2[:, 1])
    raise ValueError(f"Unknown uncertainty kind: {kind}")


def soft_illegal_loss(logits, mask, entropy_weight: bool = True):
    """
    Soft legality regularizer (SPC train term):
      mean( sg(H(z)) * sum_a softmax(z)_a * (1 - m_a) )  if entropy_weight
      else mean( sum_a softmax(z)_a * (1 - m_a) )
    """
    import torch
    import torch.nn.functional as F

    m = mask.to(dtype=logits.dtype)
    row_sum = m.sum(dim=-1, keepdim=True)
    m = torch.where(row_sum > 0, m, torch.ones_like(m))
    p = F.softmax(logits, dim=-1)
    illegal_mass = (p * (1.0 - m)).sum(dim=-1)
    if entropy_weight:
        h = prediction_entropy(logits).detach()
        return (h * illegal_mass).mean()
    return illegal_mass.mean()


def apply_gated_legal_mask(
    logits,
    mask,
    tau: float,
    neg_inf: float = -1e2,
    uncertainty: str = "entropy",
):
    """
    Uncertainty-gated legal correction:
      if U(z) >= tau -> apply legal mask; else leave logits unchanged.
    Default U = predictive entropy (backward compatible).
    """
    import torch

    if tau is None or tau < 0:
        return logits
    u = prediction_uncertainty(logits, kind=uncertainty)
    gated = apply_legal_mask_to_logits(logits, mask, neg_inf=neg_inf)
    use_gate = (u >= float(tau)).unsqueeze(-1)
    return torch.where(use_gate, gated, logits)


def legal_rate_from_preds(preds, masks) -> float:
    """Fraction of predictions that fall on a legal activity (mask==1)."""
    import torch

    if masks is None:
        return 0.0
    if isinstance(preds, list):
        preds = torch.tensor(preds, dtype=torch.long)
    if not torch.is_tensor(masks):
        masks = torch.tensor(masks, dtype=torch.float32)
    m = masks.to(dtype=torch.float32)
    preds = preds.to(device=m.device, dtype=torch.long)
    gather = m[torch.arange(m.size(0), device=m.device), preds]
    return float(gather.mean().item())
