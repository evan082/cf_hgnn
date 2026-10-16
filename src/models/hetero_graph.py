"""Step 4: build forward / backward / repeat heterogeneous process graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import torch


EdgeList = List[Tuple[int, int]]


@dataclass
class HeteroProcessGraph:
    """Activity-centric heterogeneous graph for one prefix (or global DFG)."""

    num_nodes: int
    node_ids: List[int]  # local index -> global activity id
    edges: Dict[str, EdgeList] = field(default_factory=dict)
    # optional per-edge features aligned with edges[rel] order: [E, F]
    edge_attrs: Dict[str, List[List[float]]] = field(default_factory=dict)

    def relation_types(self) -> List[str]:
        return list(self.edges.keys())

    def to_tensors(self, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "node_ids": torch.tensor(self.node_ids, dtype=torch.long, device=device),
        }
        for rel, elist in self.edges.items():
            if elist:
                src = torch.tensor([e[0] for e in elist], dtype=torch.long, device=device)
                dst = torch.tensor([e[1] for e in elist], dtype=torch.long, device=device)
            else:
                src = torch.zeros(0, dtype=torch.long, device=device)
                dst = torch.zeros(0, dtype=torch.long, device=device)
            out[f"edge_index_{rel}"] = torch.stack([src, dst], dim=0) if src.numel() else torch.zeros(2, 0, dtype=torch.long, device=device)
            if rel in self.edge_attrs and self.edge_attrs[rel]:
                out[f"edge_attr_{rel}"] = torch.tensor(
                    self.edge_attrs[rel], dtype=torch.float32, device=device
                )
        return out


def _add_edge(store: Dict[str, Set[Tuple[int, int]]], rel: str, u: int, v: int) -> None:
    store[rel].add((u, v))


def build_prefix_hetero_graph(
    prefix_ids: Sequence[int],
    num_activities: int,
    include_backward: bool = True,
    include_repeat: bool = True,
    allowed_pairs: Set[Tuple[int, int]] | None = None,
    deviation_pairs: Set[Tuple[int, int]] | None = None,
) -> HeteroProcessGraph:
    """
    Build a heterogeneous graph from a prefix activity id sequence.

    Nodes: unique activities appearing in the prefix (local indexing).
    Relations:
      - forward: directly-follows a_i -> a_{i+1}
      - backward: reverse of forward
      - repeat: connect repeated occurrences of the same activity along the prefix
      - allowed (optional): model-allowed DF pairs among present nodes
      - deviation (optional): observed transitions flagged as deviations
    """
    if not prefix_ids:
        return HeteroProcessGraph(num_nodes=0, node_ids=[], edges={})

    # map global activity id -> local node index (first occurrence order)
    seen: Dict[int, int] = {}
    node_ids: List[int] = []
    for aid in prefix_ids:
        if aid not in seen:
            seen[aid] = len(node_ids)
            node_ids.append(aid)

    edge_sets: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)

    # forward DF
    for i in range(len(prefix_ids) - 1):
        u = seen[prefix_ids[i]]
        v = seen[prefix_ids[i + 1]]
        _add_edge(edge_sets, "forward", u, v)

    if include_backward:
        for u, v in list(edge_sets["forward"]):
            _add_edge(edge_sets, "backward", v, u)

    if include_repeat:
        positions: Dict[int, List[int]] = defaultdict(list)
        for i, aid in enumerate(prefix_ids):
            positions[aid].append(i)
        for aid, pos in positions.items():
            if len(pos) < 2:
                continue
            local = seen[aid]
            # connect consecutive repetitions as a self-loop style signal via self-edge
            _add_edge(edge_sets, "repeat", local, local)

    if allowed_pairs:
        for a, b in allowed_pairs:
            if a in seen and b in seen:
                _add_edge(edge_sets, "allowed", seen[a], seen[b])

    if deviation_pairs:
        for a, b in deviation_pairs:
            if a in seen and b in seen:
                _add_edge(edge_sets, "deviation", seen[a], seen[b])

    edges = {rel: sorted(list(s)) for rel, s in edge_sets.items()}
    # ensure core relations exist even if empty
    for rel in ("forward", "backward", "repeat"):
        edges.setdefault(rel, [])

    return HeteroProcessGraph(num_nodes=len(node_ids), node_ids=node_ids, edges=edges)


def build_global_dfg(cases: Dict[str, Sequence[str]], activity2id: Dict[str, int]) -> Set[Tuple[int, int]]:
    """Collect global directly-follows pairs from cases (for reference / allowed edges)."""
    pairs: Set[Tuple[int, int]] = set()
    for acts in cases.values():
        ids = [activity2id[a] for a in acts]
        for i in range(len(ids) - 1):
            pairs.add((ids[i], ids[i + 1]))
    return pairs
