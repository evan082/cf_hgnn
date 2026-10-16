"""Steps 5/8/9: Heterogeneous GNN for next-activity prediction (pure PyTorch)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hetero_graph import HeteroProcessGraph, build_prefix_hetero_graph


class RelGraphConv(nn.Module):
    """Simple relation-specific mean aggregation + linear transform."""

    def __init__(self, in_dim: int, out_dim: int, num_rels: int, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(self, x: torch.Tensor, edge_index_list: List[torch.Tensor]) -> torch.Tensor:
        out = self.self_loop(x)
        for r, ei in enumerate(edge_index_list):
            if ei is None or ei.numel() == 0:
                continue
            src, dst = ei[0], ei[1]
            msg = x.index_select(0, src) @ self.weight[r]
            agg = torch.zeros(x.size(0), msg.size(-1), device=x.device, dtype=x.dtype)
            cnt = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            agg.index_add_(0, dst, msg)
            ones = torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype)
            cnt.index_add_(0, dst, ones)
            agg = agg / cnt.clamp(min=1.0)
            out = out + agg
        out = out + self.bias
        return self.dropout(F.relu(out))


class HGNNNextActivity(nn.Module):
    """
    Heterogeneous GNN over forward/backward/repeat (+ optional allowed/deviation).
    Optional graph-/node-level conformance features (concat).
    """

    def __init__(
        self,
        num_activities: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        relation_types: Sequence[str] = ("forward", "backward", "repeat"),
        graph_feat_dim: int = 0,
        node_conf_dim: int = 0,
    ):
        super().__init__()
        self.num_activities = num_activities
        self.relation_types = list(relation_types)
        self.graph_feat_dim = graph_feat_dim
        self.node_conf_dim = node_conf_dim

        self.embed = nn.Embedding(num_activities, embed_dim)
        self.input_proj = nn.Linear(embed_dim + node_conf_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                RelGraphConv(hidden_dim, hidden_dim, num_rels=len(self.relation_types), dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim + graph_feat_dim, num_activities)

    def encode_graph(
        self,
        graph: HeteroProcessGraph,
        node_conf: Optional[torch.Tensor] = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        hidden_dim = self.input_proj.out_features
        if graph.num_nodes == 0:
            return torch.zeros(hidden_dim, device=device)

        node_ids = torch.tensor(graph.node_ids, dtype=torch.long, device=device)
        x = self.embed(node_ids)
        if self.node_conf_dim > 0:
            if node_conf is None:
                conf = torch.zeros(graph.num_nodes, self.node_conf_dim, device=device)
            elif node_conf.dim() == 2 and node_conf.size(0) == self.num_activities:
                conf = node_conf.index_select(0, node_ids)
            elif node_conf.dim() == 2 and node_conf.size(0) == graph.num_nodes:
                conf = node_conf.to(device)
            else:
                conf = torch.zeros(graph.num_nodes, self.node_conf_dim, device=device)
            x = torch.cat([x, conf], dim=-1)
        x = self.input_proj(x)

        edge_index_list: List[torch.Tensor] = []
        for rel in self.relation_types:
            elist = graph.edges.get(rel, [])
            if elist:
                src = torch.tensor([e[0] for e in elist], dtype=torch.long, device=device)
                dst = torch.tensor([e[1] for e in elist], dtype=torch.long, device=device)
                ei = torch.stack([src, dst], dim=0)
            else:
                ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_index_list.append(ei)

        for layer in self.layers:
            x = layer(x, edge_index_list)
        return x.mean(dim=0)

    def forward_batch(
        self,
        prefix_ids_list: List[Sequence[int]],
        graph_feat: Optional[torch.Tensor] = None,
        node_conf_table: Optional[torch.Tensor] = None,
        allowed_pairs=None,
        deviation_maps: Optional[List[set]] = None,
        use_extra_rels: bool = False,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        vectors = []
        rels = list(self.relation_types)
        for i, pref in enumerate(prefix_ids_list):
            if isinstance(pref, torch.Tensor):
                pref = pref.tolist()
            allowed = allowed_pairs if use_extra_rels else None
            dev = deviation_maps[i] if (use_extra_rels and deviation_maps is not None) else None
            g = build_prefix_hetero_graph(
                pref,
                num_activities=self.num_activities,
                include_backward="backward" in rels,
                include_repeat="repeat" in rels,
                allowed_pairs=allowed,
                deviation_pairs=dev,
            )
            node_conf = None
            if self.node_conf_dim > 0 and node_conf_table is not None:
                node_conf = node_conf_table[i] if node_conf_table.dim() == 3 else node_conf_table
            vectors.append(self.encode_graph(g, node_conf=node_conf, device=device))

        h = self.dropout(torch.stack(vectors, dim=0))
        if self.graph_feat_dim > 0:
            if graph_feat is None:
                graph_feat = torch.zeros(h.size(0), self.graph_feat_dim, device=device)
            h = torch.cat([h, graph_feat.to(device)], dim=-1)
        return self.fc(h)

    def forward(self, *args, **kwargs):
        return self.forward_batch(*args, **kwargs)
