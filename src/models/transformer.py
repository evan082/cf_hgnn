"""ProcessTransformer-style encoder for next-activity prediction."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerNextActivity(nn.Module):
    """
    Activity (+ resource / extra categorical attrs / time) + PE + TransformerEncoder.
    Pooling: concat(mean of valid tokens, last valid token) -> MLP head.
    """

    def __init__(
        self,
        num_activities: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 4,
        nhead: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
        ff_mult: int = 4,
        time_feat_dim: int = 0,
        num_resources: int = 0,
        attr_sizes: Optional[Sequence[int]] = None,
        pool_mode: str = "mean_last",
        learned_pos: bool = False,
    ):
        super().__init__()
        self.num_activities = num_activities
        self.embed_dim = embed_dim
        self.time_feat_dim = time_feat_dim
        self.num_resources = num_resources
        self.attr_sizes = list(attr_sizes or [])
        self.pool_mode = pool_mode if pool_mode in ("mean", "mean_last", "last") else "mean_last"
        self.learned_pos = bool(learned_pos)

        self.embed = nn.Embedding(num_activities, embed_dim)
        self.resource_embed = (
            nn.Embedding(num_resources, embed_dim, padding_idx=0) if num_resources > 1 else None
        )
        # one embed per attr column (same order as prefix attr_ids / meta.attr_sizes)
        self.attr_embeds = nn.ModuleList(
            [
                nn.Embedding(max(n, 2), embed_dim, padding_idx=0)
                for n in self.attr_sizes
            ]
        )
        self.time_proj = nn.Linear(time_feat_dim, embed_dim) if time_feat_dim > 0 else None
        if self.learned_pos:
            self.pos_embed = nn.Embedding(max_len, embed_dim)
            self.pos_drop = nn.Dropout(dropout)
            self.pos = None
        else:
            self.pos_embed = None
            self.pos_drop = None
            self.pos = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        ff_dim = max(hidden_dim, embed_dim) * ff_mult
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        head_in = embed_dim * 2 if self.pool_mode == "mean_last" else embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_activities),
        )

    def forward(
        self,
        prefix_ids: torch.Tensor,
        lengths: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        time_feat: torch.Tensor | None = None,
        resource_ids: torch.Tensor | None = None,
        attr_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T = prefix_ids.shape
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, T, dtype=torch.bool, device=prefix_ids.device)
            for i, L in enumerate(lengths.tolist()):
                if L < T:
                    key_padding_mask[i, L:] = True

        x = self.embed(prefix_ids) * math.sqrt(self.embed_dim)
        if self.resource_embed is not None and resource_ids is not None:
            x = x + self.resource_embed(resource_ids)
        if attr_ids is not None and len(self.attr_embeds) > 0:
            # attr_ids: [B, T, n_attr]
            n = min(attr_ids.size(-1), len(self.attr_embeds))
            for j in range(n):
                x = x + self.attr_embeds[j](attr_ids[:, :, j])
        if self.time_proj is not None and time_feat is not None:
            x = x + self.time_proj(time_feat)
        if self.pos_embed is not None:
            pos = torch.arange(T, device=prefix_ids.device).unsqueeze(0).expand(B, -1)
            x = self.pos_drop(x + self.pos_embed(pos))
        else:
            x = self.pos(x)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).unsqueeze(-1).to(h.dtype)
        mean_h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        idx = (lengths - 1).clamp(min=0)
        last_h = h[torch.arange(B, device=h.device), idx]
        if self.pool_mode == "mean":
            pooled = self.dropout(mean_h)
        elif self.pool_mode == "last":
            pooled = self.dropout(last_h)
        else:
            pooled = self.dropout(torch.cat([mean_h, last_h], dim=-1))
        return self.head(pooled)
