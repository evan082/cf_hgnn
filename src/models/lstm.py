"""Step 3: LSTM baseline for next-activity prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMNextActivity(nn.Module):
    def __init__(
        self,
        num_activities: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        graph_feat_dim: int = 0,
    ):
        super().__init__()
        self.num_activities = num_activities
        self.embed = nn.Embedding(num_activities, embed_dim, padding_idx=None)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        in_dim = hidden_dim + graph_feat_dim
        self.fc = nn.Linear(in_dim, num_activities)
        self.graph_feat_dim = graph_feat_dim

    def forward(
        self,
        prefix_ids: torch.Tensor,
        lengths: torch.Tensor,
        graph_feat: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # prefix_ids: [B, T]
        x = self.embed(prefix_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]  # [B, H]
        h = self.dropout(h)
        if self.graph_feat_dim > 0:
            if graph_feat is None:
                graph_feat = torch.zeros(h.size(0), self.graph_feat_dim, device=h.device)
            h = torch.cat([h, graph_feat], dim=-1)
        return self.fc(h)
