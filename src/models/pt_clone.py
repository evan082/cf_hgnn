"""Faithful ProcessTransformer next-activity clone (architecture + padding).

Matches zaharah/processtransformer defaults as closely as practical in PyTorch:
  - activity-only, learned token+position embeddings (no sqrt scaling)
  - 1 Transformer block, post-norm, ReLU FFN
  - embed_dim=36, num_heads=4, ff_dim=64
  - pre-padding to max train prefix length
  - GlobalAveragePooling1D over full length (includes PAD) — same as Keras GAP
  - head: Dense(64, relu) -> Dense(num_activities)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PTTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
        # Keras MultiHeadAttention(key_dim=embed_dim) => head_dim = embed_dim (not d/h).
        self.num_heads = num_heads
        self.head_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, num_heads * embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads * embed_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads * embed_dim)
        self.out_proj = nn.Linear(num_heads * embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm_a = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm_b = nn.LayerNorm(embed_dim, eps=1e-6)
        self.drop_a = nn.Dropout(rate)
        self.drop_b = nn.Dropout(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        attn = self.drop_a(self.out_proj(attn))
        out_a = self.norm_a(x + attn)
        ffn = self.drop_b(self.ffn(out_a))
        return self.norm_b(out_a + ffn)


class ProcessTransformerClone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_activities: int,
        max_len: int,
        embed_dim: int = 36,
        num_heads: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.block = PTTransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_activities)

    def forward(self, prefix_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # prefix_ids: [B, max_len] already pre-padded
        B, T = prefix_ids.shape
        pos = torch.arange(T, device=prefix_ids.device)
        x = self.token_emb(prefix_ids) + self.pos_emb(pos)
        x = self.block(x)
        # Keras GlobalAveragePooling1D — average all timesteps including PAD
        x = x.mean(dim=1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


def prepad_batch(prefix_lists, max_len: int, pad_id: int = 0) -> torch.Tensor:
    B = len(prefix_lists)
    out = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(prefix_lists):
        seq = list(seq)[-max_len:]
        out[i, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
    return out
