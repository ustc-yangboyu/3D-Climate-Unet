from typing import Dict

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        max_len: int = 1000
    ):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:x.size(1), :] + x


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        hour_dim: int = 24,
        day_dim: int = 32,
        month_dim: int = 13,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.hour_embedding = nn.Embedding(hour_dim, embed_dim)
        self.day_embedding = nn.Embedding(day_dim, embed_dim)
        self.month_embedding = nn.Embedding(month_dim, embed_dim)

        self.projection1 = nn.Linear(embed_dim * 3, embed_dim)
        self.GELU = nn.GELU()
        self.projection2 = nn.Linear(embed_dim, embed_dim)

        self.positional_embedding = PositionalEmbedding(embed_dim)

    def forward(self, time: Dict[str, torch.Tensor]) -> torch.Tensor:
        hour_embedding = self.hour_embedding(time['hour'])
        day_embedding = self.day_embedding(time['day'])
        month_embedding = self.month_embedding(time['month'])

        x = torch.cat([hour_embedding, day_embedding, month_embedding], dim = -1)
        x = self.projection1(x)
        x = self.GELU(x)
        x = self.projection2(x)

        return x


class TerrainEmbedding(nn.Module):
    def __init__(self, terrain_data: torch.Tensor = None):
        super().__init__()
        self.terrain = nn.Parameter(terrain_data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, H, W = x.shape
        terrain = self.terrain.unsqueeze(0).expand(B, -1, T, -1, -1)
        return torch.cat([terrain, x], dim = 1)
