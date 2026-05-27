import matplotlib.pyplot as plt
from einops import rearrange
from typing import Dict
import torch.nn as nn
import numpy as np
import torch

class PositonalEmbedding(nn.Module):
    def __init__(
        self, 
        embed_dim : int = 64, 
        max_len : int = 1000
    ):
        super(PositonalEmbedding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.pe[: x.size(1), :] + x

class TimeEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim : int = 64,
        hour_dim : int = 24,
        day_dim : int = 32,
        month_dim : int = 13,
    ):
        super(TimeEmbedding, self).__init__()

        self.embed_dim = embed_dim
        self.hour_dim = hour_dim
        self.day_dim = day_dim
        self.month_dim = month_dim

        self.hour_embedding = nn.Embedding(hour_dim, embed_dim)
        self.day_embedding = nn.Embedding(day_dim, embed_dim)
        self.month_embedding = nn.Embedding(month_dim, embed_dim)

        self.projection1 = nn.Linear(embed_dim * 3, embed_dim)
        self.GELU = nn.GELU()
        self.projection2 = nn.Linear(embed_dim, embed_dim)

        self.positional_embedding = PositonalEmbedding(embed_dim)

    def forward(self, time : Dict[str, torch.Tensor]) -> torch.Tensor:
        hour_embedding = self.hour_embedding(time['hour'])
        day_embedding = self.day_embedding(time['day'])
        month_embedding = self.month_embedding(time['month'])

        x = torch.cat([hour_embedding, day_embedding, month_embedding], dim = -1)
        x = self.projection1(x)
        x = self.GELU(x)
        x = self.projection2(x)

        return x
    
class TerrainEmbedding(nn.Module):
    def __init__(
        self,
        terrain_data : torch.Tensor = None,
    ):
        super(TerrainEmbedding, self).__init__()
        self.terrain = nn.Parameter(terrain_data)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, _, T, H, W = x.shape
        terrain = self.terrain.unsqueeze(0).expand(B, -1, T, -1, -1)
        return torch.cat([terrain, x], dim = 1)  # (B, 2+C, T, H, W)
    
class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels : int, 
        out_channels : int, 
        kernel_size : int = 3, 
        stride : int = 1,   
    ):
        super(ConvBlock, self).__init__()
        pad = kernel_size // 2
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding = pad)
        self.gn = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU(inplace = True)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = self.gn(x)
        x = self.silu(x)
        return x
    
class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
            nn.SiLU(inplace = True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        channel_weight = self.sigmoid(avg_out + max_out)
        out = x * channel_weight
        return out

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size = 3):
        super(SpatialAttention3D, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, (1, kernel_size, kernel_size), padding=(0, padding, padding))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)

        spatial_feat = torch.cat([avg_out, max_out], dim = 1)
        spatial_weight = self.sigmoid(self.conv(spatial_feat))
        out = x * spatial_weight
        return out

class TemporalAttention3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 8):
        super(TemporalAttention3D, self).__init__()
        hidden_dim = in_channels // reduction_ratio
        
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 1),
            nn.SiLU(inplace = True),
            nn.Conv3d(hidden_dim, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        temporal_weight = self.sigmoid(avg_out + max_out)
        out = x * temporal_weight
        return out


class Downsample3D(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.temporal_attn = TemporalAttention3D(out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.spatial_attn = SpatialAttention3D()

    def forward(self, x):
        x = self.conv1(x)
        x = self.temporal_attn(x)
        x = self.conv2(x)
        x = self.spatial_attn(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_up=True):
        super().__init__()
        self.use_up = use_up
        if use_up:
            self.up = Upsample3D(in_channels, out_channels)
        self.conv1 = ConvBlock(out_channels + skip_channels, out_channels)
        self.channel_attn = ChannelAttention3D(out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip = None):
        if self.use_up:
            x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim = 1)
        x = self.conv1(x)
        x = self.channel_attn(x)
        x = self.conv2(x)
        return x


class ClimateUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 12,
        base_channels: int = 64,
        H: int = 52,
        W: int = 48,
        terrain_data: torch.Tensor = None,
        static_terrain_data: torch.Tensor = None,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.base_channels = base_channels

        self.time_embed = TimeEmbedding(embed_dim=64)
        self.terrain_embed = TerrainEmbedding(terrain_data=terrain_data)
        self.register_buffer('static_terrain', static_terrain_data)
        self.time_proj = nn.Sequential(
            nn.Linear(64, base_channels),
            nn.GELU(),
            nn.Linear(base_channels, base_channels)
        )

        self.input_proj = nn.Sequential(
            ConvBlock(15, base_channels),
            ConvBlock(base_channels, base_channels)
        )

        self.enc1 = EncoderBlock(base_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)

        self.down1 = Downsample3D(base_channels)
        self.down2 = Downsample3D(base_channels * 2)

        self.dec1 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec2 = DecoderBlock(base_channels * 2, base_channels)
        self.dec3 = DecoderBlock(base_channels, base_channels, use_up=False)

        self.conv_out = ConvBlock(base_channels, base_channels)
        self.output_proj = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 3, 3), stride=(6, 1, 1), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=1)
        )

    def _add_time_embed(self, x, time_embed):
        B, C, T, H, W = x.shape
        time_embed = self.time_proj(time_embed)
        time_embed = time_embed.unsqueeze(1).expand(B, T, self.base_channels)
        time_embed = time_embed.reshape(B * T, self.base_channels)
        time_embed = nn.functional.linear(time_embed, torch.eye(C, self.base_channels, device=x.device))
        time_embed = time_embed.reshape(B, T, C)
        time_embed = time_embed.unsqueeze(-1).unsqueeze(-1)
        time_embed = time_embed.expand(B, T, C, H, W)
        time_embed = time_embed.permute(0, 2, 1, 3, 4)
        return x + time_embed

    def forward(self, x, time_dict):
        orig_H, orig_W = x.shape[3], x.shape[4]

        target_H = ((orig_H + 7) // 8) * 8
        target_W = ((orig_W + 7) // 8) * 8
        pad_H = target_H - orig_H
        pad_W = target_W - orig_W

        if pad_H > 0 or pad_W > 0:
            x = nn.functional.pad(x, (0, pad_W, 0, pad_H))

        time_embed = self.time_embed(time_dict)

        terrain = self.terrain_embed.terrain
        terrain = terrain.unsqueeze(0).unsqueeze(0)
        if pad_H > 0 or pad_W > 0:
            terrain = nn.functional.pad(terrain, (0, pad_W, 0, pad_H))
        terrain = terrain.expand(x.size(0), x.size(2), -1, -1, -1)
        terrain = terrain.permute(0, 2, 1, 3, 4)

        static_terrain = self.static_terrain
        if pad_H > 0 or pad_W > 0:
            static_terrain = nn.functional.pad(static_terrain, (0, pad_W, 0, pad_H))
        static_terrain = static_terrain.unsqueeze(0)
        static_terrain = static_terrain.expand(x.size(0), x.size(2), -1, -1, -1)
        static_terrain = static_terrain.permute(0, 2, 1, 3, 4)
        terrain = torch.cat([terrain, static_terrain], dim=1)

        x = torch.cat([terrain, x], dim = 1)
        x = self.input_proj(x)
        x = self._add_time_embed(x, time_embed)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d = self.dec1(e3)
        d = self.dec2(d)
        d = self.dec3(d)
        d = self.conv_out(d)

        if pad_H > 0 or pad_W > 0:
            d = d[:, :, :, :orig_H, :orig_W]

        out = self.output_proj(d)
        return out