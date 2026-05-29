import torch
import torch.nn as nn

from .blocks import ConvBlock, DecoderBlock, Downsample3D, EncoderBlock
from .embedding import TerrainEmbedding, TimeEmbedding


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

        self.time_embed = TimeEmbedding(embed_dim = 64)
        self.terrain_embed = TerrainEmbedding(terrain_data = terrain_data)
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
            nn.Conv3d(base_channels, base_channels, kernel_size = (3, 3, 3), stride = (6, 1, 1), padding = (1, 1, 1)),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size = 1)
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
        terrain = torch.cat([terrain, static_terrain], dim =   1)

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
