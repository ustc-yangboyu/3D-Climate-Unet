import torch
import torch.nn as nn

from .attention import ChannelAttention3D, SpatialAttention3D, TemporalAttention3D


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=pad)
        self.gn = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = self.gn(x)
        x = self.silu(x)
        return x


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

    def forward(self, x, skip=None):
        if self.use_up:
            x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.channel_attn(x)
        x = self.conv2(x)
        return x
