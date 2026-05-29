import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super().__init__()
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
        return x * channel_weight


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, (1, kernel_size, kernel_size), padding = (0, padding, padding))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        spatial_feat = torch.cat([avg_out, max_out], dim = 1)
        spatial_weight = self.sigmoid(self.conv(spatial_feat))
        return x * spatial_weight


class TemporalAttention3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 8):
        super().__init__()
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
        return x * temporal_weight
