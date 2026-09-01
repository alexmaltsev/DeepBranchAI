from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def norm3d(channels: int, norm: str = "instance") -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "group":
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm: {norm}")


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm3d(out_channels, norm),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm3d(out_channels, norm),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass(frozen=True)
class UNet3DOutput:
    logits: torch.Tensor


class UNet3D(nn.Module):
    """Plain 3D U-Net baseline with same-shape binary logits."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        levels: int = 4,
        norm: str = "instance",
    ):
        super().__init__()
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.levels = int(levels)
        self.norm = norm

        channels = [base_channels * (2**i) for i in range(levels)]
        self.encoders = nn.ModuleList()
        current = in_channels
        for channel in channels:
            self.encoders.append(ConvBlock3D(current, channel, norm))
            current = channel

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(levels - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose3d(channels[index], channels[index - 1], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(channels[index - 1] * 2, channels[index - 1], norm))

        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> UNet3DOutput:
        skips: list[torch.Tensor] = []
        out = x
        for index, encoder in enumerate(self.encoders):
            out = encoder(out)
            if index < len(self.encoders) - 1:
                skips.append(out)
                out = self.pool(out)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = upconv(out)
            if out.shape[-3:] != skip.shape[-3:]:
                out = F.interpolate(out, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            out = decoder(torch.cat([skip, out], dim=1))

        return UNet3DOutput(logits=self.head(out))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
