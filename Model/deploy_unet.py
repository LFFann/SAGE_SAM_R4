from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_head import BoundaryHead
from .feature_dropout import complementary_channel_dropout


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.conv = ConvBlock(skip_channels * 2, out_channels, 0.0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class DeployUNet(nn.Module):
    """KnowSAM-style five-level UNet used as the only deployable R4 network."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        use_boundary_head: bool = True,
        complementary_dropout_p: float = 0.2,
    ):
        super().__init__()
        ch = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.complementary_dropout_p = complementary_dropout_p
        self.enc0 = ConvBlock(in_channels, ch[0], 0.05)
        self.enc1 = DownBlock(ch[0], ch[1], 0.10)
        self.enc2 = DownBlock(ch[1], ch[2], 0.20)
        self.enc3 = DownBlock(ch[2], ch[3], 0.30)
        self.enc4 = DownBlock(ch[3], ch[4], 0.50)
        self.up3 = UpBlock(ch[4], ch[3], ch[3])
        self.up2 = UpBlock(ch[3], ch[2], ch[2])
        self.up1 = UpBlock(ch[2], ch[1], ch[1])
        self.up0 = UpBlock(ch[1], ch[0], ch[0])
        self.out_conv = nn.Conv2d(ch[0], num_classes, kernel_size=3, padding=1)
        self.boundary_head = BoundaryHead(ch[0]) if use_boundary_head else None

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return [x0, x1, x2, x3, x4]

    def decode(self, features: list[torch.Tensor], feature_dropout=None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x0, x1, x2, x3, x4 = features
        if feature_dropout == "complementary":
            x4, _ = complementary_channel_dropout(x4, self.complementary_dropout_p)
        d3 = self.up3(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)
        return self.out_conv(d0), [d3, d2, d1, d0]

    def forward(self, x: torch.Tensor, return_features: bool = False, feature_dropout=None):
        features = self.encode(x)
        logits, decoder_features = self.decode(features, feature_dropout=feature_dropout)
        if not return_features:
            return logits
        out = {
            "logits": logits,
            "encoder_features": features,
            "decoder_features": decoder_features,
            "bottleneck": features[-1],
        }
        if self.boundary_head is not None:
            out["boundary_logits"] = self.boundary_head(decoder_features[-1])
        return out

