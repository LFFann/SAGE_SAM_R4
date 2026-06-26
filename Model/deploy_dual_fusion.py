from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_head import BoundaryHead
from .deploy_unet import DeployUNet
from .feature_dropout import complementary_channel_dropout


class ResidualMorphBlock(nn.Module):
    """Residual 2D block with a dilated path for morphology-biased context."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, dropout_p: float = 0.0):
        super().__init__()
        padding = int(dilation)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.proj(x))


class MorphDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout_p: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualMorphBlock(in_channels, out_channels, dilation=dilation, dropout_p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MorphUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.conv = ResidualMorphBlock(skip_channels * 2, out_channels, dilation=dilation, dropout_p=0.0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class DeployMorphUNet2D(nn.Module):
    """Second deploy branch: residual, dilated, and morphology-biased."""

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
        self.complementary_dropout_p = float(complementary_dropout_p)
        self.enc0 = ResidualMorphBlock(in_channels, ch[0], dilation=1, dropout_p=0.05)
        self.enc1 = MorphDownBlock(ch[0], ch[1], dilation=1, dropout_p=0.10)
        self.enc2 = MorphDownBlock(ch[1], ch[2], dilation=2, dropout_p=0.20)
        self.enc3 = MorphDownBlock(ch[2], ch[3], dilation=2, dropout_p=0.30)
        self.enc4 = MorphDownBlock(ch[3], ch[4], dilation=3, dropout_p=0.50)
        self.up3 = MorphUpBlock(ch[4], ch[3], ch[3], dilation=2)
        self.up2 = MorphUpBlock(ch[3], ch[2], ch[2], dilation=2)
        self.up1 = MorphUpBlock(ch[2], ch[1], ch[1], dilation=1)
        self.up0 = MorphUpBlock(ch[1], ch[0], ch[0], dilation=1)
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


class HAMLite(nn.Module):
    """Disagreement-aware fusion using branch probabilities, entropy, and features."""

    def __init__(self, num_classes: int, feature_channels: int, hidden_channels: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(feature_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),
        )
        fusion_in = num_classes * 3 + 2 + hidden_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    @staticmethod
    def entropy(prob: torch.Tensor) -> torch.Tensor:
        return -(prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()).sum(dim=1, keepdim=True)

    def forward(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        return_feature: bool = False,
    ):
        prob_a = torch.softmax(logits_a, dim=1)
        prob_b = torch.softmax(logits_b, dim=1)
        diff = (prob_a - prob_b).abs()
        ent_a = self.entropy(prob_a)
        ent_b = self.entropy(prob_b)
        context = self.context(torch.cat([feature_a, feature_b], dim=1))
        fusion_input = torch.cat([prob_a, prob_b, diff, ent_a, ent_b, context], dim=1)
        logits = self.fusion(fusion_input)
        if return_feature:
            return logits, context
        return logits


class DeployDualFusionSegmentor(nn.Module):
    """Deployable dual-view R4 model with fusion logits as the default output."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        use_boundary_head: bool = True,
        complementary_dropout_p: float = 0.2,
        fusion_hidden_channels: int | None = None,
    ):
        super().__init__()
        hidden = int(fusion_hidden_channels or base_channels)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.branch_a = DeployUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_boundary_head=use_boundary_head,
            complementary_dropout_p=complementary_dropout_p,
        )
        self.branch_b = DeployMorphUNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_boundary_head=use_boundary_head,
            complementary_dropout_p=complementary_dropout_p,
        )
        self.fusion = HAMLite(num_classes=num_classes, feature_channels=base_channels, hidden_channels=hidden)

    def forward(self, x: torch.Tensor, return_all: bool = False, return_features: bool = False, feature_dropout=None):
        want_dict = bool(return_all or return_features)
        out_a = self.branch_a(x, return_features=True, feature_dropout=feature_dropout)
        out_b = self.branch_b(x, return_features=True, feature_dropout=feature_dropout)
        logits, fusion_feature = self.fusion(
            out_a["logits"],
            out_b["logits"],
            out_a["decoder_features"][-1],
            out_b["decoder_features"][-1],
            return_feature=True,
        )
        if not want_dict:
            return logits
        out = {
            "logits": logits,
            "logits_a": out_a["logits"],
            "logits_b": out_b["logits"],
            "fusion_feature": fusion_feature,
            "bottleneck": torch.cat([out_a["bottleneck"], out_b["bottleneck"]], dim=1),
            "branch_a": out_a,
            "branch_b": out_b,
        }
        if out_a.get("boundary_logits") is not None and out_b.get("boundary_logits") is not None:
            out["boundary_logits"] = 0.5 * (out_a["boundary_logits"] + out_b["boundary_logits"])
            out["boundary_a"] = out_a["boundary_logits"]
            out["boundary_b"] = out_b["boundary_logits"]
        return out
