from __future__ import annotations

import torch


def complementary_channel_dropout(feature: torch.Tensor, p: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    if feature.ndim != 4:
        raise ValueError("feature must be BCHW")
    if p == 0.0 or feature.shape[1] < 2:
        return feature, feature
    keep = torch.rand((feature.shape[0], feature.shape[1], 1, 1), device=feature.device) > p
    all_zero = keep.flatten(1).sum(dim=1) == 0
    if all_zero.any():
        keep[all_zero, 0] = True
    comp = ~keep
    comp_zero = comp.flatten(1).sum(dim=1) == 0
    if comp_zero.any():
        comp[comp_zero, -1] = True
    return feature * keep.float(), feature * comp.float()

