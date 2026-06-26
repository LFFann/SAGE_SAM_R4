from __future__ import annotations

import torch
import torch.nn.functional as F


def build_boundary_target(mask: torch.Tensor):
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    m = mask.float().unsqueeze(1)
    gx = F.pad((m[:, :, :, 1:] != m[:, :, :, :-1]).float(), (0, 1, 0, 0))
    gy = F.pad((m[:, :, 1:, :] != m[:, :, :-1, :]).float(), (0, 0, 0, 1))
    return torch.clamp(gx + gy, 0, 1)

