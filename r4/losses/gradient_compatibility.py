from __future__ import annotations

import torch


def zero_gradient_compatibility_loss(reference: torch.Tensor):
    return reference.sum() * 0.0

