from __future__ import annotations

import torch
import torch.nn.functional as F


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    probs = torch.softmax(logits, dim=1)
    valid = target != ignore_index
    target_safe = target.clamp(0, num_classes - 1)
    one_hot = F.one_hot(target_safe, num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid
    losses = []
    for c in range(1, num_classes):
        inter = (probs[:, c] * one_hot[:, c]).sum()
        denom = probs[:, c].sum() + one_hot[:, c].sum()
        losses.append(1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()

