from __future__ import annotations

import torch


def per_class_dice_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    valid = target != ignore_index
    dice = []
    iou = []
    for c in range(num_classes):
        p = (pred == c) & valid
        t = (target == c) & valid
        inter = (p & t).sum().item()
        ps = p.sum().item()
        ts = t.sum().item()
        union = (p | t).sum().item()
        dice.append((2 * inter + 1e-6) / (ps + ts + 1e-6))
        iou.append((inter + 1e-6) / (union + 1e-6))
    return dice, iou


def average_foreground(values):
    if len(values) <= 1:
        return float(values[0]) if values else 0.0
    return float(sum(values[1:]) / max(1, len(values) - 1))

