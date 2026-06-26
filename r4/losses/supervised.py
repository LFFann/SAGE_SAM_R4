from __future__ import annotations

import torch.nn.functional as F

from .dice import multiclass_dice_loss


def supervised_loss(logits, target, num_classes: int, ignore_index: int = 255):
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
    dice = multiclass_dice_loss(logits, target, num_classes, ignore_index)
    return ce + dice, {"loss_sup_ce": float(ce.detach()), "loss_sup_dice": float(dice.detach())}

