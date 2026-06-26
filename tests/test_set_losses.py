from __future__ import annotations

import torch

from r4.losses.set_valued_losses import rank_margin_loss, safe_negative_loss, set_cross_entropy_loss, singleton_ce_loss


def test_singleton_empty_mask_no_nan():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    loss = singleton_ce_loss(logits, labels, torch.zeros(1, 4, 4, dtype=torch.bool))
    assert torch.isfinite(loss)


def test_set_cross_entropy_backward():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    candidate = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
    candidate[:, :2] = True
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    loss = set_cross_entropy_loss(logits, candidate, mask)
    loss.backward()
    assert logits.grad is not None


def test_rank_and_negative_losses():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    candidate = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
    candidate[:, 1] = True
    negative = torch.zeros_like(candidate)
    negative[:, 2] = True
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    loss = rank_margin_loss(logits, candidate, mask) + safe_negative_loss(logits, negative, mask)
    loss.backward()
    assert torch.isfinite(loss)

