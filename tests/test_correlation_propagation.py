from __future__ import annotations

import torch

from r4.ssl.correlation_propagation import correlation_propagation_loss, propagate_correlation_targets


def test_correlation_propagation_returns_dense_training_signal():
    feature = torch.randn(2, 4, 8, 8)
    prob = torch.softmax(torch.randn(2, 3, 32, 32), dim=1)
    reliable = torch.zeros(2, 32, 32, dtype=torch.bool)
    reliable[:, 8:16, 8:16] = True
    sam_shape = torch.ones(2, 1, 32, 32)

    propagated = propagate_correlation_targets(
        feature,
        prob,
        sam_shape=sam_shape,
        reliable_mask=reliable,
        resolution=8,
        topk=4,
        min_weight=0.05,
    )

    assert propagated["propagated_label"].shape == (2, 32, 32)
    assert propagated["propagated_weight"].shape == (2, 32, 32)
    assert propagated["expanded_reliable_mask"].shape == (2, 32, 32)
    assert propagated["propagated_weight"].mean() > 0


def test_correlation_propagation_loss_backward():
    logits = torch.randn(1, 3, 16, 16, requires_grad=True)
    propagated = {
        "propagated_label": torch.zeros(1, 16, 16, dtype=torch.long),
        "propagated_weight": torch.ones(1, 16, 16),
        "expanded_reliable_mask": torch.ones(1, 16, 16, dtype=torch.bool),
    }

    loss = correlation_propagation_loss(logits, propagated)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
