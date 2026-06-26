from __future__ import annotations

import torch

from Model.deploy_dual_fusion import DeployDualFusionSegmentor
from r4.engine.model_factory import build_deploy_model


def test_dual_fusion_forward_returns_fusion_logits_and_branches():
    model = DeployDualFusionSegmentor(in_channels=3, num_classes=3, base_channels=4, fusion_hidden_channels=4)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)
        out = model(x, return_all=True)

    assert logits.shape == (2, 3, 32, 32)
    assert out["logits"].shape == logits.shape
    assert out["logits_a"].shape == logits.shape
    assert out["logits_b"].shape == logits.shape
    assert out["fusion_feature"].shape[-2:] == (32, 32)
    assert out["bottleneck"].ndim == 4


def test_model_factory_defaults_to_dual_fusion():
    cfg = {
        "data": {"in_channels": 3, "num_classes": 3},
        "model": {"base_channels": 4, "use_boundary_head": False},
    }
    model = build_deploy_model(cfg)
    assert isinstance(model, DeployDualFusionSegmentor)
