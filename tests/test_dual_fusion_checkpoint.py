from __future__ import annotations

import torch

from r4.engine.checkpoint import export_deploy_payload, save_checkpoint, safe_load
from r4.engine.model_factory import build_deploy_model, minimal_deploy_config


def test_dual_fusion_export_round_trip(tmp_path):
    cfg = {
        "data": {"in_channels": 3, "num_classes": 3, "image_size": 32},
        "model": {
            "deploy_backbone": "dual_fusion",
            "base_channels": 4,
            "fusion_hidden_channels": 4,
            "use_boundary_head": False,
        },
    }
    model = build_deploy_model(cfg)
    ckpt = tmp_path / "latest.pth"
    deploy = tmp_path / "deploy.pth"

    save_checkpoint(
        ckpt,
        iteration=1,
        student=model,
        fast_teacher=None,
        slow_teacher=None,
        optimizer=None,
        scaler=None,
        calibrator=None,
        sam_utility=None,
        mentor=None,
        config=cfg,
        best_metrics={},
    )
    export_deploy_payload(ckpt, deploy)
    payload = safe_load(deploy)
    loaded = build_deploy_model(minimal_deploy_config(payload))
    loaded.load_state_dict(payload["model"], strict=True)

    out = loaded(torch.randn(1, 3, 32, 32))

    assert payload["model_name"] == "SAGE_SAM_R4_DualFusionDeploy"
    assert out.shape == (1, 3, 32, 32)
    assert not any(
        any(token in key.lower() for token in ("sam", "teacher", "mentor", "calibrator", "optimizer"))
        for key in payload["model"]
    )
