from __future__ import annotations

from Model.deploy_dual_fusion import DeployDualFusionSegmentor
from Model.deploy_unet import DeployUNet


def build_deploy_model(config: dict):
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    backbone = str(model_cfg.get("deploy_backbone", "dual_fusion")).lower()
    common = {
        "in_channels": int(data_cfg.get("in_channels", 3)),
        "num_classes": int(data_cfg.get("num_classes", 3)),
        "base_channels": int(model_cfg.get("base_channels", 32)),
        "use_boundary_head": bool(model_cfg.get("use_boundary_head", True)),
        "complementary_dropout_p": float(model_cfg.get("complementary_dropout_p", 0.2)),
    }
    if backbone in {"dual", "dual_fusion", "dual_fusion_deploy", "deploy_dual_fusion"}:
        return DeployDualFusionSegmentor(
            **common,
            fusion_hidden_channels=int(model_cfg.get("fusion_hidden_channels", common["base_channels"])),
        )
    if backbone in {"unet", "deploy_unet", "knowsam_unet"}:
        return DeployUNet(**common)
    raise ValueError(f"Unknown model.deploy_backbone={model_cfg.get('deploy_backbone')!r}")


def minimal_deploy_config(payload: dict):
    return {
        "data": {
            "in_channels": int(payload.get("in_channels", 3)),
            "num_classes": int(payload.get("num_classes", 3)),
        },
        "model": dict(payload.get("config_minimal", {})),
    }
