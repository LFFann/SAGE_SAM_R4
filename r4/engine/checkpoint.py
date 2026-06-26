from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def safe_load(path: str | Path, map_location="cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(path, *, iteration, student, fast_teacher, slow_teacher, optimizer, scaler, calibrator, sam_utility, config, best_metrics, mentor=None):
    mentor_state = None
    if mentor is not None:
        mentor_state = mentor.trainable_state_dict() if hasattr(mentor, "trainable_state_dict") else mentor.state_dict()
    payload = {
        "method": "SAGE-SAM-R4",
        "iteration": int(iteration),
        "student": student.state_dict(),
        "fast_teacher": fast_teacher.state_dict() if fast_teacher is not None else None,
        "slow_teacher": slow_teacher.state_dict() if slow_teacher is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "calibrator": calibrator.state_dict() if calibrator is not None else None,
        "sam_utility": sam_utility.state_dict() if sam_utility is not None else None,
        "mentor": mentor_state,
        "config": config,
        "best_metrics": best_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_student_checkpoint(model, checkpoint_path, strict=False, map_location="cpu"):
    payload = safe_load(checkpoint_path, map_location=map_location)
    state = payload.get("student") or payload.get("model") or payload
    if hasattr(model, "branch_a") and state and not any(str(k).startswith("branch_") for k in state):
        state = {f"branch_a.{k}": v for k, v in state.items()}
    report = model.load_state_dict(state, strict=strict)
    return payload, report


def export_deploy_payload(checkpoint_path, output_path, strip_boundary: bool = False):
    payload = safe_load(checkpoint_path, map_location="cpu")
    if "student" not in payload:
        raise KeyError("Checkpoint does not contain a student state_dict")
    state = payload["student"]
    if strip_boundary:
        state = {k: v for k, v in state.items() if "boundary_head" not in k}
    forbidden = ("sam", "teacher", "mentor", "calibrator", "optimizer")
    for key in state:
        lowered = key.lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError(f"Deploy export contains forbidden key: {key}")
    config = payload.get("config", {})
    data = config.get("data", {})
    model_cfg = config.get("model", {})
    deploy = {
        "model": state,
        "num_classes": data.get("num_classes", 3),
        "in_channels": data.get("in_channels", 3),
        "model_name": "SAGE_SAM_R4_DualFusionDeploy",
        "config_minimal": {
            "deploy_backbone": model_cfg.get("deploy_backbone", "dual_fusion"),
            "base_channels": model_cfg.get("base_channels", 32),
            "fusion_hidden_channels": model_cfg.get("fusion_hidden_channels", model_cfg.get("base_channels", 32)),
            "use_boundary_head": (not strip_boundary) and model_cfg.get("use_boundary_head", True),
            "complementary_dropout_p": model_cfg.get("complementary_dropout_p", 0.2),
            "image_size": data.get("image_size", 256),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deploy, output_path)
    return output_path
