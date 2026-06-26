from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=True, text=True, capture_output=True)


def test_r4_smoke_train_validate_test_export():
    run_cmd(["train_r4.py", "--config", "configs/r4_smoke_cpu.yaml", "--max-iterations", "2"])
    ckpt = ROOT / "outputs/SAGE_SAM_R4_Smoke/checkpoints/latest.pth"
    assert ckpt.exists()
    run_cmd(["validate_r4.py", "--config", "outputs/SAGE_SAM_R4_Smoke/resolved_config.yaml", "--checkpoint", "outputs/SAGE_SAM_R4_Smoke/checkpoints/latest.pth"])
    run_cmd(["test_r4.py", "--config", "outputs/SAGE_SAM_R4_Smoke/resolved_config.yaml", "--checkpoint", "outputs/SAGE_SAM_R4_Smoke/checkpoints/latest.pth", "--save-pred", "--split", "test"])
    out = ROOT / "outputs/SAGE_SAM_R4_Smoke/checkpoints/deploy_student.pth"
    run_cmd(["export_deploy_checkpoint.py", "--checkpoint", "outputs/SAGE_SAM_R4_Smoke/checkpoints/latest.pth", "--output", str(out)])
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert "model" in payload
    assert not any("teacher" in k.lower() or "sam" in k.lower() or "optimizer" in k.lower() for k in payload["model"])
