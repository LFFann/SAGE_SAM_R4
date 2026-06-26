from __future__ import annotations

import torch

from r4.calibration.prompt_reliability_calibrator import PromptReliabilityCalibrator


def test_prompt_reliability_uses_separate_sam_iou_threshold():
    cal = PromptReliabilityCalibrator(2, min_pixels_per_class=1)
    cal.teacher_q = torch.tensor([0.2, 0.2])
    cal.sam_q = torch.tensor([0.2, 0.2])
    cal.sam_iou_q = torch.tensor([0.9, 0.9])
    cal.prompt_stability_q = torch.tensor([0.2, 0.2])
    cal.prompt_q = cal.prompt_stability_q
    teacher_prob = torch.tensor([[[[0.1, 0.1], [0.1, 0.1]], [[0.9, 0.9], [0.9, 0.9]]]])
    sam_prob = teacher_prob.clone()
    low_iou = torch.tensor([[1.0, 0.4]])
    prompt_quality = torch.tensor([[1.0, 1.0]])
    gates = cal.gates(teacher_prob, sam_prob=sam_prob, sam_iou=low_iou, prompt_quality=prompt_quality)
    assert gates["semantic_gate"].all()
    assert not gates["sam_train_gate"].any()
    assert not gates["structure_gate"].any()
