from __future__ import annotations

import torch

from r4.calibration.prompt_reliability_calibrator import PromptReliabilityCalibrator
from r4.ssl.target_builder import build_set_valued_targets


def test_target_builder_always_returns_set_and_safe_negative_shapes():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True, min_participation_ratio=0.25)
    cal.teacher_q = torch.tensor([0.95, 0.95, 0.95])
    cal.sam_q = torch.tensor([0.95, 0.95, 0.95])
    teacher_prob = torch.full((2, 3, 5, 5), 0.01)
    teacher_prob[:, 0] = 0.98
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.98

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {"max_candidate_set_size": 2, "safe_negative_threshold": 0.05, "min_teacher_confidence": 0.5},
    )

    assert targets["candidate_set"].shape == teacher_prob.shape
    assert targets["safe_negative_set"].shape == teacher_prob.shape
    assert targets["candidate_set"].sum(dim=1).min() >= 1
    assert torch.isfinite(targets["candidate_weight"]).all()
    assert "safe_negative_pixel_ratio" in targets["stats"]
    assert len(targets["stats"]["per_class_safe_negative_ratio"]) == 3


def test_calibrator_coverage_fallback_keeps_soft_participation_nonzero():
    cal = PromptReliabilityCalibrator(
        2,
        min_pixels_per_class=1,
        use_soft_gate=True,
        min_participation_ratio=0.50,
        coverage_target=0.50,
        temperature=0.05,
    )
    cal.teacher_q = torch.tensor([1.0, 1.0])
    cal.sam_q = torch.tensor([1.0, 1.0])
    prob = torch.tensor([[[[0.60, 0.55], [0.50, 0.45]], [[0.40, 0.45], [0.50, 0.55]]]])

    gates = cal.gates(prob, prob)

    assert float(gates["sam_train_weight"].mean()) > 0.05
    assert float(gates["sam_train_gate"].float().mean()) >= 0.50
