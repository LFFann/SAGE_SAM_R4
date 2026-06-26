from __future__ import annotations

import torch

from r4.calibration.class_conditional_conformal import ClassConditionalConformalCalibrator


def test_conformal_prediction_sets_and_shrink():
    probs = torch.rand(2, 3, 4, 4)
    probs = probs / probs.sum(dim=1, keepdim=True)
    masks = torch.randint(0, 3, (2, 4, 4))
    cal = ClassConditionalConformalCalibrator(3, min_pixels_per_class=1000).fit(probs, masks)
    assert cal.q_per_class.shape[0] == 3
    candidate, low = cal.prediction_sets(probs)
    assert candidate.shape == probs.shape
    assert candidate.sum(dim=1).min() >= 1
    assert low.shape == masks.shape


def test_conformal_fit_accepts_amp_half_inputs():
    probs = torch.rand(2, 3, 4, 4).half()
    probs = probs / probs.sum(dim=1, keepdim=True)
    masks = torch.randint(0, 3, (2, 4, 4))
    cal = ClassConditionalConformalCalibrator(3, min_pixels_per_class=1).fit(probs, masks)
    assert cal.q_per_class.dtype == torch.float32
    assert torch.isfinite(cal.q_per_class).all()
