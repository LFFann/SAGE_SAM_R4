from __future__ import annotations

from itertools import cycle

import torch

from r4.engine.trainer import SAGESAMR4Trainer


class _FakeStudent:
    def __call__(self, x, return_features=False):
        logits = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        logits[:, 1] = 2.0
        return {"logits": logits}


class _FakeMentor:
    def __init__(self):
        self.last_image = None

    def forward_labeled(self, image, mask):
        self.last_image = image.detach().clone()
        sam_prob = torch.zeros(image.shape[0], 3, image.shape[2], image.shape[3])
        sam_prob[:, 1] = 1.0
        return {
            "valid": True,
            "sam_prob": sam_prob,
            "sam_iou": torch.ones(image.shape[0], 3),
            "prompt_quality": torch.ones(image.shape[0], 3),
        }


class _FakeCalibrator:
    def __init__(self):
        self.teacher_q = torch.zeros(3)
        self.sam_q = torch.zeros(3)
        self.sam_iou_q = torch.zeros(3)
        self.prompt_stability_q = torch.zeros(3)
        self.updated_gt_sum = None

    def should_update(self, iteration):
        return iteration == 5

    def update_from_batch(self, **kwargs):
        self.updated_gt_sum = int(kwargs["gt"].sum())


def test_prompt_calibrator_uses_calibration_split(tmp_path):
    trainer = SAGESAMR4Trainer.__new__(SAGESAMR4Trainer)
    trainer.use_sam = True
    trainer.mentor = _FakeMentor()
    trainer.student = _FakeStudent()
    trainer.calibrator = _FakeCalibrator()
    trainer.config = {"calibration": {"use_calibration_split": True}}
    trainer.device = torch.device("cpu")
    trainer.output_dir = tmp_path
    cal_image = torch.ones(1, 3, 4, 4)
    cal_mask = torch.ones(1, 4, 4, dtype=torch.long)
    trainer.calibration_iter = cycle([{"image": cal_image, "mask": cal_mask}])

    fallback_y = torch.zeros(1, 4, 4, dtype=torch.long)
    trainer._maybe_update_prompt_calibrator(
        5,
        fallback_out={"logits": torch.zeros(1, 3, 4, 4)},
        fallback_y=fallback_y,
        fallback_sam={"valid": True, "sam_prob": torch.ones(1, 3, 4, 4) / 3},
    )

    assert trainer.calibrator.updated_gt_sum == 16
    assert torch.equal(trainer.mentor.last_image, cal_image)


def test_self_reliance_default_starts_at_seventy_percent():
    trainer = SAGESAMR4Trainer.__new__(SAGESAMR4Trainer)
    trainer.config = {"sam": {"self_reliance_decay": 0.5, "self_reliance_min_weight": 0.05}, "train": {"max_iterations": 100}}

    assert trainer._sam_self_reliance_scale(70) == 1.0
    assert trainer._sam_self_reliance_scale(72) == 0.25
