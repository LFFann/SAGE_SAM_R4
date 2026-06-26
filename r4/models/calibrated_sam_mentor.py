from __future__ import annotations

from .real_sam_wrapper import RealSAMWrapper


class CalibratedSAMMentor:
    def __init__(self, wrapper: RealSAMWrapper | None):
        self.wrapper = wrapper

    def available(self):
        return self.wrapper is not None and self.wrapper.sam_is_real()

    def propose(self, images, teacher_prob, ids=None, num_classes: int = 3):
        if self.wrapper is None:
            return {"valid": False}
        out = self.wrapper.propose(images, teacher_prob, ids=ids, num_classes=num_classes)
        if "sam_prob" not in out or out["sam_prob"].shape[1] != num_classes:
            raise RuntimeError("Real SAM mentor did not return class-aligned sam_prob")
        return out

