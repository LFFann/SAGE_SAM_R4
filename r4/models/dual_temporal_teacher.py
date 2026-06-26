from __future__ import annotations

import copy

import torch
import torch.nn as nn


class DualTemporalTeacher(nn.Module):
    def __init__(self, student: nn.Module, fast_decay: float = 0.99, slow_decay: float = 0.999, use_bn_eval: bool = True):
        super().__init__()
        self.fast = copy.deepcopy(student)
        self.slow = copy.deepcopy(student)
        self.fast_decay = float(fast_decay)
        self.slow_decay = float(slow_decay)
        self.use_bn_eval = use_bn_eval
        for model in (self.fast, self.slow):
            for p in model.parameters():
                p.requires_grad_(False)
            if use_bn_eval:
                model.eval()

    @torch.no_grad()
    def _ema_update(self, teacher: nn.Module, student: nn.Module, decay: float):
        for t, s in zip(teacher.parameters(), student.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)
        for t, s in zip(teacher.buffers(), student.buffers()):
            if t.dtype.is_floating_point:
                t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)
            else:
                t.data.copy_(s.data)

    @torch.no_grad()
    def update_fast(self, student: nn.Module):
        self._ema_update(self.fast, student, self.fast_decay)

    @torch.no_grad()
    def refresh_slow(self, student: nn.Module):
        self._ema_update(self.slow, student, self.slow_decay)

    @torch.no_grad()
    def predict_weak(self, x: torch.Tensor):
        self.fast.eval()
        self.slow.eval()
        fast_logits = self.fast(x)
        slow_logits = self.slow(x)
        fast_prob = torch.softmax(fast_logits, dim=1)
        slow_prob = torch.softmax(slow_logits, dim=1)
        mean_prob = 0.5 * (fast_prob + slow_prob)
        agreement = (fast_prob.argmax(1) == slow_prob.argmax(1)).float().mean()
        return {"fast_prob": fast_prob, "slow_prob": slow_prob, "mean_prob": mean_prob, "agreement": agreement}

