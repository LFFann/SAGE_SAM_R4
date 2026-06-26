from __future__ import annotations

import torch
import torch.nn.functional as F


def _gate_to_weight(gate, shape, device, dtype):
    if gate is None:
        return torch.ones(shape, device=device, dtype=dtype)
    weight = gate.to(device)
    if weight.dtype == torch.bool:
        weight = weight.float()
    return weight.to(dtype=dtype).clamp_min(0.0)


def sam_ce_dice_loss(sam_prob: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    valid = target != ignore_index
    if valid.sum() == 0:
        return sam_prob.new_tensor(0.0)
    ce_target = target.clone()
    safe_target = target.clamp(0, num_classes - 1)
    log_prob = torch.log(sam_prob.clamp_min(1e-6))
    ce = F.nll_loss(log_prob, ce_target, ignore_index=ignore_index)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    dice_terms = []
    for c in range(1, num_classes):
        p = sam_prob[:, c] * valid.float()
        t = one_hot[:, c] * valid.float()
        dice_terms.append(1.0 - (2.0 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6))
    dice = torch.stack(dice_terms).mean() if dice_terms else sam_prob.new_tensor(0.0)
    return ce + dice * valid_f.mean().clamp_min(1e-6) / valid_f.mean().clamp_min(1e-6)


def gated_soft_sam_loss(sam_prob: torch.Tensor, soft_target: torch.Tensor, gate: torch.Tensor | None = None):
    soft_target = soft_target.detach()
    weight = _gate_to_weight(gate, (sam_prob.shape[0], sam_prob.shape[2], sam_prob.shape[3]), sam_prob.device, sam_prob.dtype)
    if weight.sum() <= 0:
        return sam_prob.new_tensor(0.0)
    log_prob = torch.log(sam_prob.clamp_min(1e-6))
    ce = -(soft_target * log_prob).sum(dim=1)
    return (ce * weight).sum() / weight.sum().clamp_min(1e-6)


def sam_student_kd_loss(student_logits: torch.Tensor, sam_prob: torch.Tensor, gate: torch.Tensor | None = None, temperature: float = 1.0):
    target = sam_prob.detach()
    weight = _gate_to_weight(gate, (student_logits.shape[0], student_logits.shape[2], student_logits.shape[3]), student_logits.device, student_logits.dtype)
    if weight.sum() <= 0:
        return student_logits.new_tensor(0.0)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    kd = F.kl_div(log_student, target, reduction="none").sum(dim=1) * (temperature**2)
    return (kd * weight).sum() / weight.sum().clamp_min(1e-6)
