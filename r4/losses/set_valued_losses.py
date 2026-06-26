from __future__ import annotations

import torch
import torch.nn.functional as F


def singleton_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, ignore_index: int = 255):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    labels = labels.clone()
    labels[~mask] = ignore_index
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def _weighted_mean(loss_map: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None):
    selected = loss_map[mask]
    if selected.numel() == 0:
        return loss_map.new_tensor(0.0)
    if weight is None:
        return selected.mean()
    w = weight.to(loss_map.device).float()[mask].clamp_min(0.0)
    denom = w.sum().clamp_min(1e-6)
    return (selected * w).sum() / denom


def set_cross_entropy_loss(logits: torch.Tensor, candidate_set: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    log_probs = torch.log_softmax(logits, dim=1)
    cand = candidate_set.bool()
    fill = torch.finfo(log_probs.dtype).min
    log_sum = torch.logsumexp(log_probs.masked_fill(~cand, fill), dim=1)
    return _weighted_mean(-log_sum, mask, weight)


def rank_margin_loss(logits: torch.Tensor, candidate_set: torch.Tensor, mask: torch.Tensor, margin: float = 0.5, weight: torch.Tensor | None = None):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    candidate_set = candidate_set.bool()
    pos_min = logits.masked_fill(~candidate_set, torch.finfo(logits.dtype).max).min(dim=1).values
    neg_max = logits.masked_fill(candidate_set, torch.finfo(logits.dtype).min).max(dim=1).values
    loss = F.relu(margin + neg_max - pos_min)
    return _weighted_mean(loss, mask, weight)


def safe_negative_loss(logits: torch.Tensor, negative_set: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None):
    if mask.sum() == 0 or negative_set.sum() == 0:
        return logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=1)
    selected = negative_set.bool() & mask.unsqueeze(1).bool()
    if selected.sum() == 0:
        return logits.new_tensor(0.0)
    per_class = -torch.log((1.0 - probs).clamp_min(1e-6))
    if weight is None:
        return per_class[selected].mean()
    w = weight.to(logits.device).float().unsqueeze(1).expand_as(per_class)[selected].clamp_min(0.0)
    return (per_class[selected] * w).sum() / w.sum().clamp_min(1e-6)


def soft_fuzzy_positive_loss(logits: torch.Tensor, soft_target: torch.Tensor | None, mask: torch.Tensor, weight: torch.Tensor | None = None):
    if soft_target is None or mask.sum() == 0:
        return logits.new_tensor(0.0)
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(soft_target.detach() * log_probs).sum(dim=1)
    return _weighted_mean(loss, mask, weight)


def set_valued_supervision_loss(logits: torch.Tensor, targets: dict, rank_margin: float = 0.5):
    labels = targets["singleton_label"]
    singleton_mask = targets["singleton_mask"].bool()
    candidate_set = targets["candidate_set"].bool()
    ambiguous_mask = targets["ambiguous_mask"].bool()
    negative_set = targets["negative_set"].bool()
    negative_mask = targets["negative_mask"].bool()
    soft_target = targets.get("soft_target")
    candidate_weight = targets.get("candidate_weight")
    negative_weight = targets.get("safe_negative_weight")
    l_single = singleton_ce_loss(logits, labels, singleton_mask)
    l_set = set_cross_entropy_loss(logits, candidate_set, ambiguous_mask, candidate_weight)
    l_rank = rank_margin_loss(logits, candidate_set, ambiguous_mask, rank_margin, candidate_weight)
    l_neg = safe_negative_loss(logits, negative_set, negative_mask, negative_weight)
    l_fuzzy = soft_fuzzy_positive_loss(logits, soft_target, ambiguous_mask, candidate_weight)
    return {
        "loss_singleton": l_single,
        "loss_set": l_set,
        "loss_rank": l_rank,
        "loss_negative": l_neg,
        "loss_fuzzy": l_fuzzy,
    }
