from __future__ import annotations

import torch
import torch.nn.functional as F


def singleton_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, ignore_index: int = 255):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    labels = labels.clone()
    labels[~mask] = ignore_index
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def set_cross_entropy_loss(logits: torch.Tensor, candidate_set: torch.Tensor, mask: torch.Tensor):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    log_probs = torch.log_softmax(logits, dim=1)
    cand = candidate_set.bool()
    fill = torch.finfo(log_probs.dtype).min
    log_sum = torch.logsumexp(log_probs.masked_fill(~cand, fill), dim=1)
    return (-log_sum[mask]).mean()


def rank_margin_loss(logits: torch.Tensor, candidate_set: torch.Tensor, mask: torch.Tensor, margin: float = 0.5):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    fill = torch.finfo(logits.dtype).min
    cand_logits = logits.masked_fill(~candidate_set.bool(), fill).max(dim=1).values
    non_logits = logits.masked_fill(candidate_set.bool(), fill).max(dim=1).values
    loss = F.relu(margin - cand_logits + non_logits)
    return loss[mask].mean()


def safe_negative_loss(logits: torch.Tensor, negative_set: torch.Tensor, mask: torch.Tensor):
    if mask.sum() == 0 or negative_set.sum() == 0:
        return logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=1)
    neg_prob = (probs * negative_set.float()).sum(dim=1)
    return (-torch.log((1.0 - neg_prob).clamp_min(1e-6))[mask]).mean()


def set_valued_supervision_loss(logits: torch.Tensor, targets: dict, rank_margin: float = 0.5):
    labels = targets["singleton_label"]
    singleton_mask = targets["singleton_mask"].bool()
    candidate_set = targets["candidate_set"].bool()
    ambiguous_mask = targets["ambiguous_mask"].bool()
    negative_set = targets["negative_set"].bool()
    negative_mask = targets["negative_mask"].bool()
    l_single = singleton_ce_loss(logits, labels, singleton_mask)
    l_set = set_cross_entropy_loss(logits, candidate_set, ambiguous_mask)
    l_rank = rank_margin_loss(logits, candidate_set, ambiguous_mask, rank_margin)
    l_neg = safe_negative_loss(logits, negative_set, negative_mask)
    return {
        "loss_singleton": l_single,
        "loss_set": l_set,
        "loss_rank": l_rank,
        "loss_negative": l_neg,
    }
