from __future__ import annotations

import torch


def build_set_valued_targets(teacher_out: dict, sam_out: dict | None, calibrator, config: dict):
    prob = teacher_out["mean_prob"]
    if sam_out and sam_out.get("valid") and "sam_prob" in sam_out:
        prob = 0.5 * (prob + sam_out["sam_prob"].to(prob.device))
    conf, argmax = prob.max(dim=1)
    candidate_set, low_reliability = calibrator.prediction_sets(prob)
    max_set = int(config.get("max_candidate_set_size", 2))
    if max_set > 0:
        topv, topi = prob.topk(k=min(max_set, prob.shape[1]), dim=1)
        top_candidate = torch.zeros_like(candidate_set)
        top_candidate.scatter_(1, topi, True)
        candidate_set = candidate_set & top_candidate
        empty = candidate_set.sum(dim=1, keepdim=True) == 0
        candidate_set = torch.where(empty, top_candidate, candidate_set)
    singleton_mask = (candidate_set.sum(dim=1) == 1) & (conf >= float(config.get("min_teacher_confidence", 0.5))) & ~low_reliability
    ambiguous_mask = (candidate_set.sum(dim=1) > 1) | low_reliability
    negative_set = prob < float(config.get("safe_negative_threshold", 0.05))
    negative_mask = negative_set.any(dim=1)
    stats = {
        "singleton_ratio": float(singleton_mask.float().mean().detach()),
        "ambiguous_ratio": float(ambiguous_mask.float().mean().detach()),
        "negative_ratio": float(negative_mask.float().mean().detach()),
        "avg_set_size": float(candidate_set.float().sum(dim=1).mean().detach()),
        "conflict_ratio": float(low_reliability.float().mean().detach()),
    }
    return {
        "singleton_label": argmax,
        "singleton_mask": singleton_mask,
        "candidate_set": candidate_set,
        "ambiguous_mask": ambiguous_mask,
        "negative_set": negative_set,
        "negative_mask": negative_mask,
        "stats": stats,
    }

