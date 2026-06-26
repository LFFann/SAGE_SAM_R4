from __future__ import annotations

import torch


def build_set_valued_targets(teacher_out: dict, sam_out: dict | None, calibrator, config: dict):
    teacher_prob = teacher_out["mean_prob"].detach()
    sam_valid = bool(sam_out and sam_out.get("valid") and "sam_prob" in sam_out)
    sam_prob = sam_out["sam_prob"].to(teacher_prob.device) if sam_valid else None
    sam_iou = sam_out.get("sam_iou").to(teacher_prob.device) if sam_valid and sam_out.get("sam_iou") is not None else None
    prompt_quality = (
        sam_out.get("prompt_quality").to(teacher_prob.device)
        if sam_valid and sam_out.get("prompt_quality") is not None
        else None
    )

    teacher_candidate, teacher_low = calibrator.prediction_sets(teacher_prob)
    if sam_valid:
        min_sam_conf = float(config.get("min_sam_confidence", 0.5))
        sam_candidate = sam_prob >= min_sam_conf
        sam_empty = sam_candidate.sum(dim=1) == 0
        if sam_empty.any():
            sam_candidate.scatter_(1, sam_prob.argmax(dim=1, keepdim=True), True)
        intersection = teacher_candidate & sam_candidate
        union = teacher_candidate | sam_candidate
        candidate_set = torch.where((intersection.sum(dim=1, keepdim=True) > 0), intersection, union)
        combined_prob = 0.5 * (teacher_prob + sam_prob.detach())
        gates = calibrator.gates(teacher_prob, sam_prob.detach(), sam_iou, prompt_quality) if hasattr(calibrator, "gates") else {}
        teacher_arg = teacher_prob.argmax(dim=1)
        sam_arg = sam_prob.argmax(dim=1)
        teacher_conf = teacher_prob.max(dim=1).values
        sam_conf = sam_prob.max(dim=1).values
        severe_conflict = (
            (teacher_arg != sam_arg)
            & (teacher_conf >= float(config.get("min_teacher_confidence", 0.5)))
            & (sam_conf >= min_sam_conf)
        )
    else:
        candidate_set = teacher_candidate
        combined_prob = teacher_prob
        gates = calibrator.gates(teacher_prob) if hasattr(calibrator, "gates") else {}
        severe_conflict = teacher_low

    max_set = int(config.get("max_candidate_set_size", 2))
    if max_set > 0:
        _, topi = combined_prob.topk(k=min(max_set, combined_prob.shape[1]), dim=1)
        top_candidate = torch.zeros_like(candidate_set)
        top_candidate.scatter_(1, topi, True)
        candidate_set = candidate_set & top_candidate
        empty = candidate_set.sum(dim=1, keepdim=True) == 0
        candidate_set = torch.where(empty, top_candidate, candidate_set)

    conf, argmax = combined_prob.max(dim=1)
    teacher_conf, teacher_label = teacher_prob.max(dim=1)
    semantic_gate = gates.get("semantic_gate", conf >= float(config.get("min_teacher_confidence", 0.5))).bool()
    sam_train_gate = gates.get("sam_train_gate", semantic_gate).bool()
    structure_gate = gates.get("structure_gate", semantic_gate).bool()
    semantic_weight = gates.get("semantic_weight", semantic_gate.float()).to(teacher_prob.device).float()
    sam_train_weight = gates.get("sam_train_weight", sam_train_gate.float()).to(teacher_prob.device).float()
    structure_weight = gates.get("structure_weight", structure_gate.float()).to(teacher_prob.device).float()
    teacher_weight = gates.get("teacher_weight", teacher_conf).to(teacher_prob.device).float()
    teacher_reliable_mask = teacher_conf >= float(config.get("min_teacher_confidence", 0.5))
    if sam_valid:
        teacher_reliable_mask = teacher_reliable_mask & ~severe_conflict
    candidate_count = candidate_set.sum(dim=1)
    singleton_mask = (
        (candidate_count == 1)
        & (conf >= float(config.get("min_teacher_confidence", 0.5)))
        & semantic_gate
        & ~severe_conflict
    )
    ambiguous_mask = ((candidate_count > 1) | teacher_low | (~singleton_mask & (candidate_count > 0))) & ~severe_conflict
    conflict_mask = severe_conflict | (~semantic_gate & (candidate_count > 1))
    negative_thresh = float(config.get("safe_negative_threshold", 0.05))
    if sam_valid:
        negative_set = (teacher_prob < negative_thresh) & (sam_prob.detach() < negative_thresh)
    else:
        negative_set = teacher_prob < negative_thresh
    negative_set = negative_set & ~candidate_set
    negative_mask = negative_set.any(dim=1) | conflict_mask
    soft_target = combined_prob / combined_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
    teacher_only_soft_target = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
    candidate_weight = torch.maximum(semantic_weight, teacher_weight).clamp(0.05, 1.0)
    safe_negative_weight = torch.maximum(1.0 - candidate_weight, negative_mask.float()).clamp(0.0, 1.0)
    per_class_participation = []
    per_class_safe_negative = []
    for c in range(teacher_prob.shape[1]):
        cls_mask = argmax == c
        if cls_mask.any():
            per_class_participation.append(float((sam_train_weight[cls_mask] > 0.05).float().mean().detach()))
        else:
            per_class_participation.append(0.0)
        per_class_safe_negative.append(float(negative_set[:, c].float().mean().detach()))
    sam_weight_flat = sam_train_weight.detach().reshape(-1)
    quantiles = torch.quantile(sam_weight_flat.float().cpu(), torch.tensor([0.25, 0.50, 0.75])) if sam_weight_flat.numel() else torch.zeros(3)
    stats = {
        "singleton_ratio": float(singleton_mask.float().mean().detach()),
        "singleton_pixel_ratio": float(singleton_mask.float().mean().detach()),
        "ambiguous_ratio": float(ambiguous_mask.float().mean().detach()),
        "ambiguous_pixel_ratio": float(ambiguous_mask.float().mean().detach()),
        "conflict_ratio": float(conflict_mask.float().mean().detach()),
        "negative_ratio": float(negative_mask.float().mean().detach()),
        "safe_negative_pixel_ratio": float(negative_mask.float().mean().detach()),
        "per_class_safe_negative_ratio": per_class_safe_negative,
        "avg_set_size": float(candidate_set.float().sum(dim=1).mean().detach()),
        "sam_semantic_gate_ratio": float(semantic_gate.float().mean().detach()),
        "sam_structure_gate_ratio": float(structure_gate.float().mean().detach()),
        "sam_train_gate_ratio": float(sam_train_gate.float().mean().detach()),
        "sam_soft_weight_mean": float(sam_train_weight.mean().detach()),
        "sam_soft_weight_p25": float(quantiles[0]),
        "sam_soft_weight_p50": float(quantiles[1]),
        "sam_soft_weight_p75": float(quantiles[2]),
        "sam_participation_ratio": float((sam_train_weight > 0.05).float().mean().detach()),
        "per_class_sam_participation_ratio": per_class_participation,
        "sam_teacher_agreement": float(gates.get("teacher_sam_agreement", teacher_prob.new_tensor(1.0)).detach()),
    }
    return {
        "singleton_label": argmax,
        "singleton_mask": singleton_mask,
        "candidate_set": candidate_set,
        "candidate_weight": candidate_weight.detach(),
        "ambiguous_mask": ambiguous_mask,
        "conflict_mask": conflict_mask,
        "negative_set": negative_set,
        "safe_negative_set": negative_set,
        "negative_mask": negative_mask,
        "safe_negative_weight": safe_negative_weight.detach(),
        "semantic_gate": semantic_gate,
        "sam_train_gate": sam_train_gate,
        "structure_gate": structure_gate,
        "sam_weight": sam_train_weight.detach(),
        "teacher_weight": teacher_weight.detach(),
        "semantic_weight": semantic_weight.detach(),
        "structure_weight": structure_weight.detach(),
        "teacher_reliable_mask": teacher_reliable_mask,
        "soft_target": soft_target.detach(),
        "teacher_only_soft_target": teacher_only_soft_target.detach(),
        "stats": stats,
    }
