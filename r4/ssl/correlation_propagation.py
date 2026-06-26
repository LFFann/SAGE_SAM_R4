from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def propagate_correlation_targets(
    feature_fusion: torch.Tensor,
    prob_fusion: torch.Tensor,
    sam_shape: torch.Tensor | None = None,
    reliable_mask: torch.Tensor | None = None,
    resolution: int = 16,
    topk: int = 8,
    temperature: float = 0.2,
    min_weight: float = 0.15,
):
    """Propagate high-confidence seed probabilities through low-resolution feature correlation."""

    if feature_fusion.ndim != 4 or prob_fusion.ndim != 4:
        raise ValueError("feature_fusion and prob_fusion must be BCHW tensors")
    bsz, num_classes, height, width = prob_fusion.shape
    resolution = int(resolution)
    if resolution <= 1:
        empty = torch.zeros(bsz, height, width, device=prob_fusion.device)
        return {
            "propagated_label": empty.long(),
            "propagated_weight": empty,
            "expanded_reliable_mask": empty.bool(),
        }

    feat = F.interpolate(feature_fusion, size=(resolution, resolution), mode="bilinear", align_corners=False)
    feat = F.normalize(feat.flatten(2).transpose(1, 2), dim=-1)
    prob_low = F.interpolate(prob_fusion, size=(resolution, resolution), mode="bilinear", align_corners=False)
    prob_flat = prob_low.flatten(2).transpose(1, 2)
    conf_flat, _ = prob_flat.max(dim=-1)

    if reliable_mask is None:
        threshold = torch.quantile(conf_flat.detach().float().cpu(), 0.80).to(prob_fusion.device, prob_fusion.dtype)
        reliable_low = conf_flat >= threshold
    else:
        reliable_low = F.interpolate(reliable_mask.float().unsqueeze(1), size=(resolution, resolution), mode="nearest").flatten(1).bool()

    shape_flat = None
    if sam_shape is not None:
        shape = sam_shape.detach().float()
        if shape.ndim == 3:
            shape = shape.unsqueeze(1)
        shape = F.interpolate(shape, size=(resolution, resolution), mode="bilinear", align_corners=False)
        shape_flat = shape.max(dim=1).values.flatten(1).clamp(0.0, 1.0)

    labels_low = torch.zeros(bsz, resolution * resolution, device=prob_fusion.device, dtype=torch.long)
    weights_low = torch.zeros(bsz, resolution * resolution, device=prob_fusion.device, dtype=prob_fusion.dtype)
    expanded_low = torch.zeros(bsz, resolution * resolution, device=prob_fusion.device, dtype=torch.bool)

    for bi in range(bsz):
        seed = reliable_low[bi]
        if int(seed.sum()) == 0:
            k_seed = max(1, int(0.2 * seed.numel()))
            seed_idx = conf_flat[bi].topk(k_seed).indices
            seed = torch.zeros_like(seed)
            seed[seed_idx] = True
        seed_prob = prob_flat[bi, seed]
        sim = (feat[bi] @ feat[bi, seed].transpose(0, 1)) / max(float(temperature), 1e-6)
        if 0 < int(topk) < sim.shape[1]:
            vals, idx = sim.topk(int(topk), dim=1)
            local_prob = seed_prob[idx]
            local_w = torch.softmax(vals, dim=1).unsqueeze(-1)
            propagated_prob = (local_w * local_prob).sum(dim=1)
        else:
            weights = torch.softmax(sim, dim=1)
            propagated_prob = weights @ seed_prob
        propagated_conf, propagated_label = propagated_prob.max(dim=1)
        if shape_flat is not None:
            propagated_conf = propagated_conf * (0.5 + 0.5 * shape_flat[bi])
        labels_low[bi] = propagated_label
        weights_low[bi] = propagated_conf
        expanded_low[bi] = seed | (propagated_conf >= float(min_weight))

    labels = F.interpolate(
        labels_low.view(bsz, 1, resolution, resolution).float(),
        size=(height, width),
        mode="nearest",
    ).squeeze(1).long()
    weights = F.interpolate(
        weights_low.view(bsz, 1, resolution, resolution),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)
    expanded = F.interpolate(
        expanded_low.view(bsz, 1, resolution, resolution).float(),
        size=(height, width),
        mode="nearest",
    ).squeeze(1).bool()
    return {
        "propagated_label": labels,
        "propagated_weight": weights,
        "expanded_reliable_mask": expanded,
    }


def correlation_propagation_loss(logits: torch.Tensor, propagated: dict):
    labels = propagated["propagated_label"].to(logits.device)
    weight = propagated["propagated_weight"].to(logits.device).float()
    mask = propagated["expanded_reliable_mask"].to(logits.device).bool()
    if mask.sum() == 0 or weight.sum() <= 0:
        return logits.new_tensor(0.0)
    ce = F.cross_entropy(logits, labels, reduction="none")
    weight = weight * mask.float()
    return (ce * weight).sum() / weight.sum().clamp_min(1e-6)
