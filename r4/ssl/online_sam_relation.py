from __future__ import annotations

import torch
import torch.nn.functional as F


def online_sam_student_relation_loss(
    student_feature: torch.Tensor,
    sam_embedding: torch.Tensor | None,
    gate: torch.Tensor | None = None,
    boundary: torch.Tensor | None = None,
    topk: int = 8,
    resolution: int = 16,
    temperature: float = 0.2,
    rank_weight: float = 0.25,
    rank_margin: float = 0.02,
):
    """Top-k KL and rank consistency between SAM and student relations."""

    if sam_embedding is None:
        return student_feature.new_tensor(0.0)
    if student_feature.ndim != 4 or sam_embedding.ndim != 4:
        raise ValueError("student_feature and sam_embedding must be BCHW")
    resolution = int(resolution)
    topk = int(topk)
    if resolution <= 1 or topk <= 0:
        return student_feature.new_tensor(0.0)

    stu = F.interpolate(student_feature, size=(resolution, resolution), mode="bilinear", align_corners=False)
    sam = F.interpolate(sam_embedding, size=(resolution, resolution), mode="bilinear", align_corners=False)
    stu = F.normalize(stu.flatten(2).transpose(1, 2), dim=-1)
    sam = F.normalize(sam.flatten(2).transpose(1, 2), dim=-1)
    n = stu.shape[1]
    if topk >= n:
        raise ValueError("online relation topk would become dense; reduce structure.online_topk")

    valid = _downsample_valid_mask(gate, boundary, stu.shape[0], resolution, stu.device)
    losses = []
    for bi in range(stu.shape[0]):
        valid_b = valid[bi]
        if valid_b.sum() <= topk + 1:
            continue
        sam_sim = sam[bi] @ sam[bi].transpose(0, 1)
        stu_sim = stu[bi] @ stu[bi].transpose(0, 1)
        vals, idx = sam_sim.topk(k=topk + 1, dim=1)
        vals = vals[:, 1:]
        idx = idx[:, 1:]
        row = torch.arange(n, device=stu.device).view(-1, 1).expand_as(idx)
        edge_mask = valid_b[row] & valid_b[idx]
        row_mask = edge_mask.any(dim=1)
        if row_mask.sum() == 0:
            continue

        vals_sel = vals[row_mask].masked_fill(~edge_mask[row_mask], -1.0e4)
        pred_sel = stu_sim[row[row_mask], idx[row_mask]].masked_fill(~edge_mask[row_mask], -1.0e4)
        target_dist = torch.softmax(vals_sel.detach() / max(temperature, 1e-6), dim=-1)
        pred_log = torch.log_softmax(pred_sel / max(temperature, 1e-6), dim=-1)
        kl = F.kl_div(pred_log, target_dist, reduction="batchmean")
        rank = _rank_consistency_loss(pred_sel, vals_sel.detach(), edge_mask[row_mask], rank_margin)
        losses.append(kl + float(rank_weight) * rank)
    if not losses:
        return student_feature.new_tensor(0.0)
    return torch.stack(losses).mean()


def _downsample_valid_mask(gate, boundary, batch_size: int, resolution: int, device):
    if gate is None:
        valid = torch.ones(batch_size, resolution * resolution, device=device, dtype=torch.bool)
    else:
        valid = F.interpolate(gate.float().unsqueeze(1), size=(resolution, resolution), mode="nearest").flatten(1).bool()
    if boundary is not None:
        bnd = F.interpolate(boundary.float(), size=(resolution, resolution), mode="nearest").flatten(1)
        valid = valid & (bnd < 0.5)
    return valid


def _rank_consistency_loss(student_vals: torch.Tensor, sam_vals: torch.Tensor, edge_mask: torch.Tensor, margin: float):
    if student_vals.shape[-1] <= 1:
        return student_vals.new_tensor(0.0)
    stu_diff = student_vals.unsqueeze(-1) - student_vals.unsqueeze(-2)
    sam_diff = sam_vals.unsqueeze(-1) - sam_vals.unsqueeze(-2)
    pair_mask = edge_mask.unsqueeze(-1) & edge_mask.unsqueeze(-2)
    pair_mask = pair_mask & (sam_diff.abs() > 1e-6)
    if pair_mask.sum() == 0:
        return student_vals.new_tensor(0.0)
    direction = torch.sign(sam_diff)
    loss = F.relu(float(margin) - direction * stu_diff)
    return loss[pair_mask].mean()
