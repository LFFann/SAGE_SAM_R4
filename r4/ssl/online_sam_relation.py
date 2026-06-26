from __future__ import annotations

import torch
import torch.nn.functional as F


def online_sam_student_relation_loss(
    student_feature: torch.Tensor,
    sam_embedding: torch.Tensor | None,
    gate: torch.Tensor | None = None,
    topk: int = 8,
    resolution: int = 16,
):
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
    losses = []
    if gate is not None:
        gate_small = F.interpolate(gate.float().unsqueeze(1), size=(resolution, resolution), mode="nearest").flatten(1).bool()
    else:
        gate_small = torch.ones(stu.shape[:2], device=stu.device, dtype=torch.bool)
    for bi in range(stu.shape[0]):
        valid = gate_small[bi]
        if valid.sum() <= topk + 1:
            continue
        sam_sim = sam[bi] @ sam[bi].transpose(0, 1)
        stu_sim = stu[bi] @ stu[bi].transpose(0, 1)
        vals, idx = sam_sim.topk(k=topk + 1, dim=1)
        idx = idx[:, 1:]
        vals = vals[:, 1:]
        row = torch.arange(n, device=stu.device).view(-1, 1).expand_as(idx)
        edge_mask = valid[row] & valid[idx]
        if edge_mask.sum() == 0:
            continue
        pred = stu_sim[row[edge_mask], idx[edge_mask]]
        target = vals[edge_mask]
        losses.append(F.mse_loss(pred, target))
    if not losses:
        return student_feature.new_tensor(0.0)
    return torch.stack(losses).mean()
