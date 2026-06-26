from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_relation_loss(student_feature: torch.Tensor, teacher_edges: dict | None = None):
    if teacher_edges is None:
        return student_feature.new_tensor(0.0)
    edge_index = teacher_edges.get("edge_index")
    edge_weight = teacher_edges.get("edge_weight")
    if edge_index is None or edge_weight is None or edge_index.numel() == 0:
        return student_feature.new_tensor(0.0)
    b, c, h, w = student_feature.shape
    flat = F.normalize(student_feature.flatten(2).transpose(1, 2), dim=-1)
    edge_index = edge_index.to(student_feature.device)
    edge_weight = edge_weight.to(student_feature.device).float()
    src, dst = edge_index[0], edge_index[1]
    src = src.clamp_max(flat.shape[1] - 1)
    dst = dst.clamp_max(flat.shape[1] - 1)
    sim = (flat[:, src] * flat[:, dst]).sum(dim=-1).mean(dim=0)
    return F.mse_loss(sim, edge_weight)

