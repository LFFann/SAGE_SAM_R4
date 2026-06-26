from __future__ import annotations

import torch
import torch.nn.functional as F


def build_topk_relation_graph(embedding: torch.Tensor, topk: int = 8):
    """Experimental offline relation graph helper kept out of the R4 main loop."""
    if embedding.ndim != 4:
        raise ValueError("embedding must be BCHW")
    feat = F.normalize(embedding.flatten(2).mean(0).transpose(0, 1), dim=-1)
    n = feat.shape[0]
    if topk >= n:
        raise ValueError("topk relation graph would become dense; reduce topk")
    sim = feat @ feat.t()
    vals, idx = sim.topk(k=min(topk + 1, n), dim=1)
    idx = idx[:, 1:]
    vals = vals[:, 1:]
    src = torch.arange(n, device=embedding.device).view(-1, 1).expand_as(idx).reshape(-1)
    return {"edge_index": torch.stack([src.cpu(), idx.reshape(-1).cpu()]), "edge_weight": vals.reshape(-1).cpu()}
