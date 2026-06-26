from __future__ import annotations

import pytest
import torch

from r4.models.real_sam_wrapper import RealSAMWrapper
from r4.ssl.experimental_sparse_sam_relation_graph import build_topk_relation_graph


def test_missing_sam_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RealSAMWrapper("vit_b", tmp_path / "missing.pth", "cpu")


def test_dense_relation_graph_forbidden():
    emb = torch.randn(1, 4, 2, 2)
    with pytest.raises(ValueError):
        build_topk_relation_graph(emb, topk=4)
