from __future__ import annotations

import torch
import torch.nn as nn

from r4.models.real_sam_wrapper import RealSAMWrapper
from r4.models.sam_peft import LoRALinear, SAMPEFTAdapter


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24)
        self.attn.proj = nn.Linear(8, 8)


class TinySAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.mask_decoder = nn.Linear(8, 8)


def test_lora_injection_targets_last_blocks_only():
    sam = TinySAM()
    adapter = SAMPEFTAdapter(
        sam,
        train_peft=True,
        peft_type="lora",
        train_mask_decoder=False,
        train_last_n_blocks=1,
        lora_rank=2,
        lora_alpha=4,
        max_trainable_ratio=1.0,
        hard_max_trainable_ratio=1.0,
    )
    assert not isinstance(sam.image_encoder.blocks[0].attn.qkv, LoRALinear)
    assert isinstance(sam.image_encoder.blocks[1].attn.qkv, LoRALinear)
    assert isinstance(sam.image_encoder.blocks[1].attn.proj, LoRALinear)
    assert adapter.report.lora_param_count > 0


class FakePELayer(nn.Module):
    def forward_with_coords(self, coords, image_size):
        out = torch.zeros(*coords.shape[:-1], 256, device=coords.device)
        out[..., :2] = coords
        return out


class FakePromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_image_size = (1024, 1024)
        self.pe_layer = FakePELayer()
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, 256) for _ in range(4)])


def test_local_prompt_preparation_uses_points_negative_points_and_boxes():
    wrapper = RealSAMWrapper.__new__(RealSAMWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.image_size = 1024
    wrapper.sam_source = "SAGE_SAM_R4/Model/sam:test"
    wrapper.use_point_prompt = True
    wrapper.use_negative_points = True
    wrapper.use_box_prompt = True
    wrapper.sam = nn.Module()
    wrapper.sam.prompt_encoder = FakePromptEncoder()
    prompts = {
        "point_coords": torch.tensor([[[0.25, 0.50]]]),
        "point_labels": torch.ones(1, 1, dtype=torch.long),
        "negative_point_coords": torch.tensor([[[0.75, 0.10]]]),
        "boxes_xyxy": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }
    points, boxes = wrapper._prepare_sparse_prompts(prompts)
    assert points[0].shape == (1, 2, 256)
    assert points[1].tolist() == [[1, 0]]
    assert boxes.shape == (1, 2, 256)
