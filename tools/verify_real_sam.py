from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r4.models.real_sam_wrapper import RealSAMWrapper
from r4.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--prompt-eps", type=float, default=1e-8)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    sam_cfg = cfg["sam"]
    data_cfg = cfg["data"]
    prompt_cfg = sam_cfg.get("prompt", {})
    wrapper = RealSAMWrapper(
        sam_cfg["model_type"],
        sam_cfg["checkpoint"],
        sam_cfg.get("device", "cpu"),
        sam_cfg.get("image_size", 1024),
        in_channels=data_cfg.get("in_channels", 3),
        num_classes=data_cfg.get("num_classes", 3),
        use_mask_prompt=prompt_cfg.get("use_mask_prompt", True),
        use_box_prompt=prompt_cfg.get("use_box_prompt", True),
        use_point_prompt=prompt_cfg.get("use_point_prompt", True),
        use_negative_points=prompt_cfg.get("use_negative_points", True),
    )
    assert wrapper.sam_is_real()
    x1 = torch.zeros(1, 3, 64, 64)
    x2 = torch.ones(1, 3, 64, 64)
    with torch.no_grad():
        e1 = wrapper.image_embedding(x1).detach().cpu()
        e2 = wrapper.image_embedding(x2).detach().cpu()
        diff = float((e1 - e2).abs().mean())
        if diff <= 1e-8:
            raise RuntimeError("SAM embedding does not depend on input image; real SAM path is broken.")
        prompt_a = _make_prompt(data_cfg.get("num_classes", 3), prompt_cfg.get("mask_prompt_size", 256), variant=0)
        prompt_b = _make_prompt(data_cfg.get("num_classes", 3), prompt_cfg.get("mask_prompt_size", 256), variant=1)
        mask_a = wrapper.forward_prompted(x1, prompt_a)["sam_masks"].detach().cpu()
        mask_b = wrapper.forward_prompted(x1, prompt_b)["sam_masks"].detach().cpu()
        prompt_diff = float((mask_a - mask_b).abs().mean())
        if prompt_diff <= args.prompt_eps:
            raise RuntimeError("SAM output does not change when prompts change; prompt path is broken.")
    out = {
        "sam_real": True,
        "sam_source": wrapper.sam_source,
        "num_sam_params": wrapper.num_sam_params,
        "checkpoint_hash": wrapper.sam_checkpoint_hash,
        "embedding_diff": diff,
        "prompt_output_diff": prompt_diff,
    }
    output = Path(cfg["experiment"]["output_dir"]) / "sam_real_verified.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


def _make_prompt(num_classes: int, mask_prompt_size: int, variant: int):
    fg = max(0, int(num_classes) - 1)
    if fg == 0:
        raise ValueError("verify_real_sam requires at least one foreground class")
    mask_prompt = torch.zeros(fg, 1, mask_prompt_size, mask_prompt_size)
    if variant == 0:
        mask_prompt[:, :, : mask_prompt_size // 2, : mask_prompt_size // 2] = 1.0
        boxes = torch.tensor([[0.05, 0.05, 0.45, 0.45]] * fg, dtype=torch.float32)
        points = torch.tensor([[[0.25, 0.25]]] * fg, dtype=torch.float32)
        negatives = torch.tensor([[[0.80, 0.80]]] * fg, dtype=torch.float32)
    else:
        mask_prompt[:, :, mask_prompt_size // 2 :, mask_prompt_size // 2 :] = 1.0
        boxes = torch.tensor([[0.55, 0.55, 0.95, 0.95]] * fg, dtype=torch.float32)
        points = torch.tensor([[[0.75, 0.75]]] * fg, dtype=torch.float32)
        negatives = torch.tensor([[[0.20, 0.20]]] * fg, dtype=torch.float32)
    return {
        "mask_prompt": mask_prompt,
        "boxes_xyxy": boxes,
        "point_coords": points,
        "point_labels": torch.ones(fg, 1, dtype=torch.long),
        "negative_point_coords": negatives,
        "image_index": torch.zeros(fg, dtype=torch.long),
        "class_ids": torch.arange(1, int(num_classes), dtype=torch.long),
        "prompt_quality": torch.ones(1, int(num_classes)),
    }


if __name__ == "__main__":
    main()
