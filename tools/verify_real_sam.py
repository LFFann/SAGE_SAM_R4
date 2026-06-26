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
    args = p.parse_args()
    cfg = load_yaml(args.config)
    sam_cfg = cfg["sam"]
    wrapper = RealSAMWrapper(
        sam_cfg["model_type"],
        sam_cfg["checkpoint"],
        sam_cfg.get("device", "cpu"),
        sam_cfg.get("image_size", 1024),
        in_channels=cfg["data"].get("in_channels", 3),
        num_classes=cfg["data"].get("num_classes", 3),
    )
    assert wrapper.sam_is_real()
    x1 = torch.zeros(1, 3, 64, 64)
    x2 = torch.ones(1, 3, 64, 64)
    e1 = wrapper.image_embedding(x1).detach().cpu()
    e2 = wrapper.image_embedding(x2).detach().cpu()
    diff = float((e1 - e2).abs().mean())
    if diff <= 1e-8:
        raise RuntimeError("SAM embedding does not depend on input image; real SAM path is broken.")
    out = {
        "sam_real": True,
        "sam_source": wrapper.sam_source,
        "num_sam_params": wrapper.num_sam_params,
        "checkpoint_hash": wrapper.sam_checkpoint_hash,
        "embedding_diff": diff,
    }
    output = Path(cfg["experiment"]["output_dir"]) / "sam_real_verified.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
