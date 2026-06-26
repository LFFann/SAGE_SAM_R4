from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r4.data.dataset_2d import SegmentationDataset2D, resolve_dataset_root
from r4.models.real_sam_wrapper import RealSAMWrapper
from r4.ssl.experimental_sparse_sam_relation_graph import build_topk_relation_graph
from r4.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="unlabeled")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output-dir", default=None)
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
    split = cfg["data"].get(f"{args.split}_subdir", args.split)
    has_mask = args.split != "unlabeled"
    data_root = resolve_dataset_root(
        cfg["data"].get("resolved_root", cfg["data"]["root"]),
        cfg["data"].get("dataset_name"),
        cfg["data"].get("labeled_subdir", "labeled"),
        cfg["data"].get("image_dir_name", "image"),
    )
    ds = SegmentationDataset2D(data_root, split, cfg["data"]["num_classes"], cfg["data"]["image_size"], cfg["data"].get("image_dir_name", "image"), cfg["data"].get("mask_dir_name", "mask"), has_mask=has_mask, ignore_index=cfg["data"].get("ignore_index", 255))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    out_dir = Path(args.output_dir or sam_cfg.get("cache_dir", "./cache/SAGE_SAM_R4/experimental_structure_cache")) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        sample_id = batch["id"][0].replace("/", "_").replace("\\", "_")
        path = out_dir / f"{sample_id}.pt"
        if path.exists() and not args.overwrite:
            continue
        emb = wrapper.image_embedding(batch["image"]).detach()
        emb_small = F.interpolate(emb.float(), size=(16, 16), mode="bilinear", align_corners=False).cpu()
        graph = build_topk_relation_graph(emb_small, sam_cfg.get("topk_edges", 8))
        torch.save({"sam_embedding": emb_small.half(), **graph, "boundary": None, "meta": {"id": sample_id}}, path)
    print(f"cache written to {out_dir}")


if __name__ == "__main__":
    main()
