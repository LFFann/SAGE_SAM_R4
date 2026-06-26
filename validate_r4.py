from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r4.data.dataset_2d import SegmentationDataset2D, resolve_dataset_root
from r4.engine.checkpoint import load_student_checkpoint
from r4.engine.evaluator import evaluate
from r4.engine.logger import append_jsonl
from r4.engine.model_factory import build_deploy_model
from r4.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save-pred", action="store_true")
    p.add_argument("--split", default="val")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    dev = cfg["train"].get("device", "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    device = torch.device(dev)
    model = build_deploy_model(cfg).to(device)
    load_student_checkpoint(model, args.checkpoint, strict=False, map_location=device)
    data_root = resolve_dataset_root(
        cfg["data"].get("resolved_root", cfg["data"]["root"]),
        cfg["data"].get("dataset_name"),
        cfg["data"].get("labeled_subdir", "labeled"),
        cfg["data"].get("image_dir_name", "image"),
    )
    ds = SegmentationDataset2D(data_root, args.split, cfg["data"]["num_classes"], cfg["data"]["image_size"], cfg["data"].get("image_dir_name", "image"), cfg["data"].get("mask_dir_name", "mask"), has_mask=True, ignore_index=cfg["data"].get("ignore_index", 255))
    loader = DataLoader(ds, batch_size=cfg.get("eval", {}).get("batch_size", 1), shuffle=False)
    save_dir = Path(cfg["experiment"]["output_dir"]) / "predictions" / args.split if args.save_pred else None
    metrics = evaluate(model, loader, cfg["data"]["num_classes"], device, cfg.get("eval", {}).get("compute_hd95", True), save_dir, cfg["data"].get("ignore_index", 255))
    append_jsonl(Path(cfg["experiment"]["output_dir"]) / "metrics.jsonl", {"iteration": "external", "phase": args.split, **metrics})
    print(metrics)


if __name__ == "__main__":
    main()
