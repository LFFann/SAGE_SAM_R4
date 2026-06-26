from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r4.engine.trainer import SAGESAMR4Trainer
from r4.utils.env import collect_environment
from r4.utils.io import save_json, save_yaml, load_yaml, set_nested
from r4.utils.seed import seed_everything


def create_tiny_dataset(root: Path, num_classes: int = 3, image_size: int = 64):
    rng = np.random.default_rng(2026)
    for split, n, has_mask in [("labeled", 4, True), ("unlabeled", 4, False), ("val", 2, True), ("test", 2, True)]:
        (root / split / "image").mkdir(parents=True, exist_ok=True)
        if has_mask:
            (root / split / "mask").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            yy, xx = np.mgrid[:image_size, :image_size]
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            cx = int(image_size * (0.35 + 0.3 * rng.random()))
            cy = int(image_size * (0.35 + 0.3 * rng.random()))
            r = int(image_size * 0.15)
            mask[((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2] = 1
            mask[(np.abs(xx - image_size * 0.65) + np.abs(yy - image_size * 0.45)) < image_size * 0.18] = min(2, num_classes - 1)
            img[..., 0] = np.clip(mask * 90 + rng.normal(70, 20, mask.shape), 0, 255)
            img[..., 1] = np.clip(mask * 55 + rng.normal(80, 18, mask.shape), 0, 255)
            img[..., 2] = np.clip(rng.normal(85, 25, mask.shape), 0, 255)
            Image.fromarray(img).save(root / split / "image" / f"{split}_{i:03d}.png")
            if has_mask:
                Image.fromarray(mask).save(root / split / "mask" / f"{split}_{i:03d}.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--device")
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--sam-checkpoint")
    p.add_argument("--use-sam", action="store_true")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--num-classes", type=int)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    if args.device:
        set_nested(config, "train.device", args.device)
        set_nested(config, "sam.device", args.device)
    if args.data_root:
        set_nested(config, "data.root", args.data_root)
    if args.output_dir:
        set_nested(config, "experiment.output_dir", args.output_dir)
    if args.sam_checkpoint:
        set_nested(config, "sam.checkpoint", args.sam_checkpoint)
    if args.use_sam:
        set_nested(config, "sam.use_sam", True)
    if args.no_sam:
        set_nested(config, "sam.use_sam", False)
    if args.num_classes:
        set_nested(config, "data.num_classes", args.num_classes)
    if args.amp:
        set_nested(config, "train.amp", True)
    if args.no_amp:
        set_nested(config, "train.amp", False)
    if args.seed is not None:
        set_nested(config, "experiment.seed", args.seed)
    if args.max_iterations is not None:
        set_nested(config, "train.max_iterations", args.max_iterations)
    seed_everything(config["experiment"].get("seed", 2026), config["experiment"].get("deterministic", False))
    data_root = Path(config["data"]["root"])
    if config["data"].get("synthetic", False) and not data_root.exists():
        create_tiny_dataset(data_root, config["data"]["num_classes"], config["data"]["image_size"])
    out = Path(config["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(config, out / "resolved_config.yaml")
    save_json(vars(args), out / "args.json")
    save_json(collect_environment(), out / "environment.json")
    save_json({"source": "standalone SAGE_SAM_R4 generated from KnowSAM-style UNet and R3 engineering conventions"}, out / "source_versions.json")
    trainer = SAGESAMR4Trainer(config)
    if args.dry_run:
        result = trainer.dry_run()
        print(json.dumps(result, indent=2))
    else:
        trainer.train(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()

