from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r4.data.dataset_2d import SegmentationDataset2D, list_images, resolve_dataset_root
from r4.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_dataset_root(
        cfg["data"].get("resolved_root", cfg["data"]["root"]),
        cfg["data"].get("dataset_name"),
        cfg["data"].get("labeled_subdir", "labeled"),
        cfg["data"].get("image_dir_name", "image"),
    )
    num_classes = cfg["data"]["num_classes"]
    image_name = cfg["data"].get("image_dir_name", "image")
    mask_name = cfg["data"].get("mask_dir_name", "mask")
    counts = {}
    pixels = Counter()
    for split, has_mask in [("labeled", True), ("unlabeled", False), ("val", True), ("test", True)]:
        split = cfg["data"].get(f"{split}_subdir", split)
        image_dir = root / split / image_name
        images = list_images(image_dir)
        counts[split] = len(images)
        if has_mask:
            for item in SegmentationDataset2D(root, split, num_classes, cfg["data"]["image_size"], image_name, mask_name, True, cfg["data"].get("ignore_index", 255)):
                vals, cnts = np.unique(item["mask"].numpy(), return_counts=True)
                for v, c in zip(vals, cnts):
                    pixels[int(v)] += int(c)
    print({"root": str(root), "counts": counts, "class_pixels": dict(sorted(pixels.items()))})


if __name__ == "__main__":
    main()
