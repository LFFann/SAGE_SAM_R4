from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing image directory: {path}")
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def resolve_dataset_root(
    root,
    dataset_name: str | None = None,
    labeled_subdir: str = "labeled",
    image_dir_name: str = "image",
) -> Path:
    """Accept either the dataset directory itself or its parent directory."""
    root = Path(root)
    direct = root / labeled_subdir / image_dir_name
    if direct.exists():
        return root
    if dataset_name:
        nested_root = root / dataset_name
        nested = nested_root / labeled_subdir / image_dir_name
        if nested.exists():
            return nested_root
        raise FileNotFoundError(
            "Missing dataset image directory. Checked both "
            f"{direct} and {nested}. Set data.root to the dataset directory "
            "or set data.root to the parent directory and data.dataset_name to the dataset folder name."
        )
    raise FileNotFoundError(
        f"Missing dataset image directory: {direct}. If data.root is a parent directory, set data.dataset_name."
    )


def mask_for_image(image_path: Path, image_dir: Path, mask_dir: Path) -> Path:
    rel = image_path.relative_to(image_dir)
    direct = mask_dir / rel
    if direct.exists():
        return direct
    for ext in IMAGE_EXTENSIONS:
        candidate = (mask_dir / rel).with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mask for {image_path}")


class SegmentationDataset2D(Dataset):
    def __init__(
        self,
        root,
        split: str,
        num_classes: int,
        image_size: int,
        image_dir_name: str = "image",
        mask_dir_name: str = "mask",
        has_mask: bool = True,
        ignore_index: int = 255,
    ):
        self.root = Path(root)
        self.split = split
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.ignore_index = ignore_index
        self.image_dir = self.root / split / image_dir_name
        self.mask_dir = self.root / split / mask_dir_name
        self.records = []
        for image_path in list_images(self.image_dir):
            self.records.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_for_image(image_path, self.image_dir, self.mask_dir) if has_mask else None,
                    "id": image_path.relative_to(self.image_dir).with_suffix("").as_posix(),
                }
            )
        if not self.records:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.records)

    def _read_array(self, path: Path):
        if path.suffix.lower() == ".npy":
            return np.load(path)
        return np.asarray(Image.open(path))

    def _load_image(self, path: Path):
        arr = self._read_array(path)
        orig_size = tuple(arr.shape[:2])
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[2] > 3:
            arr = arr[..., :3]
        image = Image.fromarray(arr.astype(np.uint8)) if arr.dtype != np.float32 else Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        out = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(out).permute(2, 0, 1).contiguous(), orig_size

    def _load_mask(self, path: Path):
        arr = self._read_array(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = Image.fromarray(arr.astype(np.uint8)).resize((self.image_size, self.image_size), Image.NEAREST)
        out = np.asarray(mask).astype(np.int64)
        valid = (out == self.ignore_index) | ((out >= 0) & (out < self.num_classes))
        if not np.all(valid):
            bad = sorted(np.unique(out[~valid]).tolist())
            raise ValueError(f"Mask {path} contains invalid class ids {bad}; expected 0..{self.num_classes - 1} or {self.ignore_index}")
        return torch.from_numpy(out).long()

    def __getitem__(self, index: int):
        rec = self.records[index]
        image, orig_size = self._load_image(rec["image_path"])
        item = {
            "image": image,
            "id": rec["id"],
            "image_path": str(rec["image_path"]),
            "mask_path": str(rec["mask_path"]) if rec["mask_path"] else "",
            "orig_size": orig_size,
        }
        if rec["mask_path"] is not None:
            item["mask"] = self._load_mask(rec["mask_path"])
        return item
