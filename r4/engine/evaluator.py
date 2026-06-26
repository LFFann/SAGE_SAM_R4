from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from r4.utils.hd95 import per_class_hd95
from r4.utils.metrics import average_foreground, per_class_dice_iou
from r4.utils.visualization import save_mask_png


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, num_classes: int, device, compute_hd95: bool = True, save_dir=None, ignore_index: int = 255):
    model.eval()
    device = torch.device(device)
    all_dice, all_iou, all_hd95 = [], [], []
    rows = []
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for batch in dataloader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        logits = model(image)
        pred = logits.argmax(dim=1)
        for i in range(pred.shape[0]):
            dice, iou = per_class_dice_iou(pred[i], mask[i], num_classes, ignore_index)
            hd = per_class_hd95(pred[i].cpu().numpy(), mask[i].cpu().numpy(), num_classes, ignore_index) if compute_hd95 else [float("nan")] * num_classes
            all_dice.append(dice)
            all_iou.append(iou)
            all_hd95.append(hd)
            sample_id = batch.get("id", [f"sample_{len(rows)}"])[i]
            rows.append({"id": sample_id, "avg_dice": average_foreground(dice), "avg_iou": average_foreground(iou), "avg_hd95": average_foreground(hd)})
            if save_dir:
                safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
                save_mask_png(pred[i].cpu().numpy(), save_dir / f"{safe_id}.png")
    class_dice = np.nanmean(np.asarray(all_dice, dtype=float), axis=0).tolist()
    class_iou = np.nanmean(np.asarray(all_iou, dtype=float), axis=0).tolist()
    if compute_hd95:
        hd_arr = np.asarray(all_hd95, dtype=float)
        class_hd95 = []
        for c in range(num_classes):
            finite = hd_arr[:, c][np.isfinite(hd_arr[:, c])]
            class_hd95.append(float(finite.mean()) if finite.size else float("nan"))
    else:
        class_hd95 = [float("nan")] * num_classes
    metrics = {
        "class_dice": class_dice,
        "class_iou": class_iou,
        "class_hd95": class_hd95,
        "avg_dice": average_foreground(class_dice),
        "avg_iou": average_foreground(class_iou),
        "avg_hd95": average_foreground(class_hd95),
    }
    if save_dir:
        with (save_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "avg_dice", "avg_iou", "avg_hd95"])
            writer.writeheader()
            writer.writerows(rows)
    return metrics
