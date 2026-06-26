from __future__ import annotations

import numpy as np


def hd95_binary(pred, target):
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except Exception:
        return float("nan")
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")
    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)
    dt_pred = distance_transform_edt(~pred_border)
    dt_target = distance_transform_edt(~target_border)
    distances = np.concatenate([dt_target[pred_border], dt_pred[target_border]])
    return float(np.percentile(distances, 95))


def per_class_hd95(pred, target, num_classes: int, ignore_index: int = 255):
    pred = np.asarray(pred)
    target = np.asarray(target)
    valid = target != ignore_index
    out = []
    for c in range(num_classes):
        out.append(hd95_binary((pred == c) & valid, (target == c) & valid))
    return out

