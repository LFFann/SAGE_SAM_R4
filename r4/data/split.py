from __future__ import annotations

import random


def create_train_calibration_split(labeled_files, ratio: float, min_images: int, seed: int):
    n = len(labeled_files)
    if n == 0:
        raise ValueError("No labeled files available for calibration split")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if n <= max(1, min_images * 2):
        return indices, indices, True
    cal_n = max(min_images, int(round(n * ratio)))
    cal_n = min(cal_n, n - 1)
    calibration = sorted(indices[:cal_n])
    train = sorted(indices[cal_n:])
    return train, calibration, False

