from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_mask_png(mask, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask).astype(np.uint8)).save(path)

