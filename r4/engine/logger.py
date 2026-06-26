from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path


def setup_logger(output_dir, console: bool = False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"SAGE_SAM_R4:{output_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    logger.propagate = False
    return logger


def _json_safe(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(row), ensure_ascii=False, default=str, allow_nan=False) + "\n")


class OneLineProgress:
    def __init__(self, total: int, width: int = 28, stream=None):
        self.total = max(1, int(total))
        self.width = width
        self.stream = stream or sys.stdout
        self.last_len = 0

    def update(self, iteration: int, **metrics):
        ratio = min(1.0, max(0.0, iteration / self.total))
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        parts = [f"[{bar}]", f"{iteration}/{self.total}", f"{ratio * 100:5.1f}%"]
        for key in ("loss", "sup", "set", "val_dice", "val_iou", "val_hd95", "lr", "sam"):
            if key in metrics and metrics[key] is not None:
                value = metrics[key]
                if isinstance(value, float):
                    parts.append(f"{key}={value:.4g}")
                else:
                    parts.append(f"{key}={value}")
        line = " ".join(parts)
        pad = " " * max(0, self.last_len - len(line))
        self.stream.write("\r" + line + pad)
        self.stream.flush()
        self.last_len = len(line)

    def close(self):
        self.stream.write("\n")
        self.stream.flush()
