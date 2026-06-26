from __future__ import annotations

import json
import platform
from pathlib import Path

import torch


def collect_environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
    }


def write_environment(path: str | Path):
    Path(path).write_text(json.dumps(collect_environment(), indent=2), encoding="utf-8")

