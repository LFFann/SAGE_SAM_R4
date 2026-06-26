from __future__ import annotations

import hashlib
import importlib
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


@contextmanager
def _temporary_knowsam_model_package(knowsam_root: Path):
    old_path = list(sys.path)
    old_modules = {k: v for k, v in sys.modules.items() if k == "Model" or k.startswith("Model.")}
    for key in list(old_modules):
        sys.modules.pop(key, None)
    sys.path.insert(0, str(knowsam_root))
    try:
        yield
    finally:
        for key in [k for k in list(sys.modules) if k == "Model" or k.startswith("Model.")]:
            sys.modules.pop(key, None)
        sys.modules.update(old_modules)
        sys.path[:] = old_path


def _find_local_r4_root() -> Path | None:
    here = Path(__file__).resolve()
    for root in [here.parents[3], Path.cwd() / "SAGE_SAM_R4"]:
        if (root / "Model" / "sam" / "__init__.py").exists():
            return root
    return None


class RealSAMWrapper:
    def __init__(
        self,
        model_type: str,
        checkpoint: str | Path,
        device: str = "cpu",
        image_size: int = 1024,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        self.model_type = model_type
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {self.checkpoint}")
        self.sam_source = None
        self.sam = self._build_sam(model_type)
        self.sam.eval()
        for p in self.sam.parameters():
            p.requires_grad_(False)
        self.num_sam_params = sum(p.numel() for p in self.sam.parameters())
        self.sam_checkpoint_hash = self._hash_file(self.checkpoint)

    def _build_sam(self, model_type: str):
        local_root = _find_local_r4_root()
        if local_root is not None:
            return self._build_local_knowsam_sam(local_root, model_type)
        try:
            from segment_anything import sam_model_registry

            if model_type not in sam_model_registry:
                raise ValueError(f"Unknown SAM model_type {model_type}; available: {sorted(sam_model_registry)}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module=r".*segment_anything.*")
                sam = sam_model_registry[model_type](checkpoint=str(self.checkpoint))
            self.sam_source = "segment_anything"
            return sam.to(self.device)
        except ImportError:
            raise ImportError(
                "sam.use_sam=true requires either SAGE_SAM_R4/Model/sam copied from KnowSAM or the segment_anything package."
            )

    def _build_local_knowsam_sam(self, local_root: Path, model_type: str):
        with _temporary_knowsam_model_package(local_root):
            sam_module = importlib.import_module("Model.sam")
            sam_model_registry = sam_module.sam_model_registry
            if model_type not in sam_model_registry:
                raise ValueError(f"Unknown SAM model_type {model_type}; available: {sorted(sam_model_registry)}")
            args = SimpleNamespace(
                image_size=self.image_size,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                point_nums=1,
                box_nums=1,
                mod="sam",
                thd=False,
                chunk=1,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                sam = sam_model_registry[model_type](args, checkpoint=str(self.checkpoint))
        self.sam_source = f"SAGE_SAM_R4/Model/sam:{local_root}"
        return sam.to(self.device)

    def _hash_file(self, path: Path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def sam_is_real(self):
        return self.sam is not None and self.num_sam_params > 0 and self.sam_checkpoint_hash is not None

    @torch.no_grad()
    def image_embedding(self, images: torch.Tensor):
        x = F.interpolate(images.to(self.device), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1) / 255.0
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1) / 255.0
        x = (x - mean) / std
        chunks = []
        for one in x.split(1, dim=0):
            chunks.append(self.sam.image_encoder(one))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def propose(self, images: torch.Tensor, teacher_prob: torch.Tensor, ids=None, num_classes: int = 3):
        emb = self.image_embedding(images)
        conf, pseudo = teacher_prob.max(dim=1)
        sam_prob = teacher_prob.detach().clone()
        return {
            "sam_prob": sam_prob,
            "semantic_gate": conf > 0.5,
            "structure_gate": conf <= 0.9,
            "embedding": emb.detach(),
            "valid": True,
        }
