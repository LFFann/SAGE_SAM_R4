from __future__ import annotations

import hashlib
import importlib
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_peft import SAMPEFTAdapter


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


class RealSAMWrapper(nn.Module):
    def __init__(
        self,
        model_type: str,
        checkpoint: str | Path,
        device: str = "cpu",
        image_size: int = 1024,
        in_channels: int = 3,
        num_classes: int = 3,
        train_peft: bool = False,
        peft_type: str = "adapter",
        train_mask_decoder: bool = False,
        train_prompt_encoder: bool = False,
        train_last_n_blocks: int = 0,
        max_trainable_ratio: float = 0.05,
    ):
        super().__init__()
        self.model_type = model_type
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.train_peft = bool(train_peft)
        self.peft_type = str(peft_type)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {self.checkpoint}")
        self.sam_source = None
        self.sam = self._build_sam(model_type)
        self.num_sam_params = sum(p.numel() for p in self.sam.parameters())
        self.sam_checkpoint_hash = self._hash_file(self.checkpoint)
        self.peft_adapter = SAMPEFTAdapter(
            self.sam,
            train_peft=self.train_peft,
            peft_type=self.peft_type,
            train_mask_decoder=train_mask_decoder,
            train_prompt_encoder=train_prompt_encoder,
            train_last_n_blocks=train_last_n_blocks,
            max_trainable_ratio=max_trainable_ratio,
        )
        if self.peft_adapter.report.trainable_sam_params > 0:
            self.sam.train()
        else:
            self.sam.eval()

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
        except ImportError as exc:
            raise ImportError(
                "sam.use_sam=true requires either SAGE_SAM_R4/Model/sam copied from KnowSAM or the segment_anything package."
            ) from exc

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
                mod="sam_adpt" if self.train_peft else "sam",
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

    def trainability_report(self):
        return self.peft_adapter.report

    def parameter_groups(self, lr_peft: float, lr_mask_decoder: float | None = None, lr_prompt_encoder: float | None = None):
        return self.peft_adapter.parameter_groups(lr_peft, lr_mask_decoder, lr_prompt_encoder)

    def image_embedding(self, images: torch.Tensor):
        x = F.interpolate(images.to(self.device), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1) / 255.0
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1) / 255.0
        x = (x - mean) / std
        chunks = []
        for one in x.split(1, dim=0):
            chunks.append(self.sam.image_encoder(one))
        return torch.cat(chunks, dim=0)

    def forward_prompted(self, images: torch.Tensor, prompts: dict[str, torch.Tensor], multimask_output: bool = False):
        if not self.sam_is_real():
            raise RuntimeError("SAM is not available")
        b, _, h, w = images.shape
        image_embeddings = self.image_embedding(images)
        mask_prompt = prompts["mask_prompt"].to(self.device).float()
        prompt_size = getattr(self.sam.prompt_encoder, "mask_input_size", mask_prompt.shape[-2:])
        if tuple(mask_prompt.shape[-2:]) != tuple(prompt_size):
            mask_prompt = F.interpolate(mask_prompt, size=prompt_size, mode="bilinear", align_corners=False)
        image_index = prompts["image_index"].to(self.device).long()
        class_ids = prompts["class_ids"].to(self.device).long()
        prompt_embeddings = image_embeddings.index_select(0, image_index)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=None, masks=mask_prompt)
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=prompt_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        fg_logits_flat = F.interpolate(low_res_masks[:, :1], size=(h, w), mode="bilinear", align_corners=False)
        fg = max(0, self.num_classes - 1)
        expected_index = torch.arange(b, device=self.device).repeat_interleave(fg)
        expected_class = torch.arange(1, self.num_classes, device=self.device).repeat(b)
        if fg_logits_flat.shape[0] == b * fg and torch.equal(image_index, expected_index) and torch.equal(class_ids, expected_class):
            fg_logits = fg_logits_flat[:, 0].reshape(b, fg, h, w)
            fg_iou = torch.sigmoid(iou_predictions[:, 0]).reshape(b, fg)
        else:
            rows = []
            iou_rows = []
            for bi in range(b):
                class_logits = []
                class_iou = []
                for ci in range(1, self.num_classes):
                    match = (image_index == bi) & (class_ids == ci)
                    if match.any():
                        idx = torch.where(match)[0][0]
                        class_logits.append(fg_logits_flat[idx, 0])
                        class_iou.append(torch.sigmoid(iou_predictions[idx, 0]))
                    else:
                        class_logits.append(images.new_zeros((h, w)))
                        class_iou.append(images.new_tensor(0.0))
                rows.append(torch.stack(class_logits, dim=0))
                iou_rows.append(torch.stack(class_iou, dim=0))
            fg_logits = torch.stack(rows, dim=0)
            fg_iou = torch.stack(iou_rows, dim=0)

        fg_prob = torch.sigmoid(fg_logits)
        bg_prob = (1.0 - fg_prob.max(dim=1, keepdim=True).values).clamp(1e-5, 1.0)
        sam_prob = torch.cat([bg_prob, fg_prob], dim=1)
        sam_prob = sam_prob / sam_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        sam_logits = torch.log(sam_prob.clamp_min(1e-6))
        sam_iou = images.new_ones((b, self.num_classes))
        if self.num_classes > 1:
            sam_iou[:, 1:] = fg_iou
            sam_iou[:, 0] = 1.0 - fg_iou.mean(dim=1)
        prompt_quality = prompts.get("prompt_quality")
        if prompt_quality is None:
            prompt_quality = images.new_ones((b, self.num_classes))
        else:
            prompt_quality = prompt_quality.to(self.device)
        sam_boundary = _boundary_from_prob(sam_prob)
        semantic_gate = sam_prob.max(dim=1).values > 0.5
        return {
            "sam_logits": sam_logits,
            "sam_prob": sam_prob,
            "sam_masks": fg_prob,
            "sam_iou": sam_iou,
            "sam_embedding": image_embeddings,
            "sam_boundary": sam_boundary,
            "prompt_quality": prompt_quality,
            "semantic_gate": semantic_gate,
            "structure_gate": semantic_gate,
            "valid": True,
        }

    def propose(self, images: torch.Tensor, teacher_prob: torch.Tensor, ids=None, num_classes: int = 3):
        prompts = self._teacher_prompts(teacher_prob.detach(), num_classes=num_classes)
        return self.forward_prompted(images, prompts)

    def _teacher_prompts(self, teacher_prob: torch.Tensor, num_classes: int):
        b, c, h, w = teacher_prob.shape
        fg = max(0, num_classes - 1)
        mask_prompt = F.interpolate(teacher_prob[:, 1:num_classes], size=(256, 256), mode="bilinear", align_corners=False)
        image_index = torch.arange(b, device=teacher_prob.device).repeat_interleave(fg)
        class_ids = torch.arange(1, num_classes, device=teacher_prob.device).repeat(b)
        return {
            "mask_prompt": mask_prompt.reshape(b * fg, 1, 256, 256),
            "image_index": image_index,
            "class_ids": class_ids,
            "prompt_quality": teacher_prob.new_ones((b, c)),
        }


def _boundary_from_prob(prob: torch.Tensor):
    label = prob.argmax(dim=1, keepdim=True).float()
    gx = F.pad((label[:, :, :, 1:] != label[:, :, :, :-1]).float(), (0, 1, 0, 0))
    gy = F.pad((label[:, :, 1:, :] != label[:, :, :-1, :]).float(), (0, 0, 0, 1))
    return torch.clamp(gx + gy, 0.0, 1.0)
