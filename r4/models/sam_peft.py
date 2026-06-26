from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class SAMTrainabilityReport:
    total_sam_params: int
    trainable_sam_params: int
    trainable_ratio: float
    trainable_module_names: list[str]


class SAMPEFTAdapter:
    """Freezes SAM by default and exposes only PEFT/decoder parameters."""

    def __init__(
        self,
        sam: nn.Module,
        train_peft: bool = True,
        peft_type: str = "adapter",
        train_mask_decoder: bool = True,
        train_prompt_encoder: bool = False,
        train_last_n_blocks: int = 0,
        max_trainable_ratio: float = 0.05,
        hard_max_trainable_ratio: float = 0.10,
    ):
        self.sam = sam
        self.train_peft = bool(train_peft)
        self.peft_type = str(peft_type)
        self.train_mask_decoder = bool(train_mask_decoder)
        self.train_prompt_encoder = bool(train_prompt_encoder)
        self.train_last_n_blocks = int(train_last_n_blocks)
        self.max_trainable_ratio = float(max_trainable_ratio)
        self.hard_max_trainable_ratio = float(hard_max_trainable_ratio)
        self.report = self.configure()

    def configure(self) -> SAMTrainabilityReport:
        for param in self.sam.parameters():
            param.requires_grad_(False)

        if self.train_peft:
            self._enable_adapter_parameters(self.train_last_n_blocks)
        if self.train_mask_decoder and hasattr(self.sam, "mask_decoder"):
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad_(True)
        if self.train_prompt_encoder and hasattr(self.sam, "prompt_encoder"):
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad_(True)

        report = self._make_report()
        if self.train_peft and report.trainable_sam_params == 0:
            raise RuntimeError("sam.train_peft=true but SAM has zero trainable parameters")
        if report.trainable_ratio > self.hard_max_trainable_ratio:
            raise RuntimeError(
                f"SAM trainable ratio {report.trainable_ratio:.4f} exceeds hard limit "
                f"{self.hard_max_trainable_ratio:.4f}"
            )
        if report.trainable_ratio > self.max_trainable_ratio:
            warnings.warn(
                f"SAM trainable ratio {report.trainable_ratio:.4f} exceeds recommended "
                f"limit {self.max_trainable_ratio:.4f}",
                RuntimeWarning,
            )
        return report

    def _enable_adapter_parameters(self, last_n_blocks: int = 0):
        image_encoder = getattr(self.sam, "image_encoder", None)
        blocks = getattr(image_encoder, "blocks", None)
        adapter_tokens = ("adapter", "lora")
        if blocks is not None and last_n_blocks > 0:
            for block in list(blocks)[-last_n_blocks:]:
                for name, param in block.named_parameters():
                    lowered = name.lower()
                    if any(token in lowered for token in adapter_tokens):
                        param.requires_grad_(True)
            return
        for name, param in self.sam.named_parameters():
            lowered = name.lower()
            if any(token in lowered for token in adapter_tokens):
                param.requires_grad_(True)

    def _enable_last_blocks(self, n_blocks: int):
        image_encoder = getattr(self.sam, "image_encoder", None)
        blocks = getattr(image_encoder, "blocks", None)
        if blocks is None:
            return
        for block in list(blocks)[-n_blocks:]:
            for name, param in block.named_parameters():
                lowered = name.lower()
                if "adapter" in lowered or "lora" in lowered:
                    param.requires_grad_(True)

    def _make_report(self) -> SAMTrainabilityReport:
        total = sum(p.numel() for p in self.sam.parameters())
        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        modules = []
        for name, param in self.sam.named_parameters():
            if param.requires_grad:
                modules.append(name.rsplit(".", 1)[0])
        module_names = sorted(set(modules))
        return SAMTrainabilityReport(
            total_sam_params=int(total),
            trainable_sam_params=int(trainable),
            trainable_ratio=float(trainable / max(1, total)),
            trainable_module_names=module_names,
        )

    def trainable_parameters(self):
        return [p for p in self.sam.parameters() if p.requires_grad]

    def parameter_groups(self, lr_peft: float, lr_mask_decoder: float | None = None, lr_prompt_encoder: float | None = None):
        lr_mask_decoder = lr_peft if lr_mask_decoder is None else lr_mask_decoder
        lr_prompt_encoder = lr_peft if lr_prompt_encoder is None else lr_prompt_encoder
        peft_params = []
        mask_params = []
        prompt_params = []
        for name, param in self.sam.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("mask_decoder."):
                mask_params.append(param)
            elif name.startswith("prompt_encoder."):
                prompt_params.append(param)
            else:
                peft_params.append(param)
        groups = []
        if peft_params:
            groups.append({"params": peft_params, "lr": lr_peft, "name": "sam_peft"})
        if mask_params:
            groups.append({"params": mask_params, "lr": lr_mask_decoder, "name": "sam_mask_decoder"})
        if prompt_params:
            groups.append({"params": prompt_params, "lr": lr_prompt_encoder, "name": "sam_prompt_encoder"})
        return groups
