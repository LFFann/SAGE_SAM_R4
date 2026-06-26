from __future__ import annotations

import math

import torch
import torch.nn as nn

from .prompt_generator import PromptGenerator
from .real_sam_wrapper import RealSAMWrapper


class PromptableSAMMentor(nn.Module):
    """Online trainable SAM co-learner with a trainable prompt generator."""

    def __init__(self, wrapper: RealSAMWrapper | None, num_classes: int = 3, config: dict | None = None, in_channels: int = 3):
        super().__init__()
        self.wrapper = wrapper
        self.num_classes = int(num_classes)
        cfg = config or {}
        prompt_cfg = cfg.get("prompt", {})
        self.prompt_generator = PromptGenerator(
            num_classes=self.num_classes,
            in_channels=in_channels,
            mask_prompt_size=prompt_cfg.get("mask_prompt_size", 256),
            min_component_area=prompt_cfg.get("min_component_area", cfg.get("min_component_area", 16)),
            residual_scale=prompt_cfg.get("residual_scale", 0.15),
        )
        self.train_prompt_generator = bool(cfg.get("train_prompt_generator", True))
        if not self.train_prompt_generator:
            for param in self.prompt_generator.parameters():
                param.requires_grad_(False)

    def available(self):
        return self.wrapper is not None and self.wrapper.sam_is_real()

    def forward_labeled(self, image: torch.Tensor, gt_mask: torch.Tensor):
        if not self.available():
            return {"valid": False}
        prompts = self.prompt_generator(image=image, gt_mask=gt_mask, mode="labeled")
        out = self.wrapper.forward_prompted(image, prompts)
        out["prompts"] = prompts
        return out

    def forward_unlabeled(self, image: torch.Tensor, teacher_prob: torch.Tensor, student_prob: torch.Tensor | None = None):
        if not self.available():
            return {"valid": False}
        prompts = self.prompt_generator(
            image=image,
            teacher_prob=teacher_prob.detach(),
            student_prob=student_prob.detach() if student_prob is not None else None,
            mode="unlabeled",
        )
        out = self.wrapper.forward_prompted(image, prompts)
        out["prompts"] = prompts
        return out

    def propose(self, images, teacher_prob, ids=None, num_classes: int | None = None):
        out = self.forward_unlabeled(images, teacher_prob)
        if "sam_prob" not in out or out["sam_prob"].shape[1] != (num_classes or self.num_classes):
            raise RuntimeError("Real SAM mentor did not return class-aligned sam_prob")
        return out

    def optimizer_param_groups(self, base_lr: float, sam_cfg: dict):
        groups = []
        if self.wrapper is not None:
            groups.extend(
                self.wrapper.parameter_groups(
                    lr_peft=float(sam_cfg.get("lr_peft", base_lr * 0.1)),
                    lr_mask_decoder=float(sam_cfg.get("lr_mask_decoder", sam_cfg.get("lr_peft", base_lr * 0.1))),
                    lr_prompt_encoder=float(sam_cfg.get("lr_prompt_encoder", sam_cfg.get("lr_peft", base_lr * 0.1))),
                )
            )
        prompt_params = [p for p in self.prompt_generator.parameters() if p.requires_grad]
        if prompt_params:
            groups.append({"params": prompt_params, "lr": float(sam_cfg.get("lr_prompt", base_lr)), "name": "prompt_generator"})
        return groups

    def trainability_report(self):
        if self.wrapper is None:
            return {}
        report = self.wrapper.trainability_report()
        prompt_params = sum(p.numel() for p in self.prompt_generator.parameters() if p.requires_grad)
        return {
            "total_sam_params": report.total_sam_params,
            "trainable_sam_params": report.trainable_sam_params,
            "trainable_sam_ratio": report.trainable_ratio,
            "trainable_sam_modules": report.trainable_module_names,
            "lora_param_count": report.lora_param_count,
            "adapter_param_count": report.adapter_param_count,
            "mask_decoder_trainable": report.mask_decoder_trainable,
            "prompt_encoder_trainable": report.prompt_encoder_trainable,
            "trainable_prompt_generator_params": int(prompt_params),
        }

    def sam_grad_norm(self):
        total = 0.0
        if self.wrapper is not None:
            for param in self.wrapper.sam.parameters():
                if param.requires_grad and param.grad is not None:
                    total += float(param.grad.detach().norm().cpu()) ** 2
        for param in self.prompt_generator.parameters():
            if param.requires_grad and param.grad is not None:
                total += float(param.grad.detach().norm().cpu()) ** 2
        return math.sqrt(total)

    def trainable_state_dict(self):
        state = {"prompt_generator": self.prompt_generator.state_dict(), "sam_trainable": {}}
        if self.wrapper is None:
            return state
        for name, param in self.wrapper.sam.named_parameters():
            if param.requires_grad:
                state["sam_trainable"][name] = param.detach().cpu()
        return state
