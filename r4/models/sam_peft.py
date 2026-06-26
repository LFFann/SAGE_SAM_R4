from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank residual branch."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling


class BottleneckAdapter(nn.Module):
    """Residual bottleneck adapter applied on the last feature dimension."""

    def __init__(self, dim: int, adapter_dim: int = 32, scale: float = 1.0):
        super().__init__()
        if dim <= 0:
            raise ValueError("Adapter input dim must be positive")
        if adapter_dim <= 0:
            raise ValueError("Adapter bottleneck dim must be positive")
        self.dim = int(dim)
        self.adapter_dim = int(adapter_dim)
        self.scale = float(scale)
        self.adapter_down = nn.Linear(self.dim, self.adapter_dim)
        self.adapter_act = nn.GELU()
        self.adapter_up = nn.Linear(self.adapter_dim, self.dim)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x: torch.Tensor):
        return x + self.scale * self.adapter_up(self.adapter_act(self.adapter_down(x)))


class BlockWithAdapter(nn.Module):
    """Wraps a SAM image-encoder block with a trainable residual adapter."""

    def __init__(self, block: nn.Module, dim: int, adapter_dim: int = 32, scale: float = 1.0):
        super().__init__()
        self.block = block
        self.adapter = BottleneckAdapter(dim=dim, adapter_dim=adapter_dim, scale=scale)

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if torch.is_tensor(out):
            return self._adapt_tensor(out)
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            return (self._adapt_tensor(out[0]), *out[1:])
        return out

    def _adapt_tensor(self, x: torch.Tensor):
        if x.shape[-1] != self.adapter.dim:
            raise RuntimeError(
                f"Adapter expected last dim {self.adapter.dim}, got tensor shape {tuple(x.shape)}"
            )
        return self.adapter(x)


@dataclass
class SAMTrainabilityReport:
    total_sam_params: int
    trainable_sam_params: int
    trainable_ratio: float
    trainable_module_names: list[str]
    lora_param_count: int
    adapter_param_count: int
    mask_decoder_trainable: bool
    prompt_encoder_trainable: bool


class SAMPEFTAdapter:
    """Freezes SAM by default and injects/enables PEFT modules."""

    def __init__(
        self,
        sam: nn.Module,
        train_peft: bool = True,
        peft_type: str = "adapter",
        train_mask_decoder: bool = True,
        train_prompt_encoder: bool = False,
        train_last_n_blocks: int = 0,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_target_modules: tuple[str, ...] = ("qkv", "proj"),
        adapter_dim: int = 32,
        adapter_scale: float = 1.0,
        max_trainable_ratio: float = 0.05,
        hard_max_trainable_ratio: float = 0.10,
    ):
        self.sam = sam
        self.train_peft = bool(train_peft)
        self.peft_type = str(peft_type).lower()
        self.train_mask_decoder = bool(train_mask_decoder)
        self.train_prompt_encoder = bool(train_prompt_encoder)
        self.train_last_n_blocks = int(train_last_n_blocks)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_target_modules = tuple(lora_target_modules)
        self.adapter_dim = int(adapter_dim)
        self.adapter_scale = float(adapter_scale)
        self.max_trainable_ratio = float(max_trainable_ratio)
        self.hard_max_trainable_ratio = float(hard_max_trainable_ratio)
        self.injected_lora_modules = 0
        self.injected_adapter_modules = 0
        self.enabled_existing_adapter_params = 0
        self.report = self.configure()

    def configure(self) -> SAMTrainabilityReport:
        for param in self.sam.parameters():
            param.requires_grad_(False)

        if self.train_peft:
            if "lora" in self.peft_type:
                self.injected_lora_modules = self.inject_lora_to_sam_image_encoder(
                    self.sam,
                    last_n_blocks=self.train_last_n_blocks,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    target_modules=self.lora_target_modules,
                )
            if "adapter" in self.peft_type:
                self.enabled_existing_adapter_params = self._enable_existing_adapter_parameters(self.train_last_n_blocks)
                self.injected_adapter_modules = self.inject_adapter_to_sam_image_encoder(
                    self.sam,
                    last_n_blocks=self.train_last_n_blocks,
                    adapter_dim=self.adapter_dim,
                    scale=self.adapter_scale,
                )

        if self.train_mask_decoder and hasattr(self.sam, "mask_decoder"):
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad_(True)
        if self.train_prompt_encoder and hasattr(self.sam, "prompt_encoder"):
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad_(True)

        report = self._make_report()
        if self.train_peft and (report.lora_param_count + report.adapter_param_count) == 0:
            raise RuntimeError("sam.train_peft=true but no LoRA/Adapter parameter was injected or found")
        if "lora" in self.peft_type and report.lora_param_count == 0:
            raise RuntimeError("sam.peft_type contains 'lora' but no LoRA parameter exists")
        if self.train_peft and "adapter" in self.peft_type and report.adapter_param_count == 0:
            raise RuntimeError("sam.peft_type contains 'adapter' but no Adapter parameter exists")
        if report.trainable_sam_params == 0 and (self.train_peft or self.train_mask_decoder or self.train_prompt_encoder):
            raise RuntimeError("SAM has zero trainable parameters after PEFT configuration")
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

    @staticmethod
    def inject_lora_to_sam_image_encoder(
        sam: nn.Module,
        last_n_blocks: int,
        rank: int,
        alpha: float,
        target_modules: tuple[str, ...] = ("qkv", "proj"),
    ) -> int:
        image_encoder = getattr(sam, "image_encoder", None)
        blocks = getattr(image_encoder, "blocks", None)
        if blocks is None:
            return 0
        selected_blocks = list(blocks)
        if last_n_blocks > 0:
            selected_blocks = selected_blocks[-int(last_n_blocks) :]
        injected = 0
        for block in selected_blocks:
            replacements = []
            for parent_name, parent in block.named_modules():
                if isinstance(parent, LoRALinear):
                    continue
                for child_name, child in list(parent.named_children()):
                    full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                    if not isinstance(child, nn.Linear):
                        continue
                    if not any(token in full_name for token in target_modules):
                        continue
                    if isinstance(child, LoRALinear):
                        continue
                    replacements.append((parent, child_name, child))
            for parent, child_name, child in replacements:
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                injected += 1
        return injected

    @staticmethod
    def inject_adapter_to_sam_image_encoder(
        sam: nn.Module,
        last_n_blocks: int,
        adapter_dim: int = 32,
        scale: float = 1.0,
    ) -> int:
        image_encoder = getattr(sam, "image_encoder", None)
        blocks = getattr(image_encoder, "blocks", None)
        if blocks is None:
            return 0
        num_blocks = len(blocks)
        start = max(0, num_blocks - int(last_n_blocks)) if last_n_blocks > 0 else 0
        injected = 0
        for idx in range(start, num_blocks):
            block = blocks[idx]
            if isinstance(block, BlockWithAdapter):
                continue
            dim = SAMPEFTAdapter._infer_block_dim(block)
            if dim is None:
                continue
            blocks[idx] = BlockWithAdapter(block, dim=dim, adapter_dim=adapter_dim, scale=scale)
            injected += 1
        return injected

    @staticmethod
    def _infer_block_dim(block: nn.Module) -> int | None:
        for module in block.modules():
            if isinstance(module, nn.LayerNorm) and module.normalized_shape:
                return int(module.normalized_shape[-1])
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and "qkv" in name.lower():
                return int(module.in_features)
        for module in block.modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                return int(module.in_features)
        return None

    def _enable_existing_adapter_parameters(self, last_n_blocks: int = 0) -> int:
        image_encoder = getattr(self.sam, "image_encoder", None)
        blocks = getattr(image_encoder, "blocks", None)
        adapter_params = 0
        if blocks is not None and last_n_blocks > 0:
            modules = list(blocks)[-last_n_blocks:]
        else:
            modules = [self.sam]
        for module in modules:
            for name, param in module.named_parameters():
                if "adapter" in name.lower():
                    param.requires_grad_(True)
                    adapter_params += param.numel()
        return int(adapter_params)

    def _make_report(self) -> SAMTrainabilityReport:
        total = sum(p.numel() for p in self.sam.parameters())
        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        modules = []
        lora_count = 0
        adapter_count = 0
        mask_trainable = False
        prompt_trainable = False
        for name, param in self.sam.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_count += param.numel()
            if "adapter" in name.lower() and param.requires_grad:
                adapter_count += param.numel()
            if param.requires_grad:
                modules.append(name.rsplit(".", 1)[0])
                mask_trainable = mask_trainable or name.startswith("mask_decoder.")
                prompt_trainable = prompt_trainable or name.startswith("prompt_encoder.")
        return SAMTrainabilityReport(
            total_sam_params=int(total),
            trainable_sam_params=int(trainable),
            trainable_ratio=float(trainable / max(1, total)),
            trainable_module_names=sorted(set(modules)),
            lora_param_count=int(lora_count),
            adapter_param_count=int(adapter_count),
            mask_decoder_trainable=bool(mask_trainable),
            prompt_encoder_trainable=bool(prompt_trainable),
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
