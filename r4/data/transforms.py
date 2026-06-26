from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def resize_image_mask(image: torch.Tensor, mask: torch.Tensor | None, size: int):
    image = F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    if mask is not None:
        mask = F.interpolate(mask[None, None].float(), size=(size, size), mode="nearest").squeeze(0).squeeze(0).long()
    return image, mask


def weak_transform(image: torch.Tensor, mask: torch.Tensor | None = None, random_flip: bool = True, **_):
    if random_flip and random.random() < 0.5:
        image = torch.flip(image, dims=[-1])
        if mask is not None:
            mask = torch.flip(mask, dims=[-1])
    return image, mask


def strong_transform(image: torch.Tensor, mask: torch.Tensor | None = None, gamma_range=(0.7, 1.5), gain_range=(0.75, 1.25), speckle_std_range=(0.0, 0.12), gaussian_blur_prob=0.0, patch_mask_prob=0.0, patch_mask_ratio=0.25, **_):
    # Keep strong views spatially aligned with the weak-view pseudo target.
    # Geometry belongs in weak_transform and must happen before target creation.
    gamma = random.uniform(*gamma_range)
    gain = random.uniform(*gain_range)
    image = torch.clamp((image.clamp_min(1e-4) ** gamma) * gain, 0.0, 1.0)
    std = random.uniform(*speckle_std_range)
    if std > 0:
        image = torch.clamp(image + image * torch.randn_like(image) * std, 0.0, 1.0)
    if patch_mask_prob > 0 and random.random() < patch_mask_prob:
        h, w = image.shape[-2:]
        ph = max(1, int(h * patch_mask_ratio))
        pw = max(1, int(w * patch_mask_ratio))
        y = random.randint(0, max(0, h - ph))
        x = random.randint(0, max(0, w - pw))
        image[..., y : y + ph, x : x + pw] = 0
    return image, mask
