from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptGenerator(nn.Module):
    """Trainable one-vs-rest mask-prompt generator for online SAM mentoring."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        mask_prompt_size: int = 256,
        min_component_area: int = 16,
        residual_scale: float = 0.15,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_foreground = max(0, self.num_classes - 1)
        self.mask_prompt_size = int(mask_prompt_size)
        self.min_component_area = int(min_component_area)
        self.residual_scale = float(residual_scale)
        refiner_in = 1 + 2 * self.num_foreground + 1
        hidden = max(16, min(64, refiner_in * 8))
        self.refiner = nn.Sequential(
            nn.Conv2d(refiner_in, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_foreground, kernel_size=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        teacher_prob: torch.Tensor | None = None,
        student_prob: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        mode: str = "unlabeled",
    ) -> dict[str, torch.Tensor]:
        if self.num_foreground == 0:
            raise ValueError("PromptGenerator requires at least one foreground class")
        if image.ndim != 4:
            raise ValueError("image must be BCHW")

        b, _, h, w = image.shape
        device = image.device
        if gt_mask is not None:
            target = gt_mask.clamp(0, self.num_classes - 1)
            base = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()[:, 1:]
            if teacher_prob is None:
                teacher_fg = base
                teacher_uncertainty = torch.zeros(b, 1, h, w, device=device)
            else:
                teacher_fg = teacher_prob.detach()[:, 1:]
                teacher_uncertainty = 1.0 - teacher_prob.detach().max(dim=1, keepdim=True).values
        else:
            if teacher_prob is None:
                raise ValueError("teacher_prob is required for unlabeled prompt generation")
            base = teacher_prob.detach()[:, 1:]
            teacher_fg = base
            teacher_uncertainty = 1.0 - teacher_prob.detach().max(dim=1, keepdim=True).values

        if student_prob is None:
            student_fg = torch.zeros_like(base)
        else:
            student_fg = student_prob.detach()[:, 1:]

        image_gray = image.detach().mean(dim=1, keepdim=True)
        refiner_in = torch.cat([image_gray, base, student_fg, teacher_uncertainty], dim=1)
        residual = torch.tanh(self.refiner(refiner_in))
        soft_prompt = (base + self.residual_scale * residual).clamp(0.0, 1.0)
        mask_prompt = F.interpolate(
            soft_prompt,
            size=(self.mask_prompt_size, self.mask_prompt_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(b * self.num_foreground, 1, self.mask_prompt_size, self.mask_prompt_size)

        prompt_quality_fg = self._quality(base, teacher_fg, gt_mask is not None)
        prompt_quality = image.new_ones((b, self.num_classes))
        prompt_quality[:, 1:] = prompt_quality_fg

        boxes, point_coords, point_labels, negative_point_coords = self._metadata_from_masks(soft_prompt.detach())
        image_index = torch.arange(b, device=device).repeat_interleave(self.num_foreground)
        class_ids = torch.arange(1, self.num_classes, device=device).repeat(b)
        return {
            "mask_prompt": mask_prompt,
            "soft_prompt": soft_prompt,
            "boxes_xyxy": boxes,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "negative_point_coords": negative_point_coords,
            "image_index": image_index,
            "class_ids": class_ids,
            "prompt_quality": prompt_quality,
            "mode": mode,
        }

    def _quality(self, base: torch.Tensor, teacher_fg: torch.Tensor, labeled: bool):
        area = base.mean(dim=(2, 3))
        if labeled:
            return (area * base.shape[-1] * base.shape[-2] >= self.min_component_area).float()
        denom = base.sum(dim=(2, 3)).clamp_min(1e-6)
        conf = (base * teacher_fg).sum(dim=(2, 3)) / denom
        area_ok = (area * base.shape[-1] * base.shape[-2] >= self.min_component_area).float()
        return (0.7 * conf + 0.3 * area_ok).clamp(0.0, 1.0)

    def _metadata_from_masks(self, masks: torch.Tensor):
        b, c, h, w = masks.shape
        boxes = []
        pos_points = []
        neg_points = []
        for bi in range(b):
            union_fg = masks[bi].max(dim=0).values
            for ci in range(c):
                m = masks[bi, ci] > 0.5
                if int(m.sum()) >= self.min_component_area:
                    yy, xx = torch.where(m)
                    x0 = xx.min().float() / max(1, w - 1)
                    x1 = xx.max().float() / max(1, w - 1)
                    y0 = yy.min().float() / max(1, h - 1)
                    y1 = yy.max().float() / max(1, h - 1)
                    px = xx.float().mean() / max(1, w - 1)
                    py = yy.float().mean() / max(1, h - 1)
                else:
                    x0 = y0 = torch.tensor(0.0, device=masks.device)
                    x1 = y1 = torch.tensor(1.0, device=masks.device)
                    px = py = torch.tensor(0.5, device=masks.device)
                bg = union_fg < 0.1
                if bg.any():
                    yy_bg, xx_bg = torch.where(bg)
                    nx = xx_bg.float().mean() / max(1, w - 1)
                    ny = yy_bg.float().mean() / max(1, h - 1)
                else:
                    nx = ny = torch.tensor(0.0, device=masks.device)
                boxes.append(torch.stack([x0, y0, x1, y1]))
                pos_points.append(torch.stack([px, py]).view(1, 2))
                neg_points.append(torch.stack([nx, ny]).view(1, 2))
        boxes_t = torch.stack(boxes, dim=0)
        points_t = torch.stack(pos_points, dim=0)
        neg_t = torch.stack(neg_points, dim=0)
        labels_t = torch.ones(points_t.shape[:2], device=masks.device, dtype=torch.long)
        return boxes_t, points_t, labels_t, neg_t
