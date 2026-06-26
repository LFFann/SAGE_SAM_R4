from __future__ import annotations

import torch

from r4.data.transforms import strong_transform, weak_transform


def make_weak_strong_views(image, pseudo_mask=None, strong_kwargs=None, weak_kwargs=None):
    if image.ndim == 4:
        weak, strong1, strong2 = [], [], []
        y1_list, y2_list = [], []
        for idx in range(image.shape[0]):
            mask_i = pseudo_mask[idx] if pseudo_mask is not None else None
            x_w, x_s1, x_s2, y_s1, y_s2 = make_weak_strong_views(
                image[idx],
                mask_i,
                strong_kwargs=strong_kwargs,
                weak_kwargs=weak_kwargs,
            )
            weak.append(x_w)
            strong1.append(x_s1)
            strong2.append(x_s2)
            if y_s1 is not None:
                y1_list.append(y_s1)
            if y_s2 is not None:
                y2_list.append(y_s2)
        y1 = torch.stack(y1_list, dim=0) if y1_list else None
        y2 = torch.stack(y2_list, dim=0) if y2_list else None
        return torch.stack(weak, dim=0), torch.stack(strong1, dim=0), torch.stack(strong2, dim=0), y1, y2

    x_w, y_w = weak_transform(image, pseudo_mask, **(weak_kwargs or {}))
    strong = dict(strong_kwargs or {})
    strong["random_flip"] = False
    x_s1, y_s1 = strong_transform(x_w.clone(), y_w.clone() if y_w is not None else None, **strong)
    x_s2, y_s2 = strong_transform(x_w.clone(), y_w.clone() if y_w is not None else None, **strong)
    return x_w, x_s1, x_s2, y_s1, y_s2
